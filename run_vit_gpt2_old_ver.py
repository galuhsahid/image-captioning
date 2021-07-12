import logging
import os
import time
import pandas as pd
from dataclasses import dataclass, field

# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.
from pathlib import Path
from typing import Dict, Optional, Tuple, Callable

import numpy as np
from tqdm import tqdm


from torchvision.transforms.functional import InterpolationMode
import flax
from flax.training.common_utils import get_metrics, shard, shard_prng_key
import jax
import jax.numpy as jnp
import optax
from flax import jax_utils, traverse_util
from flax.training import train_state
import torch
from torchvision.datasets import VisionDataset
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode
from flax.training.common_utils import get_metrics, onehot, shard
from transformers import (
    HfArgumentParser,
    PreTrainedTokenizerBase,
    TensorType,
    TrainingArguments,
    is_tensorboard_available,
    set_seed,
)

from transformers import ViTFeatureExtractor, GPT2Tokenizer
from vit_gpt2.configuration_vit_gpt2 import ViTGPT2Config
from vit_gpt2.modeling_flax_vit_gpt2 import FlaxViTGPT2ForConditionalGeneration

from transformers import ViTFeatureExtractor
from PIL import Image
import requests
import numpy as np



class ImageTextDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        file_path: str,
        captions_per_image=2,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(root, transforms, transform, target_transform)

        examples = pd.read_csv(file_path, sep="\t")

        self.image_paths = [f"data/{img}" for img in examples["image_file"].values]
        self.captions = examples["caption"].values

    def _load_image(self, idx: int):
        path = self.image_paths[idx]
        return read_image(path, mode=ImageReadMode.RGB)

    def _load_target(self, idx):
        return self.captions[idx]

    def __getitem__(self, index: int):
        image = self._load_image(index)
        target = self._load_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.captions)


@flax.struct.dataclass
class FlaxDataCollatorForImageLanguageModeling:
    model: FlaxViTGPT2ForConditionalGeneration
    feature_extractor: ViTFeatureExtractor
    tokenizer: PreTrainedTokenizerBase
    vision_sequence_length: int = 50
    max_length: int = 256
    

    def __call__(self, examples) -> Dict[str, np.ndarray]:
        images = [example[0] for example in examples]
        captions = [example[1] for example in examples]

        # In Flax, for seq2seq models we need to pass `decoder_input_ids`
        # as the Flax models don't accept `labels`, we need to prepare the decoder_input_ids here
        # for that dynamically import the `shift_tokens_right` function from the model file
        model_module = __import__(self.model.__module__, fromlist=["shift_tokens_tight"])
        shift_tokens_right_fn = getattr(model_module, "shift_tokens_right")

        # Encode
        encoder_inputs = self.feature_extractor(images=images, return_tensors="jax")
        pixel_values = encoder_inputs.pixel_values

        # Decode
        # Handle dict or lists with proper padding and conversion to tensor.
        labels = self.tokenizer(captions, max_length=self.max_length, padding="max_length", return_tensors="jax")
        #with self.tokenizer.as_target_tokenizer():
            #self.tokenizer.pad_token = self.tokenizer.eos_token
            #self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            #labels = self.tokenizer(
                #captions, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="jax"
            #)
        
        model_inputs = dict(labels)
        model_inputs['pixel_values'] = pixel_values
        model_inputs['labels'] = labels['input_ids']

        # check
        '''decoder_input_ids = shift_tokens_right_fn(
            jnp.array(labels["input_ids"]), 50265, 50265
        )'''

        #model_inputs['input_ids'] = np.asarray(decoder_input_ids)

         # We need decoder_attention_mask so we can ignore pad tokens from loss
        model_inputs["attention_mask"] = labels["attention_mask"]

        return model_inputs


# Args
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    vit_vision_name_or_path: Optional[str] = field(
        default="google/vit-base-patch16-224-in21k",
        metadata={"help": "The ViT model checkpoint for weights initialization."},
    )

    gpt2_name_or_path: Optional[str] = field(
        default="flax-community/gpt2-small-indonesian",
        metadata={"help": "The gpt2 model checkpoint for weights initialization."},
    )

    gpt2_tokenizer_name: Optional[str] = field(
        default="flax-community/gpt2-small-indonesian",
        metadata={
            "help": "Pretrained gpt2 tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": "Floating-point format in which the model weights should be initialized and trained. Choose one of `[float32, float16, bfloat16]`."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: Optional[str] = field(
        default="./images/",
        metadata={"help": "The data directory containing input files."},
    )
    train_file: Optional[str] = field(
        default="train.tsv",
        metadata={"help": "The input training data file (a tsv file)."},
    )
    validation_file: Optional[str] = field(
        default="test.tsv",
        metadata={"help": "An optional input evaluation data file (a tsv file)."},
    )
    # image_size: Optional[int] = field(default=224, metadata={"help": " The size (resolution) of each image."}) # TODO: Check

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )

    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )

    max_seq_length: Optional[int] = field(
        default=64,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated. Default to the max input length of the model."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=4,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    def __post_init__(self):
        if self.train_file is None and self.validation_file is None:
            raise ValueError("Need both training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension == "tsv", "`train_file` should be a tsv."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension == "tsv", "`validation_file` should be a tsv."


# We use torchvision for faster image pre-processing.
# We need to ensure faster processing speed as it can become a bottleneck on TPU
class Transform(torch.nn.Module):
    def __init__(self, image_size):
        super().__init__()
        self.transforms = torch.nn.Sequential(
            Resize([image_size], interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_size),
            ConvertImageDtype(torch.float),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.transforms(x)
        return x


def create_learning_rate_fn(
    train_ds_size: int,
    train_batch_size: int,
    num_train_epochs: int,
    num_warmup_steps: int,
    learning_rate: float,
) -> Callable[[int], jnp.array]:
    """Returns a linear warmup, linear_decay learning rate function."""
    steps_per_epoch = train_ds_size // train_batch_size
    num_train_steps = steps_per_epoch * num_train_epochs
    warmup_fn = optax.linear_schedule(
        init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps
    )
    decay_fn = optax.linear_schedule(
        init_value=learning_rate,
        end_value=0,
        transition_steps=num_train_steps - num_warmup_steps,
    )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps]
    )
    return schedule_fn


def write_train_metric(summary_writer, train_metrics, train_time, step):
    summary_writer.scalar("train_time", train_time, step)

    train_metrics = get_metrics(train_metrics)
    for key, vals in train_metrics.items():
        tag = f"train_{key}"
        for i, val in enumerate(vals):
            summary_writer.scalar(tag, val, step - len(vals) + i + 1)


def write_eval_metric(summary_writer, eval_metrics, step):
    for metric_name, value in eval_metrics.items():
        summary_writer.scalar(f"eval_{metric_name}", value, step)

    
def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )

    model_args, data_args, training_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Log on each process the small summary:
    logger = logging.getLogger(__name__)

    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Model
    model = FlaxViTGPT2ForConditionalGeneration.from_vit_gpt2_pretrained(
        'google/vit-base-patch16-224-in21k', 'flax-community/gpt2-small-indonesian'
    )
    
    config = model.config

    # set seed for torch dataloaders
    set_seed(training_args.seed)

    # Dataset
    preprocess = Transform(config.vit_config.image_size)
    preprocess = torch.jit.script(preprocess)

    # Initialize the image-text dataset
    train_dataset = ImageTextDataset(
        data_args.data_dir,
        data_args.train_file,
        captions_per_image=2,
        transform=preprocess,
    )

    eval_dataset = ImageTextDataset(
        data_args.data_dir,
        data_args.validation_file,
        captions_per_image=1,
        transform=preprocess,
    )

    # Tokenizer
    if model_args.gpt2_tokenizer_name:
        tokenizer = GPT2Tokenizer.from_pretrained(
            model_args.gpt2_tokenizer_name,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            pad_token="<PAD>"
        )
        #tokenizer.pad_token = tokenizer.eos_token
        #tokenizer.pad_token_id = tokenizer.eos_token_id

    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # Feature extractor
    if model_args.vit_vision_name_or_path:
        feature_extractor = ViTFeatureExtractor.from_pretrained(
                            model_args.vit_vision_name_or_path,
                            cache_dir=model_args.cache_dir,
                             use_fast=model_args.use_fast_tokenizer,
        )
        
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    
    # Data Collator
    data_collator = FlaxDataCollatorForImageLanguageModeling(model=model, tokenizer=tokenizer, feature_extractor=feature_extractor)

    # Store some constant
    num_epochs = int(training_args.num_train_epochs)
    train_batch_size = (
        int(training_args.per_device_train_batch_size) * jax.device_count()
    )
    eval_batch_size = int(training_args.per_device_eval_batch_size) * jax.device_count()
    steps_per_epoch = len(train_dataset) // train_batch_size
    total_train_steps = steps_per_epoch * num_epochs

    # Create learning rate schedule
    linear_decay_lr_schedule_fn = create_learning_rate_fn(
        len(train_dataset),
        train_batch_size,
        training_args.num_train_epochs,
        training_args.warmup_steps,
        training_args.learning_rate,
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=data_args.preprocessing_num_workers,
        persistent_workers=True,
        drop_last=True,
        collate_fn=data_collator,
    )

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=data_args.preprocessing_num_workers,
        persistent_workers=True,
        drop_last=True,
        collate_fn=data_collator,
    )

    # Enable tensorboard only on the master node
    has_tensorboard = is_tensorboard_available()
    if has_tensorboard and jax.process_index() == 0:
        try:
            from flax.metrics.tensorboard import SummaryWriter

            summary_writer = SummaryWriter(log_dir=Path(training_args.output_dir))
        except ImportError as ie:
            has_tensorboard = False
            logger.warning(
                f"Unable to display metrics through TensorBoard because some package are not installed: {ie}"
            )
    else:
        logger.warning(
            "Unable to display metrics through TensorBoard because the package is not installed: "
            "Please run pip install tensorboard to enable."
        )
    
    optimizer = optax.adamw(
        learning_rate=linear_decay_lr_schedule_fn,
        b1=training_args.adam_beta1,
        b2=training_args.adam_beta2,
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
    )

    # State
    # Initialize our training
    rng = jax.random.PRNGKey(training_args.seed)
    dropout_rngs = jax.random.split(rng, jax.local_device_count())

    # Setup train state
    state = train_state.TrainState.create(
        apply_fn=model.__call__, params=model.params, tx=optimizer
    )

    # Train Step

    # Define gradient update step fn
    def train_step(state, batch, dropout_rng):
        dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

        def loss_fn(params):
            labels = batch.pop("labels")

            logits = state.apply_fn(
                **batch, params=params, dropout_rng=dropout_rng, train=True
            )[0]

            label_mask = jnp.where(labels > 0, 1.0, 0.0)
            loss = (
                optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1]))
            )

            # take average
            loss = loss.sum() / label_mask.sum()

            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grad = grad_fn(state.params)
        grad = jax.lax.pmean(grad, "batch")
        new_state = state.apply_gradients(grads=grad)

        metrics = jax.lax.pmean(
            {"loss": loss, "learning_rate": linear_decay_lr_schedule_fn(state.step)},
            axis_name="batch",
        )

        return new_state, metrics, new_dropout_rng

    # Create parallel version of the train step
    p_train_step = jax.pmap(train_step, "batch", donate_argnums = (0,))

    # Eval Step

    # Define eval fn
    def eval_step(params, batch):
        labels = batch.pop("labels")

        logits = model(**batch, params=params, train=False)[0]

        # compute loss, ignore padded input tokens
        label_mask = jnp.where(labels > 0, 1.0, 0.0)
        loss = (
            optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1]))
            * label_mask
        )

        # compute accuracy
        accuracy = jnp.equal(jnp.argmax(logits, axis=-1), labels) * label_mask

        # summarize metrics
        metrics = {
            "loss": loss.sum(),
            "accuracy": accuracy.sum(),
            "normalizer": label_mask.sum(),
        }
        metrics = jax.lax.psum(metrics, axis_name="batch")

        return metrics

    p_eval_step = jax.pmap(eval_step, "batch")

    # Replicate the train state on each device
    state = jax_utils.replicate(state)

    # Train Loop
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel & distributed) = {train_batch_size}")
    logger.info(f"  Total optimization steps = {total_train_steps}")

    train_time = 0
    epochs = tqdm(range(num_epochs), desc=f"Epoch ... (1/{num_epochs})", position=0)
    for epoch in epochs:
        # ======================== Training ================================
        train_start = time.time()
        train_metrics = []

        # Create sampling rng
        rng, input_rng = jax.random.split(rng)

        # Generate an epoch by shuffling sampling indices from the train dataset
        num_train_samples = len(train_dataset)

        epochs = tqdm(range(num_epochs), desc=f"Epoch : (1/{num_epochs})", position=1)

        # Gather the indexes for creating the batch and do a training step
        for step, batch in enumerate(train_loader):
            batch = shard(batch)
            state, train_metric, dropout_rngs = p_train_step(state, batch, dropout_rngs)
            train_metrics.append(train_metric)

            cur_step = epoch * (num_train_samples // train_batch_size) + step

            if cur_step % training_args.logging_steps == 0 and cur_step > 0:
                # Save metrics
                train_metric = jax_utils.unreplicate(train_metric)
                train_time += time.time() - train_start
                if has_tensorboard and jax.process_index() == 0:
                    write_train_metric(summary_writer, train_metrics, train_time, cur_step)

                epochs.write(
                    f"Log at Step: ({cur_step} | Loss: {train_metric['loss']}, Learning Rate: {train_metric['learning_rate']})"
                )

                train_metrics = []

            if cur_step % training_args.eval_steps == 0 and cur_step > 0:
                # ======================== Evaluating ==============================
                num_eval_samples = len(eval_dataset)
                # eval_samples_idx = jnp.arange(num_eval_samples)
                # eval_batch_idx = generate_batch_splits(eval_samples_idx, eval_batch_size)

                eval_metrics = []
                eval_steps = len(eval_dataset) // eval_batch_size
                eval_step_progress_bar = tqdm(total=eval_steps, desc="Evaluating...", position=2, leave=False)
                for batch in eval_loader:

                    # Model forward
                    batch = shard(batch)
                    metrics = p_eval_step(state.params, batch)
                    eval_metrics.append(metrics)

                # normalize eval metrics
                eval_metrics = get_metrics(eval_metrics)
                eval_metrics = jax.tree_map(jnp.sum, eval_metrics)
                eval_normalizer = eval_metrics.pop("normalizer")
                eval_metrics = jax.tree_map(lambda x: x / eval_normalizer, eval_metrics)

                # Update progress bar
                epochs.desc = f"Eval at Step: ({cur_step} | Loss: {eval_metrics['loss']}, Acc: {eval_metrics['accuracy']})"

                # Save metrics
                if has_tensorboard and jax.process_index() == 0:
                    write_eval_metric(summary_writer, eval_metrics, cur_step)

            if cur_step % training_args.save_steps == 0 and cur_step > 0:
                # save checkpoint after each epoch and push checkpoint to the hub
                if jax.process_index() == 0:
                    params = jax.device_get(jax.tree_map(lambda x: x[0], state.params))
                    model.save_pretrained(
                        training_args.output_dir,
                        params=params,
                        push_to_hub=training_args.push_to_hub,
                        commit_message=f"Saving weights and logs of step {cur_step}",
                    )

if __name__ == "__main__":
    main()
