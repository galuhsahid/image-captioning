import logging
import os
import time
import pandas as pd
import datasets
import nltk
from filelock import FileLock
from dataclasses import dataclass, field
from datasets import Dataset, load_dataset, load_metric
from transformers.file_utils import is_offline_mode
from functools import partial

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
from flax.jax_utils import unreplicate
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
from vit_gpt2.modeling_flax_vit_gpt2_lm import FlaxViTGPT2LMForConditionalGeneration

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
        return path
        #return read_image(path, mode=ImageReadMode.RGB)

    def _load_target(self, idx):
        return self.captions[idx]

    def __getitem__(self, index: int):
        image = self._load_image(index)
        target = self._load_target(index)

        #if self.transforms is not None:
            #image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.captions)


@flax.struct.dataclass
class FlaxDataCollatorForImageLanguageModeling:
    model: FlaxViTGPT2LMForConditionalGeneration
    feature_extractor: ViTFeatureExtractor
    tokenizer: PreTrainedTokenizerBase
    vision_sequence_length: int = 50
    max_length: int = 256
    

    def __call__(self, examples) -> Dict[str, np.ndarray]:
        images = [Image.open(example[0]) for example in examples]
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
        #decoder_inputs = self.tokenizer(captions, max_length=self.max_length, padding="max_length", return_tensors="jax")
        with self.tokenizer.as_target_tokenizer():
            #self.tokenizer.pad_token = self.tokenizer.eos_token
            #self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            labels = self.tokenizer(
                captions, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="jax"
            )
        
        model_inputs = dict(labels)
        model_inputs['pixel_values'] = pixel_values
        model_inputs['labels'] = labels['input_ids']

        # check
        decoder_input_ids = shift_tokens_right_fn(
            jnp.array(labels["input_ids"]), 50265, 50265
        )

        model_inputs['input_ids'] = np.asarray(decoder_input_ids)

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
        default="val.tsv",
        metadata={"help": "An optional input evaluation data file (a tsv file)."},
    )
    predict_file: Optional[str] = field(
        default="test.tsv",
        metadata={"help": "An optional input predict data file (a tsv file)."},
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the `max_length` param of `model.generate`, which is used "
            "during evaluation."
        },
    )
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
        default=0,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    predict_with_generate: bool = field(
        default=False, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to `model.generate`, "
            "which is used during evaluation."
        },
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


class TrainState(train_state.TrainState):
    dropout_rng: jnp.ndarray

    def replicate(self):
        return jax_utils.replicate(self).replace(dropout_rng=shard_prng_key(self.dropout_rng))



def write_metric(summary_writer, train_metrics, eval_metrics, train_time, step):
    summary_writer.scalar("train_time", train_time, step)

    train_metrics = get_metrics(train_metrics)
    for key, vals in train_metrics.items():
        tag = f"train_{key}"
        for i, val in enumerate(vals):
            summary_writer.scalar(tag, val, step - len(vals) + i + 1)

    for metric_name, value in eval_metrics.items():
        summary_writer.scalar(f"eval_{metric_name}", value, step)

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

    try:
        nltk.data.find("tokenizers/punkt")
    except (LookupError, OSError):
        if is_offline_mode():
            raise LookupError(
                "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
            )
        with FileLock(".lock") as lock:
            nltk.download("punkt", quiet=True)


    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    
    # Model
    model = FlaxViTGPT2LMForConditionalGeneration.from_vit_gpt2_pretrained(
        vit_model_name_or_path='google/vit-base-patch16-224-in21k', 
        gpt2_model_name_or_path='flax-community/gpt2-small-indonesian'
    )
    
    config = model.config

    # set seed for torch dataloaders
    set_seed(training_args.seed)

    # Dataset
    preprocess = Transform(config.vit_config.image_size)
    preprocess = torch.jit.script(preprocess)

    # Initialize the image-text dataset
    if training_args.do_train:
        train_dataset = ImageTextDataset(
            data_args.data_dir,
            data_args.train_file,
            captions_per_image=2,
            transform=preprocess,
        )

    if training_args.do_eval:
        eval_dataset = ImageTextDataset(
            data_args.data_dir,
            data_args.validation_file,
            captions_per_image=1,
            transform=preprocess,
        )

    if training_args.do_predict:
        predict_dataset = ImageTextDataset(
            data_args.data_dir,
            data_args.prediction_file,
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
    
    image_column = 'image'
    caption_column = 'caption'
    pixel_values_column = 'pixel_values'



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
    if training_args.do_train:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=data_args.preprocessing_num_workers,
            #persistent_workers=True,
            drop_last=True,
            collate_fn=data_collator,
        )

    if training_args.do_eval:
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=data_args.preprocessing_num_workers,
            #persistent_workers=True,
            drop_last=True,
            collate_fn=data_collator,
        )

    if training_args.do_predict:
        pred_loader = torch.utils.data.DataLoader(
            predict_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=data_args.preprocessing_num_workers,
            #persistent_workers=True,
            drop_last=True,
            collate_fn=data_collator,
        )

    # Metric
    metric = load_metric("rouge")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(preds, labels):
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

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
    rng, dropout_rng = jax.random.split(rng)

    # Setup train state
    state = TrainState.create(
        apply_fn=model.__call__, params=model.params, tx=optimizer, dropout_rng=dropout_rng
    )

    # Train Step
    # label smoothed cross entropy
    def loss_fn(logits, labels, padding_mask, label_smoothing_factor=0.0):
        """
        The label smoothing implementation is adapted from Flax's official example:
        https://github.com/google/flax/blob/87a211135c6a377c8f29048a1cac3840e38b9da4/examples/wmt/train.py#L104
        """
        vocab_size = logits.shape[-1]
        confidence = 1.0 - label_smoothing_factor
        low_confidence = (1.0 - confidence) / (vocab_size - 1)
        normalizing_constant = -(
            confidence * jnp.log(confidence) + (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20)
        )
        soft_labels = onehot(labels, vocab_size, on_value=confidence, off_value=low_confidence)

        loss = optax.softmax_cross_entropy(logits, soft_labels)
        loss = loss - normalizing_constant

        # ignore padded tokens from loss
        loss = loss * padding_mask
        loss = loss.sum() / padding_mask.sum()
        return loss

    # Define gradient update step fn
    def train_step(state, batch, label_smoothing_factor=0.0):
        dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

        def compute_loss(params):
            labels = batch.pop("labels")
            logits = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True)[0]
            loss = loss_fn(logits, labels, batch["attention_mask"], label_smoothing_factor)
            return loss

        grad_fn = jax.value_and_grad(compute_loss)
        loss, grad = grad_fn(state.params)
        grad = jax.lax.pmean(grad, "batch")

        new_state = state.apply_gradients(grads=grad, dropout_rng=new_dropout_rng)

        metrics = {"loss": loss, "learning_rate": linear_decay_lr_schedule_fn(state.step)}
        metrics = jax.lax.pmean(metrics, axis_name="batch")

        return new_state, metrics

    # Define eval fn
    def eval_step(params, batch, label_smoothing_factor=0.0):
        labels = batch.pop("labels")
        logits = model(**batch, params=params, train=False)[0]
        loss = loss_fn(logits, labels, batch["attention_mask"], label_smoothing_factor)

        # summarize metrics
        metrics = {"loss": loss}
        metrics = jax.lax.pmean(metrics, axis_name="batch")
        return metrics

    # Define generation function
    max_length = (
        data_args.val_max_target_length if data_args.val_max_target_length is not None else model.config.max_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else model.config.num_beams
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    def generate_step(params, batch):
        model.params = params
        output_ids = model.generate(batch["pixel_values"], **gen_kwargs) #check
        return output_ids.sequences

    # Create parallel version of the train and eval step
    p_train_step = jax.pmap(
        partial(train_step, label_smoothing_factor=training_args.label_smoothing_factor), "batch", donate_argnums=(0,)
    )
    p_eval_step = jax.pmap(partial(eval_step, label_smoothing_factor=training_args.label_smoothing_factor), "batch")
    p_generate_step = jax.pmap(generate_step, "batch")

    # Replicate the train state on each device
    state = jax_utils.replicate(state)

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

        # Create sampling rng
        rng, input_rng = jax.random.split(rng)
        train_metrics = []

        steps_per_epoch = len(train_dataset) // train_batch_size

        train_step_progress_bar = tqdm(total=steps_per_epoch, desc="Training...", position=1, leave=False)
        # train
        for batch in train_loader:
            print(batch)
            batch = shard(batch)
            print(batch)
            


if __name__ == "__main__":
    main()
