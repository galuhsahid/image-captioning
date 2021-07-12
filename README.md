# vit-gpt2

Sample run:

```
python run_vit_gpt2.py \
    --output_dir="testing" \
    --data_dir="data/Flicker8k_Dataset" \
    --train_file="data/train.tsv" \
    --validation_file="data/val.tsv" \
    --do_train --do_eval \
    --num_train_epochs="3" --max_seq_length 256 \
    --per_device_train_batch_size="64" \
    --per_device_eval_batch_size="64" \
    --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
    --overwrite_output_dir \
    --preprocessing_num_workers 32 \
```

[Model code](https://github.com/ydshieh/vit-gpt2)