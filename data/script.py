import pandas as pd
from train import TRAIN
from val import VAL

train_dict = TRAIN

train_files = []
train_captions = []

for k, v in train_dict.items():
    for cap in v:
        train_files.append(k)
        train_captions.append(cap)

train_df = pd.DataFrame()
train_df["image_file"] = train_files
train_df["caption"] = train_captions

train_df.to_csv("train.csv", index=False, sep="\t")

# val

val_dict = VAL

val_files = []
val_captions = []

for k, v in val_dict.items():
    for cap in v:
        val_files.append(k)
        val_captions.append(cap)

val_df = pd.DataFrame()
val_df["image_file"] = val_files
val_df["caption"] = val_captions

val_df.to_csv("val.csv", index=False, sep="\t")