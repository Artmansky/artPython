import pandas as PD
import os as OS
import torch as TO

#Set Devices in GPU or CPU state based on hardware capabilities
device = ("cuda" if TO.cuda.is_available() else "cpu")

train_df = PD.DataFrame(columns=["img_name","label"])
train_df["img_name"] = OS.listdir("train/")
for idx, i in enumerate(OS.listdir("train/")):
    if "cat" in i:
        train_df["label"][idx] = 0
    if "dog" in i:
        train_df["label"][idx] = 1

train_df.to_csv (r'train_csv.csv', index = False, header=True)