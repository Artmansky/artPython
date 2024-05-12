import pandas as pd
import os
import torch

#Prepare Device
device = ("cuda" if torch.cuda.is_available() else "cpu")

train_df = pd.DataFrame(columns=["img_name","label"])
train_df["img_name"] = os.listdir("Data/")
for idx, i in enumerate(os.listdir("Data/")):
    if "baroque" in i:
        train_df["label"][idx] = 0
    if "cubism" in i:
        train_df["label"][idx] = 1
    if "popart" in i:
        train_df["label"][idx] = 2
    if "postimpressionism" in i:
        train_df["label"][idx] = 3

train_df.to_csv (r'train_csv.csv', index = False, header=True)