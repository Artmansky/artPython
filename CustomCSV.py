import pandas as PD
import os as OS

train_df = PD.DataFrame(columns=["img_name","label"])
train_df["img_name"] = OS.listdir("Data/")
for idx, i in enumerate(OS.listdir("Data/")):
    if "baroque" in i:
        train_df["label"][idx] = 0
    if "cubism" in i:
        train_df["label"][idx] = 1
    if "popart" in i:
        train_df["label"][idx] = 2
    if "postimpressionism" in i:
        train_df["label"][idx] = 3

train_df.to_csv (r'train_csv.csv', index = False, header=True)