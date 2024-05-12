import os as OS
import pandas as PD
import torch as TY
from torch.utils.data import Dataset as DATA
from PIL import Image as IM

class ArtDataset(DATA):
    def __init__(self,csv_file,root_dir,transform=None):
        self.annotations = PD.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_id = self.annotations.iloc[index,0]
        img = IM.open(OS.path.join(self.root_dir,img_id)).convert("RGB")
        y_label = TY.tensor(float(self.annotations.iloc[index,1]))

        if self.transform is not None:
            img = self.transform(img)

        return (img,y_label)
    