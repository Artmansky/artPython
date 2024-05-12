import os as OS
import pandas as PD
import torch as TY
from torch.utils.data import Dataset as DATA

class ArtDataset(DATA):
    def __init__(self,csv_file,root_dir,transform=None):
        self.annotations = PD.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = OS.path.join(self.root_dir, self.annotations.iloc[index,0])
        
    