import torch as TO
import torchvision as TV
from customDataset import ArtDataset

device = ("cuda" if TO.cuda.is_available() else "cpu")

