import tkinter as TK
from tkinter import filedialog as FD
from tkinter import messagebox as MS
import torch as TO
import torch.nn as NN
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import timm as TIM
import pickle as PK
from PIL import Image as IM
import matplotlib.pyplot as PLT

class artStylesDataset(Dataset):
    def __init__(self,data_dir,transform=None):
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.data[idx]

    @property
    def classes(self):
        return self.data.classes

class ArtsClassifier(NN.Module):
    def __init__(self, num_classes=6):
        super(ArtsClassifier,self).__init__()
        self.base_model = TIM.create_model("efficientnet_b0", pretrained=True)
        self.features = NN.Sequential(*list(self.base_model.children())[:-1])
        
        enet_out_size = 1280
        self.classifier = NN.Linear(enet_out_size,num_classes)
        

    def forward(self,x):
        x = self.features(x)
        output = self.classifier(x)
        return output
    
def preprocess_image(image_path, transform):
    image = IM.open(image_path).convert("RGB")
    return image, transform(image).unsqueeze(0)

def predict(model, image_tensor, device):
    model.eval()
    with TO.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = TO.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy().flatten()

def visualize_predictions(original_image, probabilities, class_names, picture_name):
    fig, axarr = PLT.subplots(1, 2, figsize=(14, 7))
    
    axarr[0].imshow(original_image)
    axarr[0].text(35,-15,picture_name)
    axarr[0].axis("off")
    
    axarr[1].barh(class_names, probabilities)
    axarr[1].set_xlabel("Probability")
    axarr[1].set_title("Era Prediction")
    axarr[1].set_xlim(0, 1)

    PLT.tight_layout()
    PLT.show()

def show_result():
    try:
        test_image = FD.askopenfilename(initialdir = "./",title = "Select Your image",filetypes = (("PNG Files","*.png"),))

        original_image, image_tensor = preprocess_image(test_image, transform)
        probabilities = predict(model, image_tensor, device)

        class_names = trainDataset.classes 
        visualize_predictions(original_image, probabilities, class_names, test_image)
    except:
        MS.showerror("Invalid File","Can't decide? Try again")
        pass
        
model_path = FD.askopenfilename(initialdir = "./",title = "Select Model File",filetypes = (("Pickle Files","*.pkl"),))
if model_path[-3:].lower() == "pkl":
    isCorrect = True
else: exit()

device = TO.device("cuda:0" if TO.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((464,300)),
    transforms.ToTensor(),
])

trainDataset = artStylesDataset("./Data", transform=transform)
model = ArtsClassifier(num_classes=6)
model.to(device)

try:
    with open(model_path, 'rb') as file:  
        model = PK.load(file)
except:
    MS.showerror("Invalid File","Pickle file corrupted, quitting!")
    exit()

window = TK.Tk()
window.title("ArtPython")
window.geometry("250x250")

plot_button = TK.Button(master = window,command = show_result,height = 2, width = 10, text = "Select Image")
plot_button.place(relx=0.5,rely=0.5,anchor="center")

window.mainloop()