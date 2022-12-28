import torch
#import streamlit as st
from torchvision import transforms
import torch
import json
from torch import nn
from torchvision import models
from PIL import Image
from torch.nn import functional as F
import gradio as gr


# load idx_to_class.json into dictionary
with open('class_to_idx.json', 'r') as f:
    class_to_idx = json.load(f)
    
idx_to_class = {v: k for k, v in class_to_idx.items()}

with open('gardenbirds_sorted.json', 'r') as f:
    common_to_latin = json.load(f)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(idx_to_class))
model.load_state_dict(torch.load('gardenbirds.pth', map_location=torch.device('cpu')))
model.eval()

data_transforms = {
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def predict(inp):
    # transform pil image to tensor
    inp = data_transforms['val'](inp).unsqueeze(0)
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
        #print(prediction)
        confidences = {list(gardenbirds_sorted.keys())[i]: float(prediction[i]) for i in range(26)}
    return confidences 

interface = gr.Interface(fn=predict, 
             inputs=gr.Image(type="pil"),
             outputs=gr.Label(num_top_classes=3),
             title="What garden bird is this?",
             examples=["parus_major.jpg", "kingfisher.jpg"])

interface.launch()