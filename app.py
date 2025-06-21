import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np

# ----------------------------
# Define the Generator class
# ----------------------------
class Generator(nn.Module):
    def __init__(self, noise_dim=100, label_dim=10, output_dim=28*28):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim + label_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        x = torch.cat((noise, labels), dim=1)
        return self.model(x)

# ----------------------------
# Helper: One-hot encode digit
# ----------------------------
def one_hot(labels, num_classes=10):
    return torch.nn.functional.one_hot(labels, num_classes).float()

# ----------------------------
# Load the generator model
# ----------------------------
@st.cache_resource
def load_generator():
    device = torch.device("cpu")
    model = Generator().to(device)
    model.load_state_dict(torch.load("generator_model.pt", map_location=device))
    model.eval()
    return model

# ----------------------------
# App UI
# ----------------------------
st.title("🧠 Handwritten Digit Generator (GAN)")

digit = st.selectbox("Select a digit to generate (0–9):", list(range(10)))
if st.button("Generate 5 Images"):
    model = load_generator()
    noise = torch.randn(5, 100)
    labels = torch.full((5,), digit, dtype=torch.long)
    one_hot_labels = one_hot(labels)

    with torch.no_grad():
        images = model(noise, one_hot_labels).view(-1, 1, 28, 28)

    # Convert generated images to NumPy format
    fake_images = images.squeeze().detach().cpu().numpy()
    
    st.write("### Generated Samples")
    cols = st.columns(5)
    
    for i in range(5):
        img = fake_images[i]
        cols[i].image(img, caption=f"{digit}", use_column_width=True)


