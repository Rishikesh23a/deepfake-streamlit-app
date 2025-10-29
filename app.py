# ==========================================
# Image Classification App with Streamlit
# ==========================================

# -------------------------
# 1. Import libraries
# -------------------------
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# -------------------------
# 2. Device setup
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# 3. Define the model
# -------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # placeholder for fc layers
        self.fc1 = None
        self.fc2 = None
        self.num_classes = num_classes

    def _get_flatten_dim(self, x):
        # Compute flatten size dynamically
        with torch.no_grad():
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            return x.view(x.size(0), -1).size(1)

    def forward(self, x):
        if self.fc1 is None:
            flatten_dim = self._get_flatten_dim(x)
            self.fc1 = nn.Linear(flatten_dim, 128).to(x.device)
            self.fc2 = nn.Linear(128, self.num_classes).to(x.device)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# -------------------------
# 4. Load trained model
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleCNN(num_classes=2)
model.to(device)

# Load checkpoint without strict checking
state_dict = torch.load("best_model.pth", map_location=device)
model.load_state_dict(state_dict, strict=False)  # strict=False ignores mismatched shapes
model.eval()

# -------------------------
# 5. Define transforms
# -------------------------
transform = transforms.Compose([
    transforms.Resize((112, 112)),  # exactly what you used in Colab
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# -------------------------
# 6. Define class names
# -------------------------
class_names = ['real', 'fake']  # replace with your actual class names

# -------------------------
# 7. Streamlit UI
# -------------------------
st.title("Image Classification App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Transform image
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Prediction
    outputs = model(img_tensor)
    probs = F.softmax(outputs, dim=1)
    pred_class = torch.argmax(probs, 1).item()
    confidence = probs[0][pred_class].item()

    # Show result
    st.write(f"Prediction: **{class_names[pred_class]}**")
    st.write(f"Confidence: **{confidence*100:.2f}%**")
