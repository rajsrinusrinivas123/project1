import streamlit as st
import numpy as np
import librosa
import os
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ==========================================================
# CONFIG
# ==========================================================
MODEL_PATH = "best_model.pth"    # path to saved weights
CLASSES_PATH = "classes.npy"     # saved label order: np.save("classes.npy", le.classes_)
SAMPLE_RATE = 22050
DURATION = 5.0
N_MELS = 128

# ==========================================================
# PAGE SETUP
# ==========================================================
st.set_page_config(page_title="Lung Sound Classifier", layout="centered")

st.title("🎙️ Lung Sound Disease Classifier")
st.write("Upload a **.wav** or **.mp3** lung sound to predict the likely respiratory disease.")

# ==========================================================
# MODEL DEFINITION
# ==========================================================
class CNNClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 27, 128)  # adjust if input size differs
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ==========================================================
# UTILITY FUNCTIONS
# ==========================================================
@st.cache_data(show_spinner=False)
def load_classes(path):
    if not os.path.exists(path):
        st.error(f"Classes file not found: {path}. Please save it as 'classes.npy'.")
        return None
    return np.load(path, allow_pickle=True).tolist()

@st.cache_resource(show_spinner=False)
def load_model(model_path, num_classes):
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}. Train and save as 'best_model.pth'.")
        return None
    model = CNNClassifier(num_classes)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

def extract_log_mel(audio_path, sr=SAMPLE_RATE, duration=DURATION, n_mels=N_MELS):
    y, sr = librosa.load(audio_path, sr=sr, duration=duration)
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    return log_mel.astype(np.float32)

def plot_spectrogram(log_mel):
    fig, ax = plt.subplots(figsize=(6, 3))
    img = ax.imshow(log_mel, origin="lower", aspect="auto")
    ax.set_xlabel("Time frames")
    ax.set_ylabel("Mel bins")
    ax.set_title("Log-Mel Spectrogram")
    plt.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.tight_layout()
    return fig

# ==========================================================
# LOAD CLASSES AND MODEL
# ==========================================================
classes = load_classes(CLASSES_PATH)
if classes is None:
    st.stop()

num_classes = len(classes)
model = load_model(MODEL_PATH, num_classes)
if model is None:
    st.stop()

# ==========================================================
# FILE UPLOAD UI
# ==========================================================
uploaded = st.file_uploader("Upload lung sound (.wav or .mp3)", type=["wav", "mp3"])

if uploaded is not None:
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    # Play audio
    st.audio(tmp_path)

    # Extract features
    log_mel = extract_log_mel(tmp_path)
    st.pyplot(plot_spectrogram(log_mel))

    # Prepare tensor
    x = torch.FloatTensor(log_mel).unsqueeze(0).unsqueeze(0)  # shape: (1,1,n_mels,time)

    # Predict
    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1).cpu().numpy()[0]

    # Get top prediction only
    top_idx = int(np.argmax(probs))
    top_class = classes[top_idx]
    top_conf = float(probs[top_idx])

    # Display result
    st.markdown(f"### 🩺 Prediction: **{top_class}**")
    st.markdown(f"**Confidence:** {top_conf*100:.2f}%")

    # Optional: add a warning for low confidence
    if top_conf < 0.6:
        st.warning("⚠️ Low confidence prediction — this may not be a valid lung recording.")
