import sounddevice as sd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import queue
import sys
from pathlib import Path

# --- CONFIGURATION ---
# UPDATE THIS PATH TO YOUR MODEL
MODEL_PATH = Path("C:/Users/egrandjean/Desktop/BattleSound_model/BattleSound/models/best_model.pt")

# AUDIO SETTINGS
SYSTEM_RATE = 48000  # Your device sample rate (check Windows settings if unsure)
MODEL_RATE = 16000  # Rate the model was trained on
DURATION = 0.5  # Seconds
CHUNK_SIZE = int(SYSTEM_RATE * DURATION)

# *** IMPORTANT: MEL SPECTROGRAM SETTINGS ***
# Based on your TFLite file "128_x_64", your model likely uses 64 Mel Bands.
# If detection is still bad, try changing this to 128.
N_MELS = 64

CLASS_LABELS = {0: "Nothing", 1: "Voices", 2: "Effects"}

# SENSITIVITY SETTINGS
GAIN_FACTOR = 1.0  # 1.0 = Default. Increase (e.g., 5.0) ONLY if mic is very quiet.
SILENCE_THRESHOLD = 0.01  # If audio is below this volume, force prediction to "Nothing"

data_queue = queue.Queue()


# --- MODEL DEFINITION ---
class Conv2DNet(nn.Module):
    def __init__(self, num_class=3):
        super(Conv2DNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 10, 5, 1, 2), nn.BatchNorm2d(10), nn.ReLU(), nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(nn.Conv2d(10, 20, 5, 1, 2), nn.BatchNorm2d(20), nn.ReLU(), nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(nn.Conv2d(20, 40, 5, 1, 2), nn.BatchNorm2d(40), nn.ReLU(), nn.MaxPool2d(2))

        # We initialize fc1 lazily (on first run) to handle shape mismatches automatically
        self.fc1 = None
        self.fc2 = nn.Linear(256, num_class)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.layer3(self.layer2(self.layer1(x)))
        x = x.view(x.size(0), -1)

        # Auto-detect input size for the Linear layer
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.shape[1], 256).to(x.device)

        x = self.fc2(self.dropout(F.relu(self.fc1(x))))
        return x


# --- SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")

model = Conv2DNet(3).to(device)

# --- ROBUST MODEL LOADING (Fixes the 'List' Error) ---
try:
    # weights_only=False fixes the security warning, allowing complex objects
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    # 1. Handle List-based Checkpoints (This fixes your crash)
    if isinstance(ckpt, list):
        print(f"Detected List-based checkpoint (len={len(ckpt)}). Using index 0 as weights.")
        state_dict = ckpt[0]
    # 2. Handle Dict-based Checkpoints
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    # Filter shapes to allow flexible loading (ignores mismatched layers)
    model_dict = model.state_dict()
    filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

    if len(filtered_dict) == 0:
        print("WARNING: No matching layers found! Model might be random.")
    else:
        print(f"Successfully loaded {len(filtered_dict)} layers.")

    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)

except Exception as e:
    print(f"\nCRITICAL ERROR LOADING MODEL: {e}")
    print("Tip: Ensure 'MODEL_PATH' points to the correct .pt file.")
    sys.exit(1)

model.eval()

# PREPROCESSING
resampler = torchaudio.transforms.Resample(orig_freq=SYSTEM_RATE, new_freq=MODEL_RATE).to(device)
transform = torchaudio.transforms.MelSpectrogram(sample_rate=MODEL_RATE, n_fft=1024, hop_length=256, n_mels=N_MELS).to(
    device
)
to_db = torchaudio.transforms.AmplitudeToDB().to(device)


def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    data_queue.put(indata[:, 0].copy())


def main():
    # VB CABLE SELECTION
    devices = sd.query_devices()
    input_device = sd.default.device[0]

    # Attempt to auto-find VB Cable
    for i, dev in enumerate(devices):
        if "CABLE Output" in dev["name"]:
            input_device = i
            print(f"Found VB Cable at index {i}")
            break

    print(f"Listening on device index {input_device}...")

    with sd.InputStream(
        device=input_device, callback=audio_callback, channels=1, samplerate=SYSTEM_RATE, blocksize=CHUNK_SIZE
    ):
        while True:
            try:
                audio = data_queue.get()

                # 1. Silence Gate (Prevents "Effect" detection on background noise)
                rms = np.sqrt(np.mean(audio**2))
                if rms < SILENCE_THRESHOLD:
                    print(f"\rSilence ({rms:.4f})  ", end="", flush=True)
                    continue

                # 2. Preprocessing
                tens = torch.tensor(audio).float().to(device) * GAIN_FACTOR
                tens = resampler(tens)
                spec = transform(tens)
                spec = to_db(spec)

                # 3. NO NORMALIZATION
                # We removed the (spec - mean) / std line.
                # This prevents silence from being blown up into "loud noise".

                # Add batch & channel dims [1, 1, H, W]
                spec = spec.unsqueeze(0).unsqueeze(0)

                # 4. Inference
                with torch.no_grad():
                    logits = model(spec)
                    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                    pred = np.argmax(probs)

                # Output
                if pred != 0:  # Only print detections (Voices/Effects)
                    print(f"\n>>> DETECTED: {CLASS_LABELS[pred]} ({probs[pred]:.2f})")
                else:
                    print(f"\rNothing ({probs[0]:.2f})  ", end="", flush=True)

            except KeyboardInterrupt:
                break
            except Exception as e:
                # print(e) # Uncomment for debug
                pass


if __name__ == "__main__":
    main()


## script classes almost every sound as effect
# import sounddevice as sd
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchaudio
# import matplotlib.pyplot as plt
# import queue
# import sys
# from collections import deque
# from pathlib import Path

# # --- CONFIGURATION ---
# MODEL_PATH = Path("C:/Users/egrandjean/Desktop/BattleSound_model/BattleSound/models/best_model.pt")

# # AUDIO SETTINGS
# SYSTEM_RATE = 48000  # Your device rate
# MODEL_RATE = 16000  # Rate the model was trained on
# DURATION = 0.5  # Seconds
# CHUNK_SIZE = int(SYSTEM_RATE * DURATION)

# # *** CRITICAL: MATCH THIS TO YOUR TRAINING ***
# # If you used the default BattleSound config, this is likely 64 or 128, not 40.
# # The TFLite filename "128_x_64" suggests n_mels=64 and time_steps=128.
# N_MELS = 64  # Try 64 or 128 if 40 fails.

# CLASS_LABELS = {0: "Nothing", 1: "Voices", 2: "Effects"}
# GAIN_FACTOR = 5.0
# SILENCE_THRESHOLD = 0.01  # Ignore silence to prevent false positives

# data_queue = queue.Queue()


# # --- MODEL DEFINITION ---
# class Conv2DNet(nn.Module):
#     def __init__(self, num_class=3):
#         super(Conv2DNet, self).__init__()
#         self.layer1 = nn.Sequential(nn.Conv2d(1, 10, 5, 1, 2), nn.BatchNorm2d(10), nn.ReLU(), nn.MaxPool2d(2))
#         self.layer2 = nn.Sequential(nn.Conv2d(10, 20, 5, 1, 2), nn.BatchNorm2d(20), nn.ReLU(), nn.MaxPool2d(2))
#         self.layer3 = nn.Sequential(nn.Conv2d(20, 40, 5, 1, 2), nn.BatchNorm2d(40), nn.ReLU(), nn.MaxPool2d(2))

#         # NOTE: If you change N_MELS, you must re-calculate this linear layer size.
#         # For 64 mels: output is likely different than 1200.
#         # This wrapper handles shape mismatch gracefully.
#         self.fc1 = None
#         self.fc2 = nn.Linear(256, num_class)
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, x):
#         x = self.layer3(self.layer2(self.layer1(x)))
#         x = x.view(x.size(0), -1)

#         # Lazy initialization for FC1 to handle different input sizes automatically
#         if self.fc1 is None:
#             self.fc1 = nn.Linear(x.shape[1], 256).to(x.device)

#         x = self.fc2(self.dropout(F.relu(self.fc1(x))))
#         return x


# # --- SETUP ---
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Running on {device}")

# model = Conv2DNet(3).to(device)

# try:
#     # Allow loading of complex objects (fixes the FutureWarning)
#     ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)

#     # 1. HANDLE LISTS (The fix for your error)
#     if isinstance(ckpt, list):
#         print(f"Checkpoint is a list with {len(ckpt)} items. Assuming index 0 is the model weights.")
#         state_dict = ckpt[0]
#     # 2. HANDLE DICTIONARIES (Standard format)
#     elif isinstance(ckpt, dict) and "model" in ckpt:
#         state_dict = ckpt["model"]
#     else:
#         state_dict = ckpt

#     # Double check we actually extracted a dictionary
#     if not isinstance(state_dict, dict):
#         raise TypeError(f"Failed to extract state_dict. Got type: {type(state_dict)}")

#     # Filter out layers that don't match (e.g. if you changed input size)
#     model_dict = model.state_dict()
#     filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

#     if len(filtered_dict) == 0:
#         print("WARNING: No overlapping layers found! The model structure might be completely different.")

#     model_dict.update(filtered_dict)
#     model.load_state_dict(model_dict)
#     print("Model loaded successfully.")

# except Exception as e:
#     print(f"CRITICAL ERROR loading model: {e}")
#     sys.exit(1)

# model.eval()

# # PREPROCESSING
# resampler = torchaudio.transforms.Resample(orig_freq=SYSTEM_RATE, new_freq=MODEL_RATE).to(device)
# transform = torchaudio.transforms.MelSpectrogram(
#     sample_rate=MODEL_RATE, n_fft=1024, hop_length=256, n_mels=N_MELS  # Standard for 16k
# ).to(device)
# to_db = torchaudio.transforms.AmplitudeToDB().to(device)


# def audio_callback(indata, frames, time, status):
#     if status:
#         print(status)
#     data_queue.put(indata[:, 0].copy())


# def main():
#     # VB CABLE SELECTION
#     devices = sd.query_devices()
#     input_device = sd.default.device[0]

#     for i, dev in enumerate(devices):
#         if "CABLE Output" in dev["name"]:
#             input_device = i
#             print(f"Found VB Cable at index {i}")
#             break

#     print("\nListening...")
#     with sd.InputStream(
#         device=input_device, callback=audio_callback, channels=1, samplerate=SYSTEM_RATE, blocksize=CHUNK_SIZE
#     ):
#         while True:
#             try:
#                 audio = data_queue.get()

#                 # 1. Check Volume (VB Cable often silent if nothing playing)
#                 rms = np.sqrt(np.mean(audio**2))
#                 if rms < SILENCE_THRESHOLD:
#                     # Print RMS to help user debug connection
#                     print(f"\rSilence... (RMS: {rms:.4f})", end="", flush=True)
#                     continue

#                 # 2. Preprocessing
#                 tens = torch.tensor(audio).float().to(device) * GAIN_FACTOR
#                 tens = resampler(tens)
#                 spec = transform(tens)
#                 spec = to_db(spec)

#                 # 3. *** NORMALIZATION FIX ***
#                 # This puts data in the range the model likely expects (approx -2 to 2)
#                 mean = spec.mean()
#                 std = spec.std()
#                 if std > 0:
#                     spec = (spec - mean) / std

#                 # Add batch & channel dims [1, 1, H, W]
#                 spec = spec.unsqueeze(0).unsqueeze(0)

#                 # 4. Inference
#                 with torch.no_grad():
#                     logits = model(spec)
#                     probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
#                     pred = np.argmax(probs)

#                 # Output
#                 if pred != 0:  # Only print detections
#                     print(f"\nDETECTED: {CLASS_LABELS[pred]} ({probs[pred]:.2f})")
#                 else:
#                     print(f"\rNothing ({probs[0]:.2f})", end="", flush=True)

#             except KeyboardInterrupt:
#                 break
#             except Exception as e:
#                 pass


# if __name__ == "__main__":
#     main()
