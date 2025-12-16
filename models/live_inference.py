import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


import torch
import numpy as np
import librosa
import sounddevice as sd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque, OrderedDict
import threading
import queue
import time

# Import your model definition
from module.model import Conv2DNet

# --- CONFIGURATION ---
SAMPLE_RATE = 44100  # Standard sample rate
DURATION = 0.5  # Duration model was trained on (seconds)
N_MELS = 128  # Frequency bins
HOP_LENGTH = 512  # STFT hop length
TARGET_SIZE = (128, 144)  # (Freq, Time) - Matches Conv2DNet flattened size
CONFIDENCE_THRESHOLD = 0.6  # Probability required to classify as "Event"
SILENCE_THRESHOLD = 0.01  # Amplitude threshold to ignore pure silence

# Classes (Background vs Event)
CLASSES = ["Background", "Gunshot/Effect"]


class LiveAudioClassifier:
    def __init__(self, model_path):
        self.q = queue.Queue()
        self.buffer_size = int(SAMPLE_RATE * DURATION)
        self.audio_buffer = np.zeros(self.buffer_size)
        self.lock = threading.Lock()

        # Load Model Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model on {self.device}...")

        # Initialize Architecture
        self.model = Conv2DNet(feature_type="spec", duration=DURATION, multi_class=False)

        # --- ROBUST LOADING LOGIC (Fixes your 'list' error) ---
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = None

        # 1. Handle List (e.g., [epoch, state_dict, optimizer])
        if isinstance(checkpoint, list):
            print(f"File is a list with {len(checkpoint)} elements. Searching for weights...")
            for item in checkpoint:
                # Look for a dictionary that has keys starting with 'layer' or 'conv'
                if isinstance(item, (dict, OrderedDict)):
                    if any(k.startswith("layer") or k.startswith("conv") for k in item.keys()):
                        state_dict = item
                        print("Found state_dict inside the list.")
                        break
            # Fallback: Use the first dict found if heuristic fails
            if state_dict is None:
                for item in checkpoint:
                    if isinstance(item, (dict, OrderedDict)):
                        state_dict = item
                        print("Fallback: Using the first dictionary found in list.")
                        break

        # 2. Handle Dictionary (Standard save)
        elif isinstance(checkpoint, (dict, OrderedDict)):
            if "model" in checkpoint:
                state_dict = checkpoint["model"]  # Common in training scripts
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint  # The dict IS the weights

        # 3. Handle Full Model Object
        elif isinstance(checkpoint, torch.nn.Module):
            self.model = checkpoint
            state_dict = None  # No need to load state_dict

        if state_dict is not None:
            # Handle DataParallel (remove 'module.' prefix)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace("module.", "")
                new_state_dict[name] = v

            try:
                self.model.load_state_dict(new_state_dict, strict=False)
            except Exception as e:
                print(f"Warning during weight loading: {e}")

        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully.")

    def audio_callback(self, indata, frames, time, status):
        """Callback for sounddevice to capture audio."""
        if status:
            print(status)

        with self.lock:
            self.audio_buffer = np.roll(self.audio_buffer, -frames)
            self.audio_buffer[-frames:] = indata[:, 0]  # Mono

            if not self.q.full():
                self.q.put(self.audio_buffer.copy())

    def preprocess(self, audio):
        """Preprocess audio to match training data specs."""

        # 1. Silence Gate (Fixes 'detects nothing' on VB Cable silence)
        if np.max(np.abs(audio)) < SILENCE_THRESHOLD:
            return None

        # 2. Mel Spectrogram
        S = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH)

        # 3. Log Scale & Normalization (Fixes 'everything is an effect')
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Standardize (Mean=0, Std=1) - Critical for BattleSound models
        mean = np.mean(S_dB)
        std = np.std(S_dB)
        if std > 0:
            S_dB = (S_dB - mean) / std
        else:
            S_dB = S_dB - mean

        # 4. Resize to (128, 144) to fit Conv2DNet
        spec_tensor = torch.tensor(S_dB).unsqueeze(0).unsqueeze(0).float()
        spec_resized = torch.nn.functional.interpolate(
            spec_tensor, size=TARGET_SIZE, mode="bilinear", align_corners=False
        )

        return spec_resized.to(self.device)

    def predict(self):
        if self.q.empty():
            return None

        audio = self.q.get()
        input_tensor = self.preprocess(audio)

        # If silence, return pure Background probability
        if input_tensor is None:
            return [1.0, 0.0]

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            return probs.cpu().numpy()[0]


def run_live_plot():
    print("\n--- Audio Devices ---")
    print(sd.query_devices())
    print("---------------------")

    # Allow user to pick VB-Cable
    try:
        default_device = sd.default.device[0]
        device_id_str = input(f"Enter Device ID for VB-Cable Output (default {default_device}): ")
        device_id = int(device_id_str) if device_id_str.strip() else default_device
    except Exception:
        device_id = sd.default.device[0]

    # Initialize
    classifier = LiveAudioClassifier("best_model.pt")

    # Plotting
    x_len = 100
    y_data = deque([0] * x_len, maxlen=x_len)
    x_data = list(range(x_len))

    fig, ax = plt.subplots(figsize=(10, 5))
    (line,) = ax.plot(x_data, y_data, lw=2, color="blue")
    ax.set_ylim(-0.1, 1.1)
    ax.set_title(f"Live Detection: {CLASSES[1]}")
    ax.set_ylabel("Probability")
    ax.set_xlabel("Time")
    ax.axhline(y=CONFIDENCE_THRESHOLD, color="red", linestyle="--", alpha=0.7, label="Threshold")
    ax.legend()

    text_label = ax.text(
        0.5, 0.85, "Initializing...", transform=ax.transAxes, ha="center", fontsize=14, fontweight="bold"
    )

    def update(frame):
        probs = classifier.predict()
        if probs is not None:
            # Index 1 is usually the "Event" class
            p_effect = probs[1] if len(probs) > 1 else 0

            y_data.append(p_effect)
            line.set_ydata(y_data)

            if p_effect > CONFIDENCE_THRESHOLD:
                text_label.set_text(f"⚠️ DETECTED: {CLASSES[1]} ({p_effect:.2f})")
                text_label.set_color("red")
            else:
                text_label.set_text(f"Status: {CLASSES[0]}")
                text_label.set_color("green")

        return line, text_label

    # Start Stream
    print(f"\nStarting audio stream on device {device_id}...")
    stream = sd.InputStream(
        device=device_id,
        channels=1,
        samplerate=SAMPLE_RATE,
        callback=classifier.audio_callback,
        blocksize=int(SAMPLE_RATE * 0.1),
    )

    with stream:
        ani = animation.FuncAnimation(fig, update, interval=50, blit=True)
        plt.show()


if __name__ == "__main__":
    run_live_plot()


# import torch
# import numpy as np
# import librosa
# import sounddevice as sd
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from collections import deque
# import threading
# import queue
# import time

# # Import your model definition
# from module.model import Conv2DNet

# # --- CONFIGURATION ---
# SAMPLE_RATE = 44100  # Standard sample rate
# DURATION = 0.5  # Duration model was trained on (seconds)
# N_MELS = 128  # Frequency bins
# HOP_LENGTH = 512  # STFT hop length
# TARGET_SIZE = (128, 144)  # (Freq, Time) - Must match model's expected flattened size (1440)
# CONFIDENCE_THRESHOLD = 0.6  # Probability required to classify as "Event"
# SILENCE_THRESHOLD = 0.01  # Amplitude threshold to ignore pure silence

# # Classes (Update these if your training had different labels)
# CLASSES = ["Background", "Gunshot/Effect"]
# # If your model has 3 classes, change this list.


# class LiveAudioClassifier:
#     def __init__(self, model_path):
#         self.q = queue.Queue()
#         self.buffer_size = int(SAMPLE_RATE * DURATION)
#         self.audio_buffer = np.zeros(self.buffer_size)
#         self.lock = threading.Lock()

#         # Load Model
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         print(f"Loading model on {self.device}...")

#         # Initialize model architecture (Assuming binary classification based on your description)
#         # If your best_model.pt is 3 classes, change multi_class=True
#         self.model = Conv2DNet(feature_type="spec", duration=DURATION, multi_class=False)

#         # Load weights
#         checkpoint = torch.load(model_path, map_location=self.device)

#         # Handle state_dict keys if they were saved with 'module.' prefix (DDP)
#         if isinstance(checkpoint, dict):
#             # Try to find state_dict in common keys
#             if "model" in checkpoint:
#                 state_dict = checkpoint["model"]
#                 # If model is a list (from main.py), take the first one
#                 if isinstance(state_dict, list):
#                     state_dict = state_dict[0]
#             else:
#                 state_dict = checkpoint

#             new_state_dict = {}
#             for k, v in state_dict.items():
#                 name = k.replace("module.", "")  # remove `module.`
#                 new_state_dict[name] = v
#             self.model.load_state_dict(new_state_dict, strict=False)
#         else:
#             # If saved as entire model object
#             self.model = checkpoint

#         self.model.to(self.device)
#         self.model.eval()
#         print("Model loaded successfully.")

#     def audio_callback(self, indata, frames, time, status):
#         """This is called for every audio block."""
#         if status:
#             print(status)

#         # Thread-safe buffer update
#         with self.lock:
#             # Shift buffer and append new data
#             self.audio_buffer = np.roll(self.audio_buffer, -frames)
#             self.audio_buffer[-frames:] = indata[:, 0]  # Use channel 0 (Mono)

#             # Put a copy in queue for processing
#             if not self.q.full():
#                 self.q.put(self.audio_buffer.copy())

#     def preprocess(self, audio):
#         """Convert raw audio to the spectrogram shape the model expects."""

#         # 1. Handle Silence (VB-Cable often sends pure zeros which breaks Log)
#         if np.max(np.abs(audio)) < SILENCE_THRESHOLD:
#             return None  # Skip processing silence

#         # 2. Compute Mel Spectrogram
#         S = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH)

#         # 3. Log Scale (dB)
#         S_dB = librosa.power_to_db(S, ref=np.max)

#         # 4. Normalize (Standard Scaling)
#         # This is critical. If your training data was normalized,
#         # live audio must be too.
#         mean = np.mean(S_dB)
#         std = np.std(S_dB)
#         if std > 0:
#             S_dB = (S_dB - mean) / std
#         else:
#             S_dB = S_dB - mean  # Avoid div by zero

#         # 5. Resize to match fixed input dimension (128, 144)
#         # We use opencv or torch for resizing. Here we use torch interpolation.
#         spec_tensor = torch.tensor(S_dB).unsqueeze(0).unsqueeze(0).float()  # (1, 1, F, T)

#         # Resize logic to ensure it fits the Conv2DNet "1440" requirement
#         # Conv2DNet with stride (4,2) reduces freq by ~64x and time by ~8x
#         # 128 / 64 * (Time/8) * 40 = 1440 => Time ~ 144
#         spec_resized = torch.nn.functional.interpolate(
#             spec_tensor, size=TARGET_SIZE, mode="bilinear", align_corners=False
#         )

#         return spec_resized.to(self.device)

#     def predict(self):
#         if self.q.empty():
#             return None

#         audio = self.q.get()
#         input_tensor = self.preprocess(audio)

#         if input_tensor is None:
#             return [1.0, 0.0]  # Default to Background if silence

#         with torch.no_grad():
#             outputs = self.model(input_tensor)
#             probs = torch.nn.functional.softmax(outputs, dim=1)
#             return probs.cpu().numpy()[0]


# # --- PLOTTING SETUP ---


# def run_live_plot():
#     # 1. Select Device
#     print("\n--- Audio Devices ---")
#     print(sd.query_devices())
#     print("---------------------")
#     default_device = sd.default.device[0]
#     device_id = input(f"Enter Device ID for VB-Cable Output (default {default_device}): ")
#     if device_id == "":
#         device_id = default_device
#     else:
#         device_id = int(device_id)

#     classifier = LiveAudioClassifier("best_model.pt")
#     # model_path = Path(__file__).parent / "best_model.pt"
#     # classifier = LiveAudioClassifier(str(model_path))

#     # Plotting variables
#     x_len = 100
#     y_data = deque([0] * x_len, maxlen=x_len)
#     x_data = list(range(x_len))

#     fig, ax = plt.subplots()
#     (line,) = ax.plot(x_data, y_data, lw=2)
#     ax.set_ylim(-0.1, 1.1)
#     ax.set_title(f"Live Audio Classification ({CLASSES[1]})")
#     ax.set_ylabel("Probability")
#     ax.set_xlabel("Time Step")
#     ax.axhline(y=CONFIDENCE_THRESHOLD, color="r", linestyle="--", alpha=0.5)

#     text_label = ax.text(0.5, 0.9, "", transform=ax.transAxes, ha="center", fontsize=12)

#     def update(frame):
#         probs = classifier.predict()
#         if probs is not None:
#             # We assume index 1 is the "Event/Effect" class
#             p_effect = probs[1] if len(probs) > 1 else 0

#             y_data.append(p_effect)
#             line.set_ydata(y_data)

#             # Update label
#             if p_effect > CONFIDENCE_THRESHOLD:
#                 text_label.set_text(f"DETECTED: {CLASSES[1]} ({p_effect:.2f})")
#                 text_label.set_color("red")
#             else:
#                 text_label.set_text(f"Status: {CLASSES[0]}")
#                 text_label.set_color("green")

#         return line, text_label

#     # Start Audio Stream
#     stream = sd.InputStream(
#         device=device_id,
#         channels=1,
#         samplerate=SAMPLE_RATE,
#         callback=classifier.audio_callback,
#         blocksize=int(SAMPLE_RATE * 0.1),  # Update every 100ms
#     )

#     print("\nStarting Stream... Close graph window to stop.")
#     with stream:
#         ani = animation.FuncAnimation(fig, update, interval=50, blit=True)
#         plt.show()


# if __name__ == "__main__":
#     run_live_plot()
