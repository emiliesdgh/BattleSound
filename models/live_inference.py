# import sys
# from pathlib import Path

# sys.path.insert(0, str(Path(__file__).parent.parent))
# # Import your model definition
# from module.model import Conv2DNet
# import torch
# import torch.nn as nn
# import numpy as np
# import librosa
# import sounddevice as sd
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from collections import deque, OrderedDict
# import threading
# import queue


# # --- CONFIGURATION ---
# SAMPLE_RATE = 44100
# DURATION = 0.5
# N_MELS = 128
# HOP_LENGTH = 512

# # CRITICAL FIX: Changed from 144 to 120 to match your checkpoint's 1200 input size
# # Calculation: 120 -> Layer1(60) -> Layer2(30) -> Layer3(15). 15 * 40 * 2 = 1200.
# TARGET_SIZE = (128, 120)

# SILENCE_THRESHOLD = 0.005
# CONFIDENCE_THRESHOLD = 0.5

# CLASSES = ["Nothing", "Voice", "Effect"]
# Y_TICKS = [0, 1, 2]


# class LiveAudioClassifier:
#     def __init__(self, model_path):
#         self.q = queue.Queue()
#         self.buffer_size = int(SAMPLE_RATE * DURATION)
#         self.audio_buffer = np.zeros(self.buffer_size)
#         self.lock = threading.Lock()

#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         print(f"Loading model on {self.device}...")

#         # 1. Initialize Standard Architecture
#         self.model = Conv2DNet(feature_type="spec", duration=DURATION, multi_class=True)

#         # 2. PATCH THE MODEL: Overwrite fc1 to accept 1200 inputs instead of 1440
#         # This fixes the "size mismatch" error.
#         print("Patching model input layer to match checkpoint (1440 -> 1200)...")
#         self.model.fc1 = nn.Linear(1200, 256)

#         # 3. Load Weights
#         checkpoint = torch.load(model_path, map_location=self.device)
#         state_dict = None

#         if isinstance(checkpoint, list):
#             for item in checkpoint:
#                 if isinstance(item, (dict, OrderedDict)) and any(k.startswith("layer") for k in item.keys()):
#                     state_dict = item
#                     break
#         elif isinstance(checkpoint, (dict, OrderedDict)):
#             state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
#         elif isinstance(checkpoint, torch.nn.Module):
#             self.model = checkpoint

#         if state_dict:
#             new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
#             self.model.load_state_dict(new_state_dict, strict=False)

#         self.model.to(self.device)
#         self.model.eval()
#         print("Model loaded successfully.")

#     def audio_callback(self, indata, frames, time, status):
#         if status:
#             print(status)
#         with self.lock:
#             self.audio_buffer = np.roll(self.audio_buffer, -frames)
#             self.audio_buffer[-frames:] = indata[:, 0]
#             if not self.q.full():
#                 self.q.put(self.audio_buffer.copy())

#     def preprocess(self, audio):
#         if np.max(np.abs(audio)) < SILENCE_THRESHOLD:
#             return None

#         S = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH)
#         # Log Scale with fixed top_db for consistent loudness detection
#         S_dB = librosa.power_to_db(S, ref=np.max, top_db=80)

#         # Global Normalization
#         S_dB = (S_dB + 40) / 40

#         spec_tensor = torch.tensor(S_dB).unsqueeze(0).unsqueeze(0).float()

#         # Resize to (128, 120) to match the patched model
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
#             return 0  # Silence = Nothing

#         with torch.no_grad():
#             outputs = self.model(input_tensor)
#             probs = torch.nn.functional.softmax(outputs, dim=1)
#             probs_np = probs.cpu().numpy()[0]

#             winner_idx = np.argmax(probs_np)
#             winner_conf = probs_np[winner_idx]

#             # Debug output (Optional: comment out if too spammy)
#             print(f"N: {probs_np[0]:.2f} | V: {probs_np[1]:.2f} | E: {probs_np[2]:.2f}")

#             if winner_conf > CONFIDENCE_THRESHOLD:
#                 return winner_idx
#             else:
#                 return 0


# def run_live_plot():
#     print("\n--- Audio Devices ---")
#     print(sd.query_devices())
#     print("---------------------")

#     try:
#         default_device = sd.default.device[0]
#         dev_input = input(f"Enter Device ID (default {default_device}): ")
#         device_id = int(dev_input) if dev_input.strip() else default_device
#     except:
#         device_id = sd.default.device[0]

#     # Load the patched classifier
#     classifier = LiveAudioClassifier("best_model.pt")

#     # Plot Setup
#     x_len = 100
#     y_data = deque([0] * x_len, maxlen=x_len)
#     x_data = list(range(x_len))

#     fig, ax = plt.subplots(figsize=(10, 5))
#     (line,) = ax.step(x_data, y_data, where="post", lw=2, color="gray")

#     ax.set_ylim(-0.5, 2.5)
#     ax.set_yticks(Y_TICKS)
#     ax.set_yticklabels(CLASSES)
#     ax.set_title("Live Audio Classification")
#     ax.grid(axis="y", linestyle="--", alpha=0.7)

#     text_label = ax.text(0.5, 0.9, "Listening...", transform=ax.transAxes, ha="center", fontsize=14, fontweight="bold")

#     def update(frame):
#         pred_class = classifier.predict()

#         if pred_class is not None:
#             y_data.append(pred_class)
#             line.set_ydata(y_data)

#             label_text = CLASSES[pred_class]
#             text_label.set_text(f"DETECTED: {label_text}")

#             if pred_class == 2:  # Effect
#                 color = "red"
#             elif pred_class == 1:  # Voice
#                 color = "blue"
#             else:  # Nothing
#                 color = "gray"

#             text_label.set_color(color)
#             line.set_color(color)

#         return line, text_label

#     print(f"\nStream started on device {device_id}.")
#     stream = sd.InputStream(
#         device=device_id,
#         channels=1,
#         samplerate=SAMPLE_RATE,
#         callback=classifier.audio_callback,
#         blocksize=int(SAMPLE_RATE * 0.1),
#     )

#     with stream:
#         ani = animation.FuncAnimation(fig, update, interval=50, blit=True)
#         plt.show()


# if __name__ == "__main__":
#     run_live_plot()

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
# Import your model definition
from module.model import Conv2DNet

import torch
import numpy as np
import librosa
import sounddevice as sd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque, OrderedDict
import threading
import queue


# --- CONFIGURATION ---
SAMPLE_RATE = 44100  # Standard sample rate
DURATION = 0.5  # Duration model was trained on (seconds)
N_MELS = 128  # Frequency bins
HOP_LENGTH = 512  # STFT hop length
TARGET_SIZE = (128, 144)  # Matches Conv2DNet flattened size
SILENCE_THRESHOLD = 0.01  # Amplitude threshold to ignore pure silence

# 3-Class Configuration
CLASSES = ["Nothing", "Voice", "Effect"]
Y_TICKS = [0, 1, 2]  # The values on the Y axis


class LiveAudioClassifier:
    def __init__(self, model_path):
        self.q = queue.Queue()
        self.buffer_size = int(SAMPLE_RATE * DURATION)
        self.audio_buffer = np.zeros(self.buffer_size)
        self.lock = threading.Lock()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model on {self.device}...")

        # --- MODEL INITIALIZATION FOR 3 CLASSES ---
        # multi_class=True ensures the final Linear layer has 3 outputs
        self.model = Conv2DNet(feature_type="spec", duration=DURATION, multi_class=True)

        # --- ROBUST WEIGHT LOADING ---
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = None

        if isinstance(checkpoint, list):
            for item in checkpoint:
                if isinstance(item, (dict, OrderedDict)):
                    # Heuristic: check for layer keys
                    if any(k.startswith("layer") or k.startswith("conv") for k in item.keys()):
                        state_dict = item
                        break
            if state_dict is None:
                # Fallback to first dict
                for item in checkpoint:
                    if isinstance(item, (dict, OrderedDict)):
                        state_dict = item
                        break
        elif isinstance(checkpoint, (dict, OrderedDict)):
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        elif isinstance(checkpoint, torch.nn.Module):
            self.model = checkpoint
            state_dict = None

        if state_dict is not None:
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace("module.", "")
                new_state_dict[name] = v
            try:
                self.model.load_state_dict(new_state_dict, strict=False)
            except Exception as e:
                print(f"Warning during loading: {e}")

        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully.")

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        with self.lock:
            self.audio_buffer = np.roll(self.audio_buffer, -frames)
            self.audio_buffer[-frames:] = indata[:, 0]
            if not self.q.full():
                self.q.put(self.audio_buffer.copy())

    def preprocess(self, audio):
        if np.max(np.abs(audio)) < SILENCE_THRESHOLD:
            return None

        S = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Standardization
        mean = np.mean(S_dB)
        std = np.std(S_dB)
        if std > 0:
            S_dB = (S_dB - mean) / std
        else:
            S_dB = S_dB - mean

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

        # If silence, return class 0 ("Nothing")
        if input_tensor is None:
            return 0

        with torch.no_grad():
            outputs = self.model(input_tensor)
            # Get the index of the highest probability (0, 1, or 2)
            predicted_class = torch.argmax(outputs, dim=1).item()
            return predicted_class


def run_live_plot():
    print("\n--- Audio Devices ---")
    print(sd.query_devices())
    print("---------------------")

    try:
        default_device = sd.default.device[0]
        device_id_str = input(f"Enter Device ID for VB-Cable Output (default {default_device}): ")
        device_id = int(device_id_str) if device_id_str.strip() else default_device
    except Exception:
        device_id = sd.default.device[0]

    classifier = LiveAudioClassifier("best_model.pt")

    # Plotting variables
    x_len = 100
    # Initialize with 0 (Nothing)
    y_data = deque([0] * x_len, maxlen=x_len)
    x_data = list(range(x_len))

    fig, ax = plt.subplots(figsize=(10, 5))

    # Use a step plot to make it look like the "digital signal" style in your image
    (line,) = ax.step(x_data, y_data, where="post", lw=2, color="black")

    # Configure Y-Axis for 3 discrete classes
    ax.set_ylim(-0.5, 2.5)
    ax.set_yticks(Y_TICKS)
    ax.set_yticklabels(CLASSES)  # Label ticks as "Nothing", "Voice", "Effect"
    ax.set_title("Live Audio Classification")
    ax.set_xlabel("Time Step")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    text_label = ax.text(
        0.5, 0.9, "Initializing...", transform=ax.transAxes, ha="center", fontsize=14, fontweight="bold"
    )

    def update(frame):
        pred_class = classifier.predict()

        # Only update if we got a new prediction
        if pred_class is not None:
            y_data.append(pred_class)
            line.set_ydata(y_data)

            # Update dynamic label
            label_text = CLASSES[pred_class]
            text_label.set_text(f"DETECTED: {label_text}")

            # Change color based on class
            if pred_class == 2:  # Effect
                text_label.set_color("red")
                line.set_color("red")
            elif pred_class == 1:  # Voice
                text_label.set_color("blue")
                line.set_color("blue")
            else:  # Nothing
                text_label.set_color("gray")
                line.set_color("gray")

        return line, text_label

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


# if __name__ == "__main__":
#     run_live_plot()


# # import sys
# # from pathlib import Path

# # sys.path.insert(0, str(Path(__file__).parent.parent))


# # import torch
# # import numpy as np
# # import librosa
# # import sounddevice as sd
# # import matplotlib.pyplot as plt
# # import matplotlib.animation as animation
# # from collections import deque, OrderedDict
# # import threading
# # import queue
# # import time

# # # Import your model definition
# # from module.model import Conv2DNet

# # # --- CONFIGURATION ---
# # SAMPLE_RATE = 44100  # Standard sample rate
# # DURATION = 0.5  # Duration model was trained on (seconds)
# # N_MELS = 128  # Frequency bins
# # HOP_LENGTH = 512  # STFT hop length
# # TARGET_SIZE = (128, 144)  # (Freq, Time) - Matches Conv2DNet flattened size
# # CONFIDENCE_THRESHOLD = 0.6  # Probability required to classify as "Event"
# # SILENCE_THRESHOLD = 0.01  # Amplitude threshold to ignore pure silence

# # # Classes (Background vs Event)
# # CLASSES = ["Background", "Gunshot/Effect"]


# # class LiveAudioClassifier:
# #     def __init__(self, model_path):
# #         self.q = queue.Queue()
# #         self.buffer_size = int(SAMPLE_RATE * DURATION)
# #         self.audio_buffer = np.zeros(self.buffer_size)
# #         self.lock = threading.Lock()

# #         # Load Model Device
# #         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #         print(f"Loading model on {self.device}...")

# #         # Initialize Architecture
# #         self.model = Conv2DNet(feature_type="spec", duration=DURATION, multi_class=False)

# #         # --- ROBUST LOADING LOGIC (Fixes your 'list' error) ---
# #         checkpoint = torch.load(model_path, map_location=self.device)
# #         state_dict = None

# #         # 1. Handle List (e.g., [epoch, state_dict, optimizer])
# #         if isinstance(checkpoint, list):
# #             print(f"File is a list with {len(checkpoint)} elements. Searching for weights...")
# #             for item in checkpoint:
# #                 # Look for a dictionary that has keys starting with 'layer' or 'conv'
# #                 if isinstance(item, (dict, OrderedDict)):
# #                     if any(k.startswith("layer") or k.startswith("conv") for k in item.keys()):
# #                         state_dict = item
# #                         print("Found state_dict inside the list.")
# #                         break
# #             # Fallback: Use the first dict found if heuristic fails
# #             if state_dict is None:
# #                 for item in checkpoint:
# #                     if isinstance(item, (dict, OrderedDict)):
# #                         state_dict = item
# #                         print("Fallback: Using the first dictionary found in list.")
# #                         break

# #         # 2. Handle Dictionary (Standard save)
# #         elif isinstance(checkpoint, (dict, OrderedDict)):
# #             if "model" in checkpoint:
# #                 state_dict = checkpoint["model"]  # Common in training scripts
# #             elif "state_dict" in checkpoint:
# #                 state_dict = checkpoint["state_dict"]
# #             else:
# #                 state_dict = checkpoint  # The dict IS the weights

# #         # 3. Handle Full Model Object
# #         elif isinstance(checkpoint, torch.nn.Module):
# #             self.model = checkpoint
# #             state_dict = None  # No need to load state_dict

# #         if state_dict is not None:
# #             # Handle DataParallel (remove 'module.' prefix)
# #             new_state_dict = {}
# #             for k, v in state_dict.items():
# #                 name = k.replace("module.", "")
# #                 new_state_dict[name] = v

# #             try:
# #                 self.model.load_state_dict(new_state_dict, strict=False)
# #             except Exception as e:
# #                 print(f"Warning during weight loading: {e}")

# #         self.model.to(self.device)
# #         self.model.eval()
# #         print("Model loaded successfully.")

# #     def audio_callback(self, indata, frames, time, status):
# #         """Callback for sounddevice to capture audio."""
# #         if status:
# #             print(status)

# #         with self.lock:
# #             self.audio_buffer = np.roll(self.audio_buffer, -frames)
# #             self.audio_buffer[-frames:] = indata[:, 0]  # Mono

# #             if not self.q.full():
# #                 self.q.put(self.audio_buffer.copy())

# #     def preprocess(self, audio):
# #         """Preprocess audio to match training data specs."""

# #         # 1. Silence Gate (Fixes 'detects nothing' on VB Cable silence)
# #         if np.max(np.abs(audio)) < SILENCE_THRESHOLD:
# #             return None

# #         # 2. Mel Spectrogram
# #         S = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH)

# #         # 3. Log Scale & Normalization (Fixes 'everything is an effect')
# #         S_dB = librosa.power_to_db(S, ref=np.max)

# #         # Standardize (Mean=0, Std=1) - Critical for BattleSound models
# #         mean = np.mean(S_dB)
# #         std = np.std(S_dB)
# #         if std > 0:
# #             S_dB = (S_dB - mean) / std
# #         else:
# #             S_dB = S_dB - mean

# #         # 4. Resize to (128, 144) to fit Conv2DNet
# #         spec_tensor = torch.tensor(S_dB).unsqueeze(0).unsqueeze(0).float()
# #         spec_resized = torch.nn.functional.interpolate(
# #             spec_tensor, size=TARGET_SIZE, mode="bilinear", align_corners=False
# #         )

# #         return spec_resized.to(self.device)

# #     def predict(self):
# #         if self.q.empty():
# #             return None

# #         audio = self.q.get()
# #         input_tensor = self.preprocess(audio)

# #         # If silence, return pure Background probability
# #         if input_tensor is None:
# #             return [1.0, 0.0]

# #         with torch.no_grad():
# #             outputs = self.model(input_tensor)
# #             probs = torch.nn.functional.softmax(outputs, dim=1)
# #             return probs.cpu().numpy()[0]


# # def run_live_plot():
# #     print("\n--- Audio Devices ---")
# #     print(sd.query_devices())
# #     print("---------------------")

# #     # Allow user to pick VB-Cable
# #     try:
# #         default_device = sd.default.device[0]
# #         device_id_str = input(f"Enter Device ID for VB-Cable Output (default {default_device}): ")
# #         device_id = int(device_id_str) if device_id_str.strip() else default_device
# #     except Exception:
# #         device_id = sd.default.device[0]

# #     # Initialize
# #     classifier = LiveAudioClassifier("best_model.pt")

# #     # Plotting
# #     x_len = 100
# #     y_data = deque([0] * x_len, maxlen=x_len)
# #     x_data = list(range(x_len))

# #     fig, ax = plt.subplots(figsize=(10, 5))
# #     (line,) = ax.plot(x_data, y_data, lw=2, color="blue")
# #     ax.set_ylim(-0.1, 1.1)
# #     ax.set_title(f"Live Detection: {CLASSES[1]}")
# #     ax.set_ylabel("Probability")
# #     ax.set_xlabel("Time")
# #     ax.axhline(y=CONFIDENCE_THRESHOLD, color="red", linestyle="--", alpha=0.7, label="Threshold")
# #     ax.legend()

# #     text_label = ax.text(
# #         0.5, 0.85, "Initializing...", transform=ax.transAxes, ha="center", fontsize=14, fontweight="bold"
# #     )

# #     def update(frame):
# #         probs = classifier.predict()
# #         if probs is not None:
# #             # Index 1 is usually the "Event" class
# #             p_effect = probs[1] if len(probs) > 1 else 0

# #             y_data.append(p_effect)
# #             line.set_ydata(y_data)

# #             if p_effect > CONFIDENCE_THRESHOLD:
# #                 text_label.set_text(f"⚠️ DETECTED: {CLASSES[1]} ({p_effect:.2f})")
# #                 text_label.set_color("red")
# #             else:
# #                 text_label.set_text(f"Status: {CLASSES[0]}")
# #                 text_label.set_color("green")

# #         return line, text_label

# #     # Start Stream
# #     print(f"\nStarting audio stream on device {device_id}...")
# #     stream = sd.InputStream(
# #         device=device_id,
# #         channels=1,
# #         samplerate=SAMPLE_RATE,
# #         callback=classifier.audio_callback,
# #         blocksize=int(SAMPLE_RATE * 0.1),
# #     )

# #     with stream:
# #         ani = animation.FuncAnimation(fig, update, interval=50, blit=True)
# #         plt.show()


# # if __name__ == "__main__":
# #     run_live_plot()
