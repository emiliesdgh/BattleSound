import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from module.model import Conv2DNet
import torch
import torch.nn as nn
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque, OrderedDict
import threading
import queue
import torchaudio
import warnings

# --- CONFIGURATION ---
SAMPLE_RATE = 16000
DURATION = 0.5
SILENCE_THRESHOLD = 0.01
CONFIDENCE_THRESHOLD = 0.5

# SHAPE FIX: Your model.py forces 1200 inputs (128x120 after pooling)
# So we MUST resize the spectrogram to this shape.
TARGET_SIZE = (128, 120)

CLASSES = ["Voice", "Gunshot", "Mix", "Other"]
Y_TICKS = [-1, 0, 1, 2, 3]


class LiveAudioClassifier:
    def __init__(self, model_path):
        self.q = queue.Queue()
        self.buffer_size = int(SAMPLE_RATE * DURATION)
        self.audio_buffer = np.zeros(self.buffer_size)
        self.lock = threading.Lock()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model on {self.device}...")

        # 1. Initialize Model
        self.model = Conv2DNet(feature_type="spec", duration=DURATION, multi_class=True)

        # 2. PATCH INPUT: Match the 1200 inputs in your checkpoint
        self.model.fc1 = nn.Linear(1200, 256)

        # 3. PATCH OUTPUT: Match the 4 classes
        self.model.fc2 = nn.Linear(256, 4)

        # 4. Load Weights
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            checkpoint = torch.load(model_path, map_location=self.device)

        state_dict = None
        if isinstance(checkpoint, list):
            for item in checkpoint:
                if isinstance(item, (dict, OrderedDict)) and any(k.startswith("fc") for k in item.keys()):
                    state_dict = item
                    break
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))

        if state_dict:
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(new_state_dict, strict=False)

        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully.")

        # 5. Transform (Standard Spectrogram)
        self.transform = torchaudio.transforms.Spectrogram(
            n_fft=512, win_length=400, hop_length=200, normalized=True, power=2
        ).to(self.device)

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

        # --- HYBRID FIX ---
        # 1. VOLUME: Do NOT divide by 30000 (Data is fixed!)
        # We multiply by ~1.09 to match the training pipeline (32767/30000)
        audio_scaled = audio * 1.09

        audio_tensor = torch.from_numpy(audio_scaled).float().to(self.device)
        spec = self.transform(audio_tensor)

        # 2. SHAPE: Resize to (128, 120) because model.py is hardcoded to 1200 inputs
        spec = spec.unsqueeze(0).unsqueeze(0)
        spec_resized = torch.nn.functional.interpolate(spec, size=TARGET_SIZE, mode="bilinear", align_corners=False)

        return spec_resized

    def predict(self):
        if self.q.empty():
            return None

        audio = self.q.get()
        vol = np.max(np.abs(audio))

        input_tensor = self.preprocess(audio)
        if input_tensor is None:
            return -1

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            probs_np = probs.cpu().numpy()[0]

            winner_idx = np.argmax(probs_np)
            winner_conf = probs_np[winner_idx]

            print(
                f"Vol: {vol:.2f} | Voi: {probs_np[0]:.2f} | Gun: {probs_np[1]:.2f} | Mix: {probs_np[2]:.2f} | Oth: {probs_np[3]:.2f}"
            )

            if winner_conf > CONFIDENCE_THRESHOLD:
                return winner_idx
            else:
                return -2


def run_live_plot():
    print("\n--- Audio Devices ---")
    print(sd.query_devices())
    try:
        device_id = int(input(f"Enter Device ID (default {sd.default.device[0]}): ") or sd.default.device[0])
    except:
        device_id = sd.default.device[0]

    classifier = LiveAudioClassifier("best_model.pt")

    x_len = 100
    y_data = deque([0] * x_len, maxlen=x_len)
    x_data = list(range(x_len))

    fig, ax = plt.subplots(figsize=(10, 5))
    (line,) = ax.step(x_data, y_data, where="post", lw=2, color="gray")
    ax.set_ylim(-2.5, 3.5)
    ax.set_yticks([-2, -1, 0, 1, 2, 3])
    ax.set_yticklabels(["Unsure", "Silence"] + CLASSES)
    ax.set_title("Live Audio Classification")
    text_label = ax.text(0.5, 0.9, "Listening...", transform=ax.transAxes, ha="center", fontsize=14, fontweight="bold")

    def update(frame):
        pred_class = classifier.predict()
        if pred_class is not None:
            y_data.append(pred_class)
            line.set_ydata(y_data)

            colors = {-2: "orange", -1: "gray", 0: "red", 1: "purple", 2: "green", 3: "blue"}

            if pred_class < 0:
                label_text = "SILENCE" if pred_class == -1 else "UNSURE"
            else:
                label_text = CLASSES[pred_class]

            color = colors.get(pred_class, "black")
            text_label.set_text(f"DETECTED: {label_text}")
            text_label.set_color(color)
            line.set_color(color)
        return line, text_label

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
# # SAMPLE_RATE = 44100
# SAMPLE_RATE = 16000
# DURATION = 0.5
# N_FFT = 512  # to match custom_dataset.py n_fft
# N_MELS = 128
# # HOP_LENGTH = 512
# HOP_LENGTH = 200  # to match custom_dataset.py hop_length
# WIN_LENGTH = 400  # to match custom_dataset.py win_length

# # CRITICAL FIX: Keep this input size patch (1200) as per your previous checkpoint
# # TARGET_SIZE = (128, 120)
# TARGET_SIZE = (257, 41)  # Updated to match new model input size

# SILENCE_THRESHOLD = 0.01
# CONFIDENCE_THRESHOLD = 0.5

# # UPDATE: Added "Other" to the classes
# CLASSES = ["Voice", "Gunshot", "Mix", "Other"]
# Y_TICKS = [-1, 0, 1, 2, 3]  # -1=Silence, 0-3=Model Classes


# class LiveAudioClassifier:
#     def __init__(self, model_path):
#         self.q = queue.Queue()
#         self.buffer_size = int(SAMPLE_RATE * DURATION)
#         self.audio_buffer = np.zeros(self.buffer_size)
#         self.lock = threading.Lock()

#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         print(f"Loading model on {self.device}...")

#         # 1. Initialize Standard Architecture
#         # Note: If Conv2DNet defaults to 3 classes, we must patch it below.
#         self.model = Conv2DNet(feature_type="spec", duration=DURATION, multi_class=True)

#         # 2. PATCH THE MODEL
#         print("Patching model layers to match 4-class checkpoint...")

#         # Patch Input: 1440 -> 1200 (Your previous fix)
#         self.model.fc1 = nn.Linear(1200, 256)

#         # Patch Output: 3 -> 4 Classes (For your new "Other" class)
#         # This ensures the model has 4 output neurons to match your new weights.
#         self.model.fc2 = nn.Linear(256, 4)

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

#             # This loads the weights. If your model file is still 3 classes, this line will crash!
#             # Ensure 'best_model.pt' is the NEW file.
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

#     # def preprocess(self, audio):
#     #     if np.max(np.abs(audio)) < SILENCE_THRESHOLD:
#     #         return None

#     #     S = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH)
#     #     S_dB = librosa.power_to_db(S, ref=np.max, top_db=80)
#     #     S_dB = (S_dB + 40) / 40

#     #     spec_tensor = torch.tensor(S_dB).unsqueeze(0).unsqueeze(0).float()

#     #     spec_resized = torch.nn.functional.interpolate(
#     #         spec_tensor, size=TARGET_SIZE, mode="bilinear", align_corners=False
#     #     )
#     #     return spec_resized.to(self.device)

#     def preprocess(self, audio):
#         # 1. Silence Check
#         if np.max(np.abs(audio)) < SILENCE_THRESHOLD:
#             return None

#         # 2. Generate Linear Spectrogram (NOT Mel, NOT Log)
#         # matches: torchaudio.transforms.Spectrogram(n_fft=512, power=2)
#         S = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
#         S_power = np.abs(S) ** 2  # Power Spectrogram

#         # 3. Convert to Tensor
#         # No Log scaling, No (S+40)/40 normalization because custom_dataset.py didn't use them for 'spec'
#         spec_tensor = torch.tensor(S_power).unsqueeze(0).unsqueeze(0).float()

#         # 4. Resize if necessary (To handle slight time-step variations)
#         # Your model expects a fixed size, so we force it here.
#         spec_resized = torch.nn.functional.interpolate(
#             spec_tensor, size=TARGET_SIZE, mode="bilinear", align_corners=False
#         )

#         return spec_resized.to(self.device)

#     def predict(self):
#         if self.q.empty():
#             return None

#         audio = self.q.get()
#         vol = np.max(np.abs(audio))

#         input_tensor = self.preprocess(audio)
#         if input_tensor is None:
#             return -1  # Silence

#         with torch.no_grad():
#             outputs = self.model(input_tensor)
#             probs = torch.nn.functional.softmax(outputs, dim=1)
#             probs_np = probs.cpu().numpy()[0]

#             winner_idx = np.argmax(probs_np)
#             winner_conf = probs_np[winner_idx]

#             # UPDATED DEBUG LOG: Shows 4 probabilities now
#             print(
#                 f"Vol: {vol:.2f} | V: {probs_np[0]:.2f} | G: {probs_np[1]:.2f} | M: {probs_np[2]:.2f} | O: {probs_np[3]:.2f}"
#             )

#             if winner_conf > CONFIDENCE_THRESHOLD:
#                 return winner_idx
#             else:
#                 return -1  # Unsure


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

#     # Load classifier
#     classifier = LiveAudioClassifier("best_model.pt")

#     # Plot Setup
#     x_len = 100
#     y_data = deque([0] * x_len, maxlen=x_len)
#     x_data = list(range(x_len))

#     fig, ax = plt.subplots(figsize=(10, 5))
#     (line,) = ax.step(x_data, y_data, where="post", lw=2, color="gray")

#     # UPDATE: Y-Axis to fit 4 classes (-1 to 3)
#     ax.set_ylim(-1.5, 3.5)
#     ax.set_yticks(Y_TICKS)
#     ax.set_yticklabels(["Silence", "Voice", "Gunshot", "Mix", "Other"])
#     ax.set_title("Live Audio Classification (4 Classes)")
#     ax.grid(axis="y", linestyle="--", alpha=0.7)

#     text_label = ax.text(0.5, 0.9, "Listening...", transform=ax.transAxes, ha="center", fontsize=14, fontweight="bold")

#     def update(frame):
#         pred_class = classifier.predict()

#         if pred_class is not None:
#             y_data.append(pred_class)
#             line.set_ydata(y_data)

#             # Color Coding
#             if pred_class == -1:  # Silence / Unsure
#                 label_text = "SILENCE / UNSURE"
#                 color = "gray"
#             elif pred_class == 0:  # Voice
#                 label_text = "VOICE"
#                 color = "blue"
#             elif pred_class == 1:  # Gunshot
#                 label_text = "GUNSHOT !!!"
#                 color = "red"
#             elif pred_class == 2:  # Mix
#                 label_text = "MIX (GUNSHOT+)"
#                 color = "purple"
#             elif pred_class == 3:  # Other
#                 label_text = "OTHER / NOISE"
#                 color = "orange"
#             else:
#                 label_text = "UNKNOWN"
#                 color = "black"

#             text_label.set_text(f"DETECTED: {label_text}")
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


# # import sys
# # from pathlib import Path

# # sys.path.insert(0, str(Path(__file__).parent.parent))
# # # Import your model definition
# # from module.model import Conv2DNet
# # import torch
# # import torch.nn as nn
# # import numpy as np
# # import librosa
# # import sounddevice as sd
# # import matplotlib.pyplot as plt
# # import matplotlib.animation as animation
# # from collections import deque, OrderedDict
# # import threading
# # import queue


# # # --- CONFIGURATION ---
# # SAMPLE_RATE = 44100
# # DURATION = 0.5
# # N_MELS = 128
# # HOP_LENGTH = 512

# # # CRITICAL FIX: Changed from 144 to 120 to match your checkpoint's 1200 input size
# # # Calculation: 120 -> Layer1(60) -> Layer2(30) -> Layer3(15). 15 * 40 * 2 = 1200.
# # TARGET_SIZE = (128, 120)

# # SILENCE_THRESHOLD = 0.005
# # CONFIDENCE_THRESHOLD = 0.5

# # CLASSES = ["Voice", "Gunshot", "Mix"]
# # Y_TICKS = [-1, 0, 1, 2]  # added -1 for silence


# # class LiveAudioClassifier:
# #     def __init__(self, model_path):
# #         self.q = queue.Queue()
# #         self.buffer_size = int(SAMPLE_RATE * DURATION)
# #         self.audio_buffer = np.zeros(self.buffer_size)
# #         self.lock = threading.Lock()

# #         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #         print(f"Loading model on {self.device}...")

# #         # 1. Initialize Standard Architecture
# #         self.model = Conv2DNet(feature_type="spec", duration=DURATION, multi_class=True)

# #         # 2. PATCH THE MODEL: Overwrite fc1 to accept 1200 inputs instead of 1440
# #         # This fixes the "size mismatch" error.
# #         print("Patching model input layer to match checkpoint (1440 -> 1200)...")
# #         self.model.fc1 = nn.Linear(1200, 256)

# #         # 3. Load Weights
# #         checkpoint = torch.load(model_path, map_location=self.device)
# #         state_dict = None

# #         if isinstance(checkpoint, list):
# #             for item in checkpoint:
# #                 if isinstance(item, (dict, OrderedDict)) and any(k.startswith("layer") for k in item.keys()):
# #                     state_dict = item
# #                     break
# #         elif isinstance(checkpoint, (dict, OrderedDict)):
# #             state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
# #         elif isinstance(checkpoint, torch.nn.Module):
# #             self.model = checkpoint

# #         if state_dict:
# #             new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
# #             self.model.load_state_dict(new_state_dict, strict=False)

# #         self.model.to(self.device)
# #         self.model.eval()
# #         print("Model loaded successfully.")

# #     def audio_callback(self, indata, frames, time, status):
# #         if status:
# #             print(status)
# #         with self.lock:
# #             self.audio_buffer = np.roll(self.audio_buffer, -frames)
# #             self.audio_buffer[-frames:] = indata[:, 0]
# #             if not self.q.full():
# #                 self.q.put(self.audio_buffer.copy())

# #     def preprocess(self, audio):
# #         if np.max(np.abs(audio)) < SILENCE_THRESHOLD:
# #             return None

# #         S = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH)
# #         # Log Scale with fixed top_db for consistent loudness detection
# #         S_dB = librosa.power_to_db(S, ref=np.max, top_db=80)

# #         # Global Normalization
# #         S_dB = (S_dB + 40) / 40

# #         spec_tensor = torch.tensor(S_dB).unsqueeze(0).unsqueeze(0).float()

# #         # Resize to (128, 120) to match the patched model
# #         spec_resized = torch.nn.functional.interpolate(
# #             spec_tensor, size=TARGET_SIZE, mode="bilinear", align_corners=False
# #         )
# #         return spec_resized.to(self.device)

# #     def predict(self):
# #         if self.q.empty():
# #             return None

# #         audio = self.q.get()

# #         # DEBUG: Print Volume to ensure we aren't clipping
# #         vol = np.max(np.abs(audio))

# #         # 1. SILENCE CHECK (Handled manually)
# #         input_tensor = self.preprocess(audio)
# #         if input_tensor is None:
# #             return -1  # Return -1 for Silence (instead of 0)

# #         with torch.no_grad():
# #             outputs = self.model(input_tensor)
# #             probs = torch.nn.functional.softmax(outputs, dim=1)
# #             probs_np = probs.cpu().numpy()[0]

# #             winner_idx = np.argmax(probs_np)
# #             winner_conf = probs_np[winner_idx]

# #             # 2. DEBUG LOGGING
# #             # Prints: "Vol: 0.15 | V: 0.10 | G: 0.85 | M: 0.05"
# #             print(f"Vol: {vol:.2f} | Voice: {probs_np[0]:.2f} | Gun: {probs_np[1]:.2f} | Mix: {probs_np[2]:.2f}")

# #             # 3. CONFIDENCE CHECK
# #             if winner_conf > CONFIDENCE_THRESHOLD:
# #                 return winner_idx
# #             else:
# #                 return -1  # Return -1 if unsure (Don't default to Voice!)


# # def run_live_plot():
# #     print("\n--- Audio Devices ---")
# #     print(sd.query_devices())
# #     print("---------------------")

# #     try:
# #         default_device = sd.default.device[0]
# #         dev_input = input(f"Enter Device ID (default {default_device}): ")
# #         device_id = int(dev_input) if dev_input.strip() else default_device
# #     except:
# #         device_id = sd.default.device[0]

# #     # Load the patched classifier
# #     classifier = LiveAudioClassifier("best_model.pt")

# #     # Plot Setup
# #     x_len = 100
# #     y_data = deque([0] * x_len, maxlen=x_len)
# #     x_data = list(range(x_len))

# #     fig, ax = plt.subplots(figsize=(10, 5))
# #     (line,) = ax.step(x_data, y_data, where="post", lw=2, color="gray")

# #     ax.set_ylim(-1.5, 2.5)
# #     ax.set_yticks(Y_TICKS)
# #     ax.set_yticklabels(["Silence", "Voice", "Gunshot", "Mix"])  # Update labels
# #     ax.set_title("Live Audio Classification")
# #     ax.grid(axis="y", linestyle="--", alpha=0.7)

# #     text_label = ax.text(0.5, 0.9, "Listening...", transform=ax.transAxes, ha="center", fontsize=14, fontweight="bold")

# #     def update(frame):
# #         pred_class = classifier.predict()

# #         if pred_class is not None:
# #             y_data.append(pred_class)
# #             line.set_ydata(y_data)

# #             # Color Coding
# #             if pred_class == -1:  # Silence / Unsure
# #                 label_text = "SILENCE / UNSURE"
# #                 color = "gray"
# #             elif pred_class == 0:  # Voice
# #                 label_text = "VOICE"
# #                 color = "blue"
# #             elif pred_class == 1:  # Gunshot
# #                 label_text = "GUNSHOT !!!"
# #                 color = "red"
# #             elif pred_class == 2:  # Mix (Voice + Gunshot)
# #                 label_text = "MIX (GUNSHOT+)"
# #                 color = "purple"  # Distinct color for Mix

# #             text_label.set_text(f"DETECTED: {label_text}")
# #             text_label.set_color(color)
# #             line.set_color(color)

# #         return line, text_label

# #     print(f"\nStream started on device {device_id}.")
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
