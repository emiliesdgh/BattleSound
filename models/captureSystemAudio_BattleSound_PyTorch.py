import sounddevice as sd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt
import queue
import sys
from collections import deque
from pathlib import Path

# --- CONFIG ---
MODEL_PATH = Path("C:/Users/egrandjean/Desktop/BattleSound_model/BattleSound/models/best_model.pt")

# CRITICAL FIX: Distinguish between Capture Rate and Model Rate
SYSTEM_RATE = 48000  # Your mic/cable output (usually 44100 or 48000)
MODEL_RATE = 16000  # What the AI was trained on
DURATION = 0.5  # Seconds

# We capture more samples to get the same duration in real time
CAPTURE_SIZE = int(SYSTEM_RATE * DURATION)

CLASS_LABELS = {0: "Nothing", 1: "Voices", 2: "Effects"}
HISTORY_SIZE = 100
GAIN_FACTOR = 5.0

data_queue = queue.Queue()


# --- MODEL (3 Layer / 5x5) ---
class Conv2DNet(nn.Module):
    def __init__(self, num_class=3):
        super(Conv2DNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 10, 5, 1, 2), nn.BatchNorm2d(10), nn.ReLU(), nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(nn.Conv2d(10, 20, 5, 1, 2), nn.BatchNorm2d(20), nn.ReLU(), nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(nn.Conv2d(20, 40, 5, 1, 2), nn.BatchNorm2d(40), nn.ReLU(), nn.MaxPool2d(2))
        self.fc1 = nn.Linear(1200, 256)
        self.fc2 = nn.Linear(256, num_class)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.layer3(self.layer2(self.layer1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc2(self.dropout(F.relu(self.fc1(x))))
        return x


# --- SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading model on {device}...")

model = Conv2DNet(3).to(device)
try:
    ckpt = torch.load(MODEL_PATH, map_location=device)
    if isinstance(ckpt, list):
        model.load_state_dict(ckpt[0])
    elif isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"][0] if isinstance(ckpt["model"], list) else ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

model.eval()

# --- TRANSFORMS ---
# 1. Resampler: Converts 48k (System) -> 16k (Model)
resampler = torchaudio.transforms.Resample(orig_freq=SYSTEM_RATE, new_freq=MODEL_RATE).to(device)

# 2. MelSpectrogram: Must allow sample_rate=MODEL_RATE (16k)
transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=MODEL_RATE, n_fft=512, hop_length=128, n_mels=40  # The AI sees 16000Hz data now
).to(device)


def audio_callback(indata, frames, time, status):
    if status:
        print(f"Status: {status}", file=sys.stderr)

    # 1. Copy data
    audio_chunk = indata[:, 0].copy()

    # 2. Manual Gain
    audio_chunk = audio_chunk * GAIN_FACTOR

    # 3. Clip
    audio_chunk = np.clip(audio_chunk, -1.0, 1.0)

    data_queue.put(audio_chunk)


def main():
    # DEVICE SELECT
    input_device_id = None
    devices = sd.query_devices()
    print("\nScanning for 'CABLE Output'...")
    for i, dev in enumerate(devices):
        if "CABLE Output" in dev["name"] and dev["max_input_channels"] > 0:
            input_device_id = i
            print(f"Found 'CABLE Output' ID: {i}")
            break
    if input_device_id is None:
        input_device_id = sd.default.device[0]
        print(f"Using Default: {input_device_id}")

    # GRAPH
    print("Opening Graph...")
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 4))
    y_data = deque([0] * HISTORY_SIZE, maxlen=HISTORY_SIZE)
    (line,) = ax.step(np.arange(HISTORY_SIZE), y_data, where="post", color="blue", linewidth=2)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(list(CLASS_LABELS.values()))
    ax.set_title(f"PyTorch Real-Time (Resampling {SYSTEM_RATE}->{MODEL_RATE})")

    # STREAM: Use SYSTEM_RATE and CAPTURE_SIZE
    stream = sd.InputStream(
        device=input_device_id,
        callback=audio_callback,
        channels=1,
        samplerate=SYSTEM_RATE,
        blocksize=CAPTURE_SIZE,  # Captures 24000 samples (0.5s at 48k)
    )

    with stream:
        print(f"\nListening... Press Ctrl+C to stop.\n")
        print(f"{'VOL':<8} | {'PREDICTION':<15} | {'CONF':<6} | {'RAW PROBS'}")
        print("-" * 60)

        while True:
            try:
                # Get raw audio (numpy)
                audio_np = data_queue.get_nowait()

                # --- VOLUME CHECK ---
                rms = np.sqrt(np.mean(audio_np**2))

                # --- PROCESSING ---
                # 1. To Tensor (on GPU)
                audio_tens = torch.tensor(audio_np).float().to(device)

                # 2. Resample (48k -> 16k)
                # This squashes the 24k samples down to 8k samples
                audio_resampled = resampler(audio_tens)

                # 3. Mel Spectrogram
                spec = transform(audio_resampled.unsqueeze(0))
                spec = torch.log(spec + 1e-9).unsqueeze(1)

                # 4. Shape Fix (Ensure exactly 48 width)
                if spec.shape[3] > 48:
                    spec = spec[:, :, :, :48]
                elif spec.shape[3] < 48:
                    spec = F.pad(spec, (0, 48 - spec.shape[3]))

                # 5. Inference
                with torch.no_grad():
                    probs = torch.softmax(model(spec), dim=1).cpu().numpy()[0]
                    pred = int(np.argmax(probs))

                # UPDATE GRAPH
                y_data.append(pred)
                line.set_ydata(y_data)

                # PRINT
                raw_s = f"[{probs[0]:.2f}, {probs[1]:.2f}, {probs[2]:.2f}]"
                print(f"{rms:.4f}   | {CLASS_LABELS[pred]:<15} | {probs[pred]:.2f}   | {raw_s}")

                fig.canvas.draw_idle()
                fig.canvas.flush_events()

            except queue.Empty:
                plt.pause(0.01)
            except Exception as e:
                print(f"Error: {e}")
                break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")


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

# # --- CONFIG ---
# MODEL_PATH = Path("C:/Users/egrandjean/Desktop/BattleSound_model/BattleSound/models/best_model.pt")
# SYSTEM_RATE = 48000
# MODEL_RATE = 16000
# DURATION = 0.5
# CHUNK_SIZE = int(MODEL_RATE * DURATION)
# CLASS_LABELS = {0: "Nothing", 1: "Voices", 2: "Effects"}
# HISTORY_SIZE = 100
# data_queue = queue.Queue()


# # --- MODEL (3 Layer / 5x5) ---
# class Conv2DNet(nn.Module):
#     def __init__(self, num_class=3):
#         super(Conv2DNet, self).__init__()
#         self.layer1 = nn.Sequential(nn.Conv2d(1, 10, 5, 1, 2), nn.BatchNorm2d(10), nn.ReLU(), nn.MaxPool2d(2))
#         self.layer2 = nn.Sequential(nn.Conv2d(10, 20, 5, 1, 2), nn.BatchNorm2d(20), nn.ReLU(), nn.MaxPool2d(2))
#         self.layer3 = nn.Sequential(nn.Conv2d(20, 40, 5, 1, 2), nn.BatchNorm2d(40), nn.ReLU(), nn.MaxPool2d(2))
#         self.fc1 = nn.Linear(1200, 256)
#         self.fc2 = nn.Linear(256, num_class)
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, x):
#         x = self.layer3(self.layer2(self.layer1(x)))
#         x = x.view(x.size(0), -1)
#         x = self.fc2(self.dropout(F.relu(self.fc1(x))))
#         return x


# # --- SETUP ---
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Loading model on {device}...")
# model = Conv2DNet(3).to(device)
# ckpt = torch.load(MODEL_PATH, map_location=device)
# if isinstance(ckpt, list):
#     model.load_state_dict(ckpt[0])
# elif isinstance(ckpt, dict) and "model" in ckpt:
#     model.load_state_dict(ckpt["model"][0] if isinstance(ckpt["model"], list) else ckpt["model"])
# else:
#     model.load_state_dict(ckpt)
# model.eval()

# transform = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=512, hop_length=128, n_mels=40).to(
#     device
# )


# # def audio_callback(indata, frames, time, status):
# #     if status:
# #         print(f"Status: {status}", file=sys.stderr)
# #     data_queue.put(indata[:, 0].copy())
# # ==========================================
# # UPDATED CALLBACK WITH GAIN BOOST
# # ==========================================
# GAIN_FACTOR = 5.0  # Multiplies volume by 5. Try 2.0, 5.0, or 10.0


# def audio_callback(indata, frames, time, status):
#     if status:
#         print(f"Status: {status}", file=sys.stderr)

#     # 1. Copy data
#     audio_chunk = indata[:, 0].copy()

#     # 2. Apply Manual Gain (Boost Volume)
#     audio_chunk = audio_chunk * GAIN_FACTOR

#     # 3. Clip to prevent distortion (Max is 1.0)
#     audio_chunk = np.clip(audio_chunk, -1.0, 1.0)

#     # 4. Optional: Print if clipping (Debug)
#     # if np.max(np.abs(audio_chunk)) >= 1.0:
#     #     print("Warning: Audio Clipping!", file=sys.stderr)

#     data_queue.put(audio_chunk)


# def main():
#     # DEVICE SELECT
#     input_device_id = None
#     devices = sd.query_devices()
#     print("\nScanning for 'CABLE Output'...")
#     for i, dev in enumerate(devices):
#         if "CABLE Output" in dev["name"] and dev["max_input_channels"] > 0:
#             input_device_id = i
#             print(f"Found 'CABLE Output' ID: {i}")
#             break
#     if input_device_id is None:
#         input_device_id = sd.default.device[0]
#         print(f"Using Default: {input_device_id}")

#     # GRAPH
#     print("Opening Graph...")
#     plt.ion()
#     fig, ax = plt.subplots(figsize=(10, 4))
#     y_data = deque([0] * HISTORY_SIZE, maxlen=HISTORY_SIZE)
#     (line,) = ax.step(np.arange(HISTORY_SIZE), y_data, where="post", color="blue", linewidth=2)
#     ax.set_yticks([0, 1, 2])
#     ax.set_yticklabels(list(CLASS_LABELS.values()))
#     ax.set_title("PyTorch Real-Time")

#     stream = sd.InputStream(
#         device=input_device_id, callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE
#     )

#     with stream:
#         print(f"\nListening... Press Ctrl+C to stop.\n")
#         print(f"{'VOL':<8} | {'PREDICTION':<15} | {'CONF':<6} | {'RAW PROBS'}")
#         print("-" * 60)

#         while True:
#             try:
#                 audio = data_queue.get_nowait()

#                 # --- VOLUME CHECK (Debug) ---
#                 rms = np.sqrt(np.mean(audio**2))

#                 # INFERENCE
#                 tens = torch.tensor(audio).float().to(device)
#                 spec = transform(tens.unsqueeze(0))
#                 spec = torch.log(spec + 1e-9).unsqueeze(1)

#                 if spec.shape[3] > 48:
#                     spec = spec[:, :, :, :48]
#                 elif spec.shape[3] < 48:
#                     spec = F.pad(spec, (0, 48 - spec.shape[3]))

#                 with torch.no_grad():
#                     probs = torch.softmax(model(spec), dim=1).cpu().numpy()[0]
#                     pred = int(np.argmax(probs))

#                 y_data.append(pred)
#                 line.set_ydata(y_data)

#                 # PRINT
#                 raw_s = f"[{probs[0]:.2f}, {probs[1]:.2f}, {probs[2]:.2f}]"
#                 print(f"{rms:.4f}   | {CLASS_LABELS[pred]:<15} | {probs[pred]:.2f}   | {raw_s}")

#                 fig.canvas.draw_idle()
#                 fig.canvas.flush_events()
#             except queue.Empty:
#                 plt.pause(0.01)
#             except Exception:
#                 break


# if __name__ == "__main__":
#     try:
#         main()
#     except KeyboardInterrupt:
#         print("\nStopped.")


# # import sounddevice as sd
# # import numpy as np
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # import torchaudio
# # import matplotlib.pyplot as plt
# # import queue
# # import sys
# # from collections import deque
# # from pathlib import Path

# # # --- 1. CONFIGURATION ---
# # MODEL_PATH = Path("C:/Users/egrandjean/Desktop/BattleSound_model/BattleSound/models/best_model.pt")
# # SAMPLE_RATE = 16000
# # DURATION = 0.5
# # CHUNK_SIZE = int(SAMPLE_RATE * DURATION)

# # # Mapping: 0=Nothing, 1=Voices, 2=Effects
# # CLASS_LABELS = {0: "Nothing", 1: "Voices", 2: "Effects"}
# # HISTORY_SIZE = 100

# # data_queue = queue.Queue()


# # # --- 2. MODEL DEFINITION (3 Layers, Kernel 5x5) ---
# # class Conv2DNet(nn.Module):
# #     def __init__(self, num_class=3):
# #         super(Conv2DNet, self).__init__()
# #         self.layer1 = nn.Sequential(
# #             nn.Conv2d(1, 10, kernel_size=5, stride=1, padding=2), nn.BatchNorm2d(10), nn.ReLU(), nn.MaxPool2d(2)
# #         )
# #         self.layer2 = nn.Sequential(
# #             nn.Conv2d(10, 20, kernel_size=5, stride=1, padding=2), nn.BatchNorm2d(20), nn.ReLU(), nn.MaxPool2d(2)
# #         )
# #         self.layer3 = nn.Sequential(
# #             nn.Conv2d(20, 40, kernel_size=5, stride=1, padding=2), nn.BatchNorm2d(40), nn.ReLU(), nn.MaxPool2d(2)
# #         )
# #         self.fc1 = nn.Linear(1200, 256)
# #         self.fc2 = nn.Linear(256, num_class)
# #         self.dropout = nn.Dropout(0.5)

# #     def forward(self, x):
# #         x = self.layer1(x)
# #         x = self.layer2(x)
# #         x = self.layer3(x)
# #         x = x.view(x.size(0), -1)
# #         x = F.relu(self.fc1(x))
# #         x = self.dropout(x)
# #         x = self.fc2(x)
# #         return x


# # # --- 3. LOAD MODEL ---
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # print(f"Loading model on {device}...")

# # model = Conv2DNet(num_class=3).to(device)
# # checkpoint = torch.load(MODEL_PATH, map_location=device)

# # if isinstance(checkpoint, list):
# #     model.load_state_dict(checkpoint[0])
# # elif isinstance(checkpoint, dict) and "model" in checkpoint:
# #     if isinstance(checkpoint["model"], list):
# #         model.load_state_dict(checkpoint["model"][0])
# #     else:
# #         model.load_state_dict(checkpoint["model"])
# # else:
# #     model.load_state_dict(checkpoint)

# # model.eval()

# # transform = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=512, hop_length=128, n_mels=40).to(
# #     device
# # )


# # # --- 4. AUDIO CALLBACK ---
# # def audio_callback(indata, frames, time, status):
# #     if status:
# #         print(f"Audio Status: {status}", file=sys.stderr)
# #     data_queue.put(indata[:, 0].copy())


# # # --- 5. MAIN LOOP ---
# # def main():
# #     # ==========================================
# #     # AUTOMATIC DEVICE SELECTION (CABLE Output)
# #     # ==========================================
# #     input_device_id = None
# #     device_list = sd.query_devices()

# #     print("\nScanning for 'CABLE Output'...")
# #     for i, device_info in enumerate(device_list):
# #         # Look for CABLE Output (which acts as input loopback)
# #         if "CABLE Output" in device_info["name"]:
# #             if device_info["max_input_channels"] > 0:
# #                 input_device_id = i
# #                 print(f"Found 'CABLE Output' device with ID: {input_device_id}")
# #                 break

# #     if input_device_id is None:
# #         print("Could not find 'CABLE Output'. Falling back to default system input.")
# #         input_device_id = sd.default.device[0]

# #     # ==========================================
# #     # GRAPH SETUP
# #     # ==========================================
# #     print("\nOpening Graph Window...")
# #     plt.ion()
# #     fig, ax = plt.subplots(figsize=(10, 4))

# #     y_data = deque([0] * HISTORY_SIZE, maxlen=HISTORY_SIZE)
# #     x_data = np.arange(HISTORY_SIZE)

# #     # "Step" plot for clean transitions between classes
# #     (line,) = ax.step(x_data, y_data, where="post", color="blue", linewidth=2)

# #     # Configure Axes
# #     ax.set_ylim(-0.5, 2.5)
# #     ax.set_yticks([0, 1, 2])
# #     ax.set_yticklabels([CLASS_LABELS[0], CLASS_LABELS[1], CLASS_LABELS[2]])
# #     ax.set_ylabel("Detected Class")
# #     ax.set_xlabel("Time (Chunks)")
# #     ax.set_title(f"Listening on Device {input_device_id}")
# #     ax.grid(True, axis="y", linestyle="--", alpha=0.7)

# #     # Start Stream
# #     stream = sd.InputStream(
# #         device=input_device_id, callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE
# #     )

# #     with stream:
# #         print(f"\nListening... Press Ctrl+C to stop.\n")
# #         print(f"{'PREDICTION':<15} | {'CONFIDENCE':<10} | {'RAW PROBS'}")
# #         print("-" * 50)

# #         while True:
# #             try:
# #                 audio_data = data_queue.get_nowait()

# #                 # --- INFERENCE ---
# #                 tensor = torch.tensor(audio_data).float().to(device)
# #                 spec = transform(tensor.unsqueeze(0))
# #                 spec = torch.log(spec + 1e-9).unsqueeze(1)

# #                 # Fix Shape (Force 48 frames)
# #                 target_width = 48
# #                 if spec.shape[3] > target_width:
# #                     spec = spec[:, :, :, :target_width]
# #                 elif spec.shape[3] < target_width:
# #                     spec = F.pad(spec, (0, target_width - spec.shape[3]))

# #                 with torch.no_grad():
# #                     output = model(spec)
# #                     probs = torch.softmax(output, dim=1).cpu().numpy()[0]
# #                     prediction = int(np.argmax(probs))
# #                     confidence = probs[prediction]

# #                 # --- UPDATE GRAPH ---
# #                 y_data.append(prediction)
# #                 line.set_ydata(y_data)

# #                 # --- TERMINAL DEBUG ---
# #                 raw_str = f"[{probs[0]:.2f}, {probs[1]:.2f}, {probs[2]:.2f}]"
# #                 print(f"{CLASS_LABELS[prediction]:<15} | {confidence:.2f}       | {raw_str}")

# #                 # Redraw
# #                 fig.canvas.draw_idle()
# #                 fig.canvas.flush_events()

# #             except queue.Empty:
# #                 plt.pause(0.01)
# #             except Exception as e:
# #                 print(f"Error: {e}")
# #                 break


# # if __name__ == "__main__":
# #     try:
# #         main()
# #     except KeyboardInterrupt:
# #         print("\nStopped.")

# # # import sounddevice as sd
# # # import numpy as np
# # # import torch
# # # import torch.nn as nn
# # # import torch.nn.functional as F
# # # import torchaudio
# # # import matplotlib.pyplot as plt
# # # import queue
# # # import sys
# # # from collections import deque
# # # from pathlib import Path

# # # # --- 1. CONFIGURATION ---
# # # # Use raw string r"..." to fix Windows path issues
# # # MODEL_PATH = Path("C:/Users/egrandjean/Desktop/BattleSound_model/BattleSound/models/best_model.pt")
# # # SAMPLE_RATE = 16000
# # # DURATION = 0.5
# # # CHUNK_SIZE = int(SAMPLE_RATE * DURATION)
# # # CLASSES = ["Class 0", "Class 1", "Class 2"]
# # # HISTORY_SIZE = 50  # How many points to show on graph (50 * 0.5s = 25 seconds history)

# # # data_queue = queue.Queue()


# # # # --- 2. MODEL DEFINITION (Corrected: 3 Layers, Kernel 5x5) ---
# # # class Conv2DNet(nn.Module):
# # #     def __init__(self, num_class=3):
# # #         super(Conv2DNet, self).__init__()

# # #         # Layer 1: Conv(1->10, k=5) -> BN -> ReLU -> MaxPool
# # #         self.layer1 = nn.Sequential(
# # #             nn.Conv2d(1, 10, kernel_size=5, stride=1, padding=2), nn.BatchNorm2d(10), nn.ReLU(), nn.MaxPool2d(2)
# # #         )

# # #         # Layer 2: Conv(10->20, k=5) -> BN -> ReLU -> MaxPool
# # #         self.layer2 = nn.Sequential(
# # #             nn.Conv2d(10, 20, kernel_size=5, stride=1, padding=2), nn.BatchNorm2d(20), nn.ReLU(), nn.MaxPool2d(2)
# # #         )

# # #         # Layer 3: Conv(20->40, k=5) -> BN -> ReLU -> MaxPool
# # #         self.layer3 = nn.Sequential(
# # #             nn.Conv2d(20, 40, kernel_size=5, stride=1, padding=2), nn.BatchNorm2d(40), nn.ReLU(), nn.MaxPool2d(2)
# # #         )

# # #         # FC Layers (Weights match 1200 inputs)
# # #         self.fc1 = nn.Linear(1200, 256)
# # #         self.fc2 = nn.Linear(256, num_class)
# # #         self.dropout = nn.Dropout(0.5)

# # #     def forward(self, x):
# # #         x = self.layer1(x)
# # #         x = self.layer2(x)
# # #         x = self.layer3(x)
# # #         x = x.view(x.size(0), -1)
# # #         x = F.relu(self.fc1(x))
# # #         x = self.dropout(x)
# # #         x = self.fc2(x)
# # #         return x


# # # # --- 3. LOAD MODEL (Robust Version) ---
# # # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # print(f"Loading model on {device}...")

# # # model = Conv2DNet(num_class=len(CLASSES)).to(device)
# # # checkpoint = torch.load(MODEL_PATH, map_location=device)

# # # # Handle different save formats (List vs Dict)
# # # if isinstance(checkpoint, list):
# # #     model.load_state_dict(checkpoint[0])
# # # elif isinstance(checkpoint, dict) and "model" in checkpoint:
# # #     if isinstance(checkpoint["model"], list):
# # #         model.load_state_dict(checkpoint["model"][0])
# # #     else:
# # #         model.load_state_dict(checkpoint["model"])
# # # else:
# # #     model.load_state_dict(checkpoint)

# # # model.eval()

# # # # Spectrogram Transform
# # # transform = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=512, hop_length=128, n_mels=40).to(
# # #     device
# # # )


# # # # --- 4. AUDIO CALLBACK ---
# # # def audio_callback(indata, frames, time, status):
# # #     if status:
# # #         print(status, file=sys.stderr)
# # #     data_queue.put(indata[:, 0].copy())


# # # # --- 5. MAIN VISUALIZATION LOOP ---
# # # def main():
# # #     print("Opening Rolling Graph Window...")
# # #     plt.ion()
# # #     fig, ax = plt.subplots(figsize=(10, 5))

# # #     # Setup Data buffers for 3 Classes
# # #     y_data = [deque([0] * HISTORY_SIZE, maxlen=HISTORY_SIZE) for _ in range(len(CLASSES))]
# # #     x_data = list(range(HISTORY_SIZE))

# # #     # Create Lines
# # #     lines = []
# # #     colors = ["r", "g", "b"]  # Class 0=Red, 1=Green, 2=Blue
# # #     for i in range(len(CLASSES)):
# # #         (line,) = ax.plot(x_data, y_data[i], label=CLASSES[i], color=colors[i], linewidth=2)
# # #         lines.append(line)

# # #     ax.set_ylim(0, 1.1)
# # #     ax.set_ylabel("Probability")
# # #     ax.set_xlabel("Time (Frames)")
# # #     ax.legend(loc="upper right")
# # #     ax.set_title("Real-Time Event Detection")
# # #     ax.grid(True)

# # #     stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE)

# # #     with stream:
# # #         print("Listening... Close the graph window to stop.")
# # #         while True:
# # #             try:
# # #                 audio_data = data_queue.get_nowait()

# # #                 # --- INFERENCE ---
# # #                 tensor = torch.tensor(audio_data).float().to(device)
# # #                 spec = transform(tensor.unsqueeze(0))
# # #                 spec = torch.log(spec + 1e-9).unsqueeze(1)

# # #                 # *** CRITICAL FIX: FORCE SHAPE TO 48 FRAMES ***
# # #                 # Your microphone gives ~56 frames. The model needs exactly 48.
# # #                 target_width = 48
# # #                 current_width = spec.shape[3]

# # #                 if current_width > target_width:
# # #                     spec = spec[:, :, :, :target_width]  # Crop
# # #                 elif current_width < target_width:
# # #                     pad_amount = target_width - current_width
# # #                     spec = F.pad(spec, (0, pad_amount))  # Pad

# # #                 # Run Model
# # #                 with torch.no_grad():
# # #                     output = model(spec)
# # #                     probs = torch.softmax(output, dim=1).cpu().numpy()[0]

# # #                 # --- UPDATE GRAPH ---
# # #                 for i in range(len(CLASSES)):
# # #                     y_data[i].append(probs[i])  # Add new point
# # #                     lines[i].set_ydata(y_data[i])  # Update line

# # #                 fig.canvas.draw_idle()
# # #                 fig.canvas.flush_events()

# # #             except queue.Empty:
# # #                 plt.pause(0.01)
# # #             except Exception as e:
# # #                 print(f"Error: {e}")
# # #                 break


# # # if __name__ == "__main__":
# # #     try:
# # #         main()
# # #     except KeyboardInterrupt:
# # #         print("\nStopped.")


# # # # import sounddevice as sd
# # # # import numpy as np
# # # # import torch
# # # # import torch.nn as nn
# # # # import torchaudio
# # # # import torch.nn.functional as F


# # # # # --- 1. MODEL DEFINITION (Must match your training) ---
# # # # class Conv2DNet(nn.Module):
# # # #     def __init__(self, num_class=3):
# # # #         super(Conv2DNet, self).__init__()
# # # #         # Adjusted to match the 1200 input size we found earlier
# # # #         self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# # # #         self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# # # #         self.pool = nn.MaxPool2d(2, 2)

# # # #         # Calculate flat features based on input 128xSomething
# # # #         # For 0.5s duration, we determined this leads to 1200 features
# # # #         self.fc1 = nn.Linear(1200, 256)
# # # #         self.fc2 = nn.Linear(256, num_class)
# # # #         self.dropout = nn.Dropout(0.5)

# # # #     def forward(self, x):
# # # #         # x shape: [Batch, 1, Freq, Time]
# # # #         x = self.pool(F.relu(self.conv1(x)))
# # # #         x = self.pool(F.relu(self.conv2(x)))
# # # #         x = x.view(x.size(0), -1)  # Flatten
# # # #         x = F.relu(self.fc1(x))
# # # #         x = self.dropout(x)
# # # #         x = self.fc2(x)
# # # #         return x


# # # # # --- 2. CONFIGURATION ---
# # # # MODEL_PATH = "best_model.pt"
# # # # SAMPLE_RATE = 16000
# # # # DURATION = 0.5  # Seconds (matches your training)
# # # # CHUNK_SIZE = int(SAMPLE_RATE * DURATION)
# # # # CLASSES = ["Class 0", "Class 1", "Class 2"]  # Rename these to Bomb, Gunshot, etc.

# # # # # Load Model
# # # # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # # print(f"Loading model on {device}...")

# # # # # Initialize and load weights
# # # # model = Conv2DNet(num_class=len(CLASSES)).to(device)
# # # # checkpoint = torch.load(MODEL_PATH, map_location=device)

# # # # # Handle loading state_dict whether it's wrapped in 'model' key or not
# # # # if "model" in checkpoint:
# # # #     # If the saved dict is a list (DataParallel), take the first element
# # # #     if isinstance(checkpoint["model"], list):
# # # #         model.load_state_dict(checkpoint["model"][0])
# # # #     else:
# # # #         model.load_state_dict(checkpoint["model"])
# # # # else:
# # # #     model.load_state_dict(checkpoint)

# # # # model.eval()

# # # # # Spectrogram Transformer (Matches BattleSound config)
# # # # transform = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=512, hop_length=128, n_mels=40).to(
# # # #     device
# # # # )


# # # # def process_audio(indata, frames, time, status):
# # # #     if status:
# # # #         print(status)

# # # #     # 1. Prepare Audio Tensor
# # # #     # indata is shape (Frames, Channels). Take channel 0.
# # # #     audio_tensor = torch.tensor(indata[:, 0]).float().to(device)

# # # #     # 2. Preprocess (Make Spectrogram)
# # # #     # Unsqueeze to add batch dimension [1, Length]
# # # #     spec = transform(audio_tensor.unsqueeze(0))
# # # #     # Log conversion (Log-Mel)
# # # #     spec = torch.log(spec + 1e-9)
# # # #     # Add channel dim: [1, 1, Freq, Time]
# # # #     spec = spec.unsqueeze(1)

# # # #     # 3. Inference
# # # #     with torch.no_grad():
# # # #         output = model(spec)
# # # #         probs = torch.softmax(output, dim=1).cpu().numpy()[0]
# # # #         prediction = np.argmax(probs)

# # # #     # 4. Display Result
# # # #     score = probs[prediction]
# # # #     if score > 0.5:  # Only print if confident
# # # #         print(f"Detected: {CLASSES[prediction]} ({score:.2f})")
# # # #         # Visual Bar
# # # #         bar = ["|" if i == prediction else "." for i in range(len(CLASSES))]
# # # #         print(f"[{''.join(bar)}] {probs}")


# # # # # --- 3. START STREAM ---
# # # # print(f"Listening... (Press Ctrl+C to stop)")
# # # # try:
# # # #     with sd.InputStream(callback=process_audio, channels=1, samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE):
# # # #         while True:
# # # #             sd.sleep(1000)
# # # # except KeyboardInterrupt:
# # # #     print("\nStopped.")
# # # # except Exception as e:
# # # #     print(f"Error: {e}")
