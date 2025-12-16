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

# AUDIO RATES
SYSTEM_RATE = 48000  # Input device rate
MODEL_RATE = 16000  # AI Model rate
DURATION = 0.5  # Window size for prediction
UPDATE_INTERVAL = 0.1  # How often we predict (Sliding Window step)

# CALCULATED SIZES
WINDOW_SIZE = int(SYSTEM_RATE * DURATION)  # 24000 samples (0.5s)
STEP_SIZE = int(SYSTEM_RATE * UPDATE_INTERVAL)  # 4800 samples (0.1s)

CLASS_LABELS = {0: "Nothing", 1: "Voices", 2: "Effects"}
HISTORY_SIZE = 100
GAIN_FACTOR = 3.0  # Reduced slightly since we are filtering noise

data_queue = queue.Queue()


# --- MODEL ---
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
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
model.eval()

# # --- TRANSFORMS ---
# resampler = torchaudio.transforms.Resample(orig_freq=SYSTEM_RATE, new_freq=MODEL_RATE).to(device)
# transform = torchaudio.transforms.MelSpectrogram(sample_rate=MODEL_RATE, n_fft=512, hop_length=128, n_mels=40).to(
#     device
# )
# --- TRANSFORMS ---
resampler = torchaudio.transforms.Resample(orig_freq=SYSTEM_RATE, new_freq=MODEL_RATE).to(device)

# Use AmplitudeToDB instead of manual Log. This matches standard training pipelines.
to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80).to(device)

transform = torchaudio.transforms.MelSpectrogram(sample_rate=MODEL_RATE, n_fft=512, hop_length=128, n_mels=40).to(
    device
)


# Callback: Pushes small 0.1s chunks
def audio_callback(indata, frames, time, status):
    if status:
        print(f"Status: {status}", file=sys.stderr)
    data_queue.put(indata[:, 0].copy())


def main():
    # DEVICE SELECT
    input_device_id = sd.default.device[0]
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if "CABLE Output" in dev["name"] and dev["max_input_channels"] > 0:
            input_device_id = i
            print(f"Found CABLE Output: {i}")
            break

    # PLOT SETUP
    print("Opening Graph...")
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    # Prediction History
    y_data = deque([0] * HISTORY_SIZE, maxlen=HISTORY_SIZE)
    (line,) = ax1.step(np.arange(HISTORY_SIZE), y_data, where="post", color="cyan", linewidth=2)
    ax1.set_ylim(-0.5, 2.5)
    ax1.set_yticks([0, 1, 2])
    ax1.set_yticklabels(list(CLASS_LABELS.values()))
    ax1.set_title("Live Prediction (Sliding Window)")

    # Spectrogram
    dummy_spec = np.zeros((40, 63))
    im = ax2.imshow(dummy_spec, aspect="auto", origin="lower", cmap="inferno", vmin=-11.5, vmax=2.5)
    ax2.set_title("Input (Filtered > 300Hz)")
    ax2.set_xlabel("Time")

    # BUFFER INITIALIZATION
    # We hold 0.5s of audio in this buffer
    rolling_buffer = np.zeros(WINDOW_SIZE, dtype=np.float32)

    stream = sd.InputStream(
        device=input_device_id,
        callback=audio_callback,
        channels=1,
        samplerate=SYSTEM_RATE,
        blocksize=STEP_SIZE,  # Captures small 0.1s chunks
    )

    with stream:
        print("\nListening...")
        print(f"{'RMS':<8} | {'PRED':<10} | {'CONF':<6}")
        print("-" * 30)

        while True:
            try:
                # 1. Get new small chunk (0.1s)
                new_chunk = data_queue.get_nowait()

                # 2. Update Rolling Buffer (Shift Left, Append New)
                rolling_buffer = np.roll(rolling_buffer, -len(new_chunk))
                rolling_buffer[-len(new_chunk) :] = new_chunk

                # 3. Apply Gain
                audio_proc = rolling_buffer * GAIN_FACTOR
                audio_proc = np.clip(audio_proc, -1.0, 1.0)

                # Check volume of just the new bit
                rms = np.sqrt(np.mean(new_chunk**2))

                # # 4. Process on GPU
                # audio_tens = torch.tensor(audio_proc).float().to(device)

                # # --- FREQUENCY FILTER (The Fix for Engine Noise) ---
                # # High-pass filter at 300Hz to kill engine rumble
                # audio_tens = torchaudio.functional.highpass_biquad(audio_tens, SYSTEM_RATE, cutoff_freq=300)

                # # 5. Resample & Spectrogram
                # audio_resampled = resampler(audio_tens)
                # spec = transform(audio_resampled.unsqueeze(0))
                # spec = torch.log(spec + 1e-9).unsqueeze(1)

                # # 6. Shape Check
                # if spec.shape[3] > 48:
                #     spec_in = spec[:, :, :, :48]
                # elif spec.shape[3] < 48:
                #     spec_in = F.pad(spec, (0, 48 - spec.shape[3]))
                # else:
                #     spec_in = spec

                # # 7. Inference
                # with torch.no_grad():
                #     probs = torch.softmax(model(spec_in), dim=1).cpu().numpy()[0]
                #     pred = int(np.argmax(probs))

                # 4. Process on GPU
                audio_tens = torch.tensor(audio_proc).float().to(device)

                # High-pass filter (Keep this, it's good for engine noise)
                audio_tens = torchaudio.functional.highpass_biquad(audio_tens, SYSTEM_RATE, cutoff_freq=300)

                # 5. Resample & Spectrogram
                audio_resampled = resampler(audio_tens)
                spec = transform(audio_resampled.unsqueeze(0))

                # CRITICAL CHANGE: Use DB scale, not raw Log
                spec = to_db(spec).unsqueeze(1)

                # 6. Shape Fix (CRITICAL: Slice from the END)
                # We want the LAST 48 frames (the newest audio), not the first 48
                if spec.shape[3] > 48:
                    spec_in = spec[:, :, :, -48:]  # <--- Changed :48 to -48:
                elif spec.shape[3] < 48:
                    spec_in = F.pad(spec, (0, 48 - spec.shape[3]))
                else:
                    spec_in = spec

                # 7. Inference
                with torch.no_grad():
                    # Check what the model actually outputs
                    logits = model(spec_in)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    pred = int(np.argmax(probs))

                # Update Plots
                y_data.append(pred)
                line.set_ydata(y_data)

                spec_vis = spec_in.squeeze().cpu().numpy()
                im.set_data(spec_vis)
                im.set_clim(vmin=spec_vis.min(), vmax=spec_vis.max())

                if pred != 0:  # Print only interesting events
                    print(f"{rms:.4f}   | {CLASS_LABELS[pred]:<10} | {probs[pred]:.2f}")

                fig.canvas.draw_idle()
                fig.canvas.flush_events()

            except queue.Empty:
                plt.pause(0.001)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(e)
                break


if __name__ == "__main__":
    main()

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

# # AUDIO RATES
# SYSTEM_RATE = 48000  # Input device rate (e.g. CABLE Output)
# MODEL_RATE = 16000  # AI Model training rate
# DURATION = 0.5  # Seconds per chunk
# CAPTURE_SIZE = int(SYSTEM_RATE * DURATION)  # Samples to capture per chunk

# CLASS_LABELS = {0: "Nothing", 1: "Voices", 2: "Effects"}
# HISTORY_SIZE = 100
# GAIN_FACTOR = 5.0

# data_queue = queue.Queue()


# # --- MODEL DEFINITION ---
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


# # --- INITIALIZATION ---
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Loading model on {device}...")

# model = Conv2DNet(3).to(device)
# try:
#     ckpt = torch.load(MODEL_PATH, map_location=device)
#     if isinstance(ckpt, list):
#         model.load_state_dict(ckpt[0])
#     elif isinstance(ckpt, dict) and "model" in ckpt:
#         model.load_state_dict(ckpt["model"][0] if isinstance(ckpt["model"], list) else ckpt["model"])
#     else:
#         model.load_state_dict(ckpt)
#     print("Model loaded.")
# except Exception as e:
#     print(f"Error loading model: {e}")
#     sys.exit(1)
# model.eval()

# # --- TRANSFORMS ---
# # 1. Resampler: 48k -> 16k
# resampler = torchaudio.transforms.Resample(orig_freq=SYSTEM_RATE, new_freq=MODEL_RATE).to(device)

# # 2. MelSpectrogram: Standard settings for 16k audio
# transform = torchaudio.transforms.MelSpectrogram(sample_rate=MODEL_RATE, n_fft=512, hop_length=128, n_mels=40).to(
#     device
# )


# def audio_callback(indata, frames, time, status):
#     if status:
#         print(f"Status: {status}", file=sys.stderr)

#     # Copy and Gain
#     audio_chunk = indata[:, 0].copy() * GAIN_FACTOR

#     # Clip to keep valid range
#     audio_chunk = np.clip(audio_chunk, -1.0, 1.0)

#     data_queue.put(audio_chunk)


# def main():
#     # DEVICE SELECTION
#     input_device_id = None
#     devices = sd.query_devices()
#     print("\nScanning devices...")
#     for i, dev in enumerate(devices):
#         if "CABLE Output" in dev["name"] and dev["max_input_channels"] > 0:
#             input_device_id = i
#             print(f"Found 'CABLE Output' ID: {i}")
#             break
#     if input_device_id is None:
#         input_device_id = sd.default.device[0]
#         print(f"Using Default Device ID: {input_device_id}")

#     # --- PLOT SETUP ---
#     print("Opening Visual Debugger...")
#     plt.ion()
#     # Create 2 subplots: Top for Predictions, Bottom for Spectrogram
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

#     # 1. Prediction Graph
#     y_data = deque([0] * HISTORY_SIZE, maxlen=HISTORY_SIZE)
#     (line,) = ax1.step(np.arange(HISTORY_SIZE), y_data, where="post", color="cyan", linewidth=2)
#     ax1.set_ylim(-0.5, 2.5)
#     ax1.set_yticks([0, 1, 2])
#     ax1.set_yticklabels(list(CLASS_LABELS.values()))
#     ax1.set_title("Live Prediction")
#     ax1.grid(True, alpha=0.3)

#     # 2. Spectrogram Graph
#     # Initialize with zeros. Shape: (n_mels=40, time_steps=48 approx)
#     dummy_spec = np.zeros((40, 63))  # 63 is approx width for 0.5s at 16k
#     im = ax2.imshow(dummy_spec, aspect="auto", origin="lower", cmap="inferno", vmin=-11.5, vmax=2.5)
#     ax2.set_title(f"What the AI Sees (MelSpec @ {MODEL_RATE}Hz)")
#     ax2.set_xlabel("Time")
#     ax2.set_ylabel("Mel Frequency")

#     plt.tight_layout()

#     # STREAM START
#     stream = sd.InputStream(
#         device=input_device_id, callback=audio_callback, channels=1, samplerate=SYSTEM_RATE, blocksize=CAPTURE_SIZE
#     )

#     with stream:
#         print(f"\nListening... Press Ctrl+C to stop.\n")
#         print(f"{'VOL':<8} | {'PREDICTION':<15} | {'CONF':<6}")
#         print("-" * 40)

#         while True:
#             try:
#                 # 1. Get Audio
#                 audio_np = data_queue.get_nowait()
#                 rms = np.sqrt(np.mean(audio_np**2))

#                 # 2. Process to Tensor
#                 audio_tens = torch.tensor(audio_np).float().to(device)

#                 # 3. Resample (Crucial Step)
#                 audio_resampled = resampler(audio_tens)

#                 # 4. Create Spectrogram
#                 spec = transform(audio_resampled.unsqueeze(0))
#                 spec = torch.log(spec + 1e-9).unsqueeze(1)  # Log scale

#                 # 5. Fix Dimensions for Model (Exact 48 width)
#                 # (We keep a copy for visualization before padding/cutting if we want,
#                 # but let's visualize exactly what goes into the model)
#                 if spec.shape[3] > 48:
#                     spec_in = spec[:, :, :, :48]
#                 elif spec.shape[3] < 48:
#                     spec_in = F.pad(spec, (0, 48 - spec.shape[3]))
#                 else:
#                     spec_in = spec

#                 # 6. Inference
#                 with torch.no_grad():
#                     probs = torch.softmax(model(spec_in), dim=1).cpu().numpy()[0]
#                     pred = int(np.argmax(probs))

#                 # --- UPDATE VISUALS ---

#                 # Update Line Graph
#                 y_data.append(pred)
#                 line.set_ydata(y_data)

#                 # Update Spectrogram
#                 # Squeeze to (40, 48) and move to CPU numpy
#                 spec_vis = spec_in.squeeze().cpu().numpy()
#                 im.set_data(spec_vis)
#                 im.set_clim(vmin=spec_vis.min(), vmax=spec_vis.max())  # Auto-contrast

#                 # Print Stats
#                 print(f"{rms:.4f}   | {CLASS_LABELS[pred]:<15} | {probs[pred]:.2f}")

#                 # Draw
#                 fig.canvas.draw_idle()
#                 fig.canvas.flush_events()

#             except queue.Empty:
#                 plt.pause(0.01)
#             except Exception as e:
#                 print(f"Error: {e}")
#                 break


# if __name__ == "__main__":
#     try:
#         main()
#     except KeyboardInterrupt:
#         print("\nStopped.")
