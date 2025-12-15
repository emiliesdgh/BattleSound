import sounddevice as sd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa
import queue
import sys
from collections import deque
from pathlib import Path

# --- 1. CONFIGURATION ---
MODEL_PATH = Path("C:/Users/egrandjean/Desktop/BattleSound_model/BattleSound/benchmark/checkpoint/128_x_64_2D.tflite")
SAMPLE_RATE = 16000
DURATION = 0.5
CHUNK_SIZE = int(SAMPLE_RATE * DURATION)
HISTORY_SIZE = 100

data_queue = queue.Queue()

# --- 2. LOAD TFLITE MODEL ---
print("Loading TFLite model...")
try:
    interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]["shape"]
output_shape = output_details[0]["shape"]

print(f"Model Input Shape: {input_shape}")
print(f"Model Output Shape: {output_shape}")

# DETECT NUMBER OF CLASSES AUTOMATICALLY
NUM_CLASSES = output_shape[1]
print(f"Detected {NUM_CLASSES} Classes.")
# Create generic labels if we don't know them
CLASSES = [f"Class {i}" for i in range(NUM_CLASSES)]

TARGET_MELS = input_shape[1]
TARGET_FRAMES = input_shape[2]


# --- 3. PREPROCESSING ---
def get_spectrogram(audio):
    melspec = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=TARGET_MELS, n_fft=512, hop_length=128)
    log_mel = librosa.power_to_db(melspec, ref=np.max)
    current_width = log_mel.shape[1]
    if current_width < TARGET_FRAMES:
        pad_width = TARGET_FRAMES - current_width
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)))
    else:
        log_mel = log_mel[:, :TARGET_FRAMES]
    return log_mel.reshape(input_shape).astype(np.float32)


def audio_callback(indata, frames, time, status):
    if status:
        print(f"Audio Status: {status}", file=sys.stderr)
    data_queue.put(indata[:, 0].copy())


# --- 4. MAIN LOOP ---
def main():
    # DEVICE SELECTION
    input_device_id = None
    device_list = sd.query_devices()
    print("\nScanning for 'CABLE Output'...")
    for i, device_info in enumerate(device_list):
        if "CABLE Output" in device_info["name"]:
            if device_info["max_input_channels"] > 0:
                input_device_id = i
                print(f"Found 'CABLE Output' with ID: {input_device_id}")
                break
    if input_device_id is None:
        input_device_id = sd.default.device[0]
        print(f"Using Default Device: {input_device_id}")

    # GRAPH
    print("\nOpening Graph Window...")
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 4))
    y_data = deque([0] * HISTORY_SIZE, maxlen=HISTORY_SIZE)
    x_data = np.arange(HISTORY_SIZE)
    (line,) = ax.step(x_data, y_data, where="post", color="green", linewidth=2)
    ax.set_ylim(-0.5, NUM_CLASSES - 0.5)
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_yticklabels(CLASSES)
    ax.set_title(f"TFLite ({NUM_CLASSES} Classes)")
    ax.grid(True, axis="y", linestyle="--")

    stream = sd.InputStream(
        device=input_device_id, callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE
    )

    with stream:
        print(f"\nListening... Press Ctrl+C to stop.\n")
        print(f"{'VOL':<8} | {'PREDICTION':<15} | {'CONF':<6} | {'RAW PROBS'}")
        print("-" * 60)

        while True:
            try:
                audio_raw = data_queue.get_nowait()

                # CHECK VOLUME (RMS)
                rms = np.sqrt(np.mean(audio_raw**2))

                input_tensor = get_spectrogram(audio_raw)
                interpreter.set_tensor(input_details[0]["index"], input_tensor)
                interpreter.invoke()
                probs = interpreter.get_tensor(output_details[0]["index"])[0]

                prediction = int(np.argmax(probs))
                confidence = probs[prediction]

                y_data.append(prediction)
                line.set_ydata(y_data)

                # DYNAMIC PRINTING (Fixes Index Error)
                probs_str = "[" + ", ".join([f"{p:.2f}" for p in probs]) + "]"
                print(f"{rms:.4f}   | {CLASSES[prediction]:<15} | {confidence:.2f}   | {probs_str}")

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
# import tensorflow as tf  # Or import tflite_runtime.interpreter as tflite
# import matplotlib.pyplot as plt
# import librosa
# import queue
# import sys
# from collections import deque
# from pathlib import Path

# # --- 1. CONFIGURATION ---
# # Use raw string r"..." for Windows paths
# MODEL_PATH = Path("C:/Users/egrandjean/Desktop/BattleSound_model/BattleSound/benchmark/checkpoint/128_x_64_2D.tflite")
# SAMPLE_RATE = 16000
# DURATION = 0.5
# CHUNK_SIZE = int(SAMPLE_RATE * DURATION)

# # Mapping: 0=Nothing, 1=Voices, 2=Effects
# CLASS_LABELS = {0: "Nothing", 1: "Voices", 2: "Effects"}
# CLASSES = ["Nothing", "Voices", "Effects"]
# HISTORY_SIZE = 100

# data_queue = queue.Queue()

# # --- 2. LOAD TFLITE MODEL ---
# print("Loading TFLite model...")
# try:
#     interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
#     interpreter.allocate_tensors()
# except Exception as e:
#     print(f"Error loading model: {e}")
#     sys.exit(1)

# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
# input_shape = input_details[0]["shape"]

# print(f"Model Input Shape: {input_shape}")
# # TFLite shape is usually [Batch, Height, Width, Channels]
# # For audio, Height is usually n_mels, Width is Time frames.
# TARGET_MELS = input_shape[1]
# TARGET_FRAMES = input_shape[2]


# # --- 3. PREPROCESSING FUNCTION (Librosa) ---
# def get_spectrogram(audio):
#     """
#     Converts raw audio to the exact Log-Mel Spectrogram the TFLite model expects.
#     """
#     # 1. Compute Mel Spectrogram
#     melspec = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=TARGET_MELS, n_fft=512, hop_length=128)

#     # 2. Convert to Log Scale (dB)
#     log_mel = librosa.power_to_db(melspec, ref=np.max)

#     # 3. Resize/Crop to match TFLite Input Width (Time)
#     current_width = log_mel.shape[1]

#     if current_width < TARGET_FRAMES:
#         # Pad with zeros if too short
#         pad_width = TARGET_FRAMES - current_width
#         log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)))
#     else:
#         # Crop if too long
#         log_mel = log_mel[:, :TARGET_FRAMES]

#     # 4. Reshape to [1, Height, Width, Channels] (Standard TFLite Format)
#     # Most TFLite models expect a "Channel Last" format or a specific 4D shape
#     # We reshape to whatever input_shape the model demands.
#     input_data = log_mel.reshape(input_shape).astype(np.float32)
#     return input_data


# # --- 4. AUDIO CALLBACK ---
# def audio_callback(indata, frames, time, status):
#     if status:
#         print(f"Audio Status: {status}", file=sys.stderr)
#     data_queue.put(indata[:, 0].copy())


# # --- 5. MAIN LOOP ---
# def main():
#     # ==========================================
#     # AUTOMATIC DEVICE SELECTION (CABLE Output)
#     # ==========================================
#     input_device_id = None
#     device_list = sd.query_devices()

#     print("\nScanning for 'CABLE Output'...")
#     for i, device_info in enumerate(device_list):
#         if "CABLE Output" in device_info["name"]:
#             if device_info["max_input_channels"] > 0:
#                 input_device_id = i
#                 print(f"Found 'CABLE Output' device with ID: {input_device_id}")
#                 break

#     if input_device_id is None:
#         print("Could not find 'CABLE Output'. Falling back to default system input.")
#         input_device_id = sd.default.device[0]

#     # ==========================================
#     # GRAPH SETUP
#     # ==========================================
#     print("\nOpening Graph Window...")
#     plt.ion()
#     fig, ax = plt.subplots(figsize=(10, 4))

#     y_data = deque([0] * HISTORY_SIZE, maxlen=HISTORY_SIZE)
#     x_data = np.arange(HISTORY_SIZE)

#     # "Step" plot for clean transitions
#     (line,) = ax.step(x_data, y_data, where="post", color="green", linewidth=2)

#     # Configure Axes
#     ax.set_ylim(-0.5, 2.5)
#     ax.set_yticks([0, 1, 2])
#     ax.set_yticklabels([CLASS_LABELS[0], CLASS_LABELS[1], CLASS_LABELS[2]])
#     ax.set_ylabel("Detected Class")
#     ax.set_xlabel("Time (Chunks)")
#     ax.set_title(f"TFLite Inference on Device {input_device_id}")
#     ax.grid(True, axis="y", linestyle="--", alpha=0.7)

#     # Start Stream
#     stream = sd.InputStream(
#         device=input_device_id, callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE
#     )

#     with stream:
#         print(f"\nListening... Press Ctrl+C to stop.\n")
#         print(f"{'PREDICTION':<15} | {'CONFIDENCE':<10} | {'RAW PROBS'}")
#         print("-" * 50)

#         while True:
#             try:
#                 # Get raw audio from queue
#                 audio_raw = data_queue.get_nowait()

#                 # --- PREPROCESSING (Librosa) ---
#                 # Convert raw wav -> Spectrogram -> Reshape to Model Input
#                 input_tensor = get_spectrogram(audio_raw)

#                 # --- INFERENCE (TFLite) ---
#                 interpreter.set_tensor(input_details[0]["index"], input_tensor)
#                 interpreter.invoke()
#                 output_data = interpreter.get_tensor(output_details[0]["index"])

#                 # Process results
#                 probs = output_data[0]  # List of probabilities
#                 prediction = int(np.argmax(probs))
#                 confidence = probs[prediction]

#                 # --- UPDATE GRAPH ---
#                 y_data.append(prediction)
#                 line.set_ydata(y_data)

#                 # --- TERMINAL DEBUG ---
#                 # Safe printing even if model outputs weird probability ranges
#                 raw_str = f"[{probs[0]:.2f}, {probs[1]:.2f}, {probs[2]:.2f}]"
#                 print(f"{CLASS_LABELS[prediction]:<15} | {confidence:.2f}       | {raw_str}")

#                 # Redraw
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

# # import sounddevice as sd
# # import numpy as np
# # import tensorflow as tf  # Or import tflite_runtime.interpreter as tflite

# # from pathlib import Path

# # # --- CONFIGURATION ---
# # # MODEL_PATH = "128_x_64_2D.tflite"  # Change to "128_x_128_2D.tflite" if needed
# # MODEL_PATH = Path("C:/Users/egrandjean/Desktop/BattleSound_model/BattleSound/benchmark/checkpoint/128_x_64_2D.tflite")
# # SAMPLE_RATE = 16000
# # DURATION = 0.5
# # CHUNK_SIZE = int(SAMPLE_RATE * DURATION)
# # CLASSES = ["Class 0", "Class 1", "Class 2"]

# # # --- 1. LOAD TFLITE MODEL ---
# # print("Loading TFLite model...")
# # interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
# # interpreter.allocate_tensors()

# # input_details = interpreter.get_input_details()
# # output_details = interpreter.get_output_details()
# # input_shape = input_details[0]["shape"]

# # print(f"Model Input Shape: {input_shape}")

# # # Pre-calculate Mel Filterbank for numpy (since we can't use torchaudio here easily)
# # # Simplification: We will try to map raw audio to the shape the TFLite model expects.
# # # Note: TFLite models usually expect the EXACT spectrogram input.
# # # If your TFLite model doesn't include the spectrogram layer inside it,
# # # we need to compute it manually in numpy.


# # def get_spectrogram(audio):
# #     # This is a basic numpy implementation of a Log-Mel Spectrogram
# #     # to match typical audio model inputs.

# #     # 1. FFT
# #     # Simple spectrogram using matplotlib logic or scipy
# #     # For now, we use a placeholder reshape if the model expects raw audio,
# #     # OR we assume the model takes spectrograms.

# #     # CHECK: Does the TFLite model take Raw Audio [1, N] or Image [1, H, W, C]?
# #     # Based on the filename "128_x_64_2D", it expects an IMAGE (Spectrogram).

# #     import librosa  # We use librosa for robust numpy feature extraction

# #     # Compute Mel Spectrogram
# #     melspec = librosa.feature.melspectrogram(
# #         y=audio, sr=SAMPLE_RATE, n_mels=input_shape[1], n_fft=512, hop_length=128  # Usually 128 or 40
# #     )
# #     log_mel = librosa.power_to_db(melspec, ref=np.max)

# #     # Resize/Crop to match input shape
# #     # The model expects specific dimensions (e.g. 128x64)
# #     target_width = input_shape[2]

# #     if log_mel.shape[1] < target_width:
# #         # Pad
# #         pad_width = target_width - log_mel.shape[1]
# #         log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)))
# #     else:
# #         # Crop
# #         log_mel = log_mel[:, :target_width]

# #     # Reshape for TFLite [1, Height, Width, Channels]
# #     input_data = log_mel.reshape(input_shape).astype(np.float32)
# #     return input_data


# # def process_audio(indata, frames, time, status):
# #     if status:
# #         print(status)

# #     audio_data = indata[:, 0].flatten().astype(np.float32)

# #     try:
# #         # Prepare Input
# #         input_data = get_spectrogram(audio_data)

# #         # Set Tensor
# #         interpreter.set_tensor(input_details[0]["index"], input_data)

# #         # Run Inference
# #         interpreter.invoke()

# #         # Get Result
# #         output_data = interpreter.get_tensor(output_details[0]["index"])
# #         probs = output_data[0]
# #         prediction = np.argmax(probs)

# #         # Display
# #         if probs[prediction] > 0.5:
# #             print(f"TFLite: {CLASSES[prediction]} ({probs[prediction]:.2f})")

# #     except Exception as e:
# #         # Usually happens if audio chunk is silence/zeros
# #         pass


# # # --- START ---
# # print(f"Listening... (TFLite)")
# # with sd.InputStream(callback=process_audio, channels=1, samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE):
# #     while True:
# #         sd.sleep(1000)
