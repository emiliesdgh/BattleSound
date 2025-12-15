import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np

# CONFIG
SAMPLE_RATE = 16000
DURATION = 5  # Seconds
OUTPUT_FILENAME = "debug_what_ai_hears.wav"

# FIND DEVICE
input_device_id = None
device_list = sd.query_devices()
for i, device_info in enumerate(device_list):
    if "CABLE Output" in device_info["name"] and device_info["max_input_channels"] > 0:
        input_device_id = i
        print(f"Found CABLE Output at ID {i}")
        break

if input_device_id is None:
    input_device_id = sd.default.device[0]
    print(f"Using default device ID {input_device_id}")

print(f"\nRecording 5 seconds... PLAY YOUR SOUNDS NOW!")
recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, device=input_device_id)
sd.wait()  # Wait until recording is finished
print("Recording finished.")

# NORMALIZE AND SAVE
# We save it exactly as received to check volume levels
wav.write(OUTPUT_FILENAME, SAMPLE_RATE, recording)
print(f"Saved to {OUTPUT_FILENAME}")
print("1. Open this file.")
print("2. If it is SILENT -> Check Windows Sound Settings (Input Device).")
print("3. If it is VERY QUIET -> We need to add Gain to the script.")
print("4. If it sounds SLOW/FAST -> Sample Rate mismatch.")
