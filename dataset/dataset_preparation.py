import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

from pathlib import Path

# --- CONFIGURATION ---
# 1. Put your long raw files (mp3, wav, flac) here:
SOURCE_FOLDER = Path("C:/Users/egrandjean/Desktop/BattleSound_model/BattleSound/dataset/train/3_unsplit")

# 2. This is where the sliced .wavs will go (e.g., your dataset/train/3 folder)
DEST_FOLDER = Path("C:/Users/egrandjean/Desktop/BattleSound_model/BattleSound/dataset/train/3")

# 3. BattleSound Requirements
TARGET_SR = 16000
CHUNK_SAMPLES = 8000  # 0.5 seconds


def slice_audio_to_wav():
    if not os.path.exists(SOURCE_FOLDER):
        print(f"Error: Source folder '{SOURCE_FOLDER}' does not exist.")
        return

    os.makedirs(DEST_FOLDER, exist_ok=True)

    # Get all audio files
    valid_exts = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
    files = [f for f in os.listdir(SOURCE_FOLDER) if f.lower().endswith(valid_exts)]

    print(f"Found {len(files)} files. Slicing into {DEST_FOLDER}...")

    chunk_counter = 0

    for fname in tqdm(files):
        path = os.path.join(SOURCE_FOLDER, fname)

        try:
            # 1. Load and Resample (Using librosa is safest)
            y, sr = librosa.load(path, sr=TARGET_SR, mono=True)
        except Exception as e:
            print(f"Skipping {fname}: {e}")
            continue

        # 2. Slice Logic
        total_samples = len(y)
        num_chunks = int(np.ceil(total_samples / CHUNK_SAMPLES))

        if num_chunks == 0:
            continue  # Skip empty files

        for i in range(num_chunks):
            start = i * CHUNK_SAMPLES
            end = start + CHUNK_SAMPLES

            chunk = y[start:end]

            # Check for Silence (Don't save dead air)
            if np.max(np.abs(chunk)) < 0.005:
                continue

            # Pad if too short (Standardize length to 0.5s)
            if len(chunk) < CHUNK_SAMPLES:
                pad_amt = CHUNK_SAMPLES - len(chunk)
                chunk = np.pad(chunk, (0, pad_amt), mode="constant")

            # 3. Save as standard .WAV
            # We use soundfile to write clean 16-bit PCM wavs
            save_name = f"other_slice_{chunk_counter:06d}.wav"
            save_path = os.path.join(DEST_FOLDER, save_name)

            sf.write(save_path, chunk, TARGET_SR)
            chunk_counter += 1

    print(f"\nDone! Created {chunk_counter} sliced wav files.")
    print("You can listen to them now to verify content.")
    print("NEXT STEP: Run 'convert_dataset.py' to turn these into .npz files.")


if __name__ == "__main__":
    slice_audio_to_wav()
