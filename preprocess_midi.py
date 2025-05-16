'''
create emopia_data_structured.npz on disk which contains triples (spectrogram, MIDI tokens input, MIDI tokens targets)
'''
import os 
import zipfile
import numpy as np
import pretty_midi
import librosa
from miditoolkit import MidiFile
from tqdm import tqdm

# --- Paths ---
midi_zip_path = "midis.zip"
midi_extract_path = "midis_emopia"
dataset_save_path = "emopia_data_structured.npz"
failed_log_path = "failed_midis.txt"
os.makedirs(midi_extract_path, exist_ok=True)

# --- Unzip MIDI files ---
with zipfile.ZipFile(midi_zip_path, 'r') as zip_ref:
    zip_ref.extractall(midi_extract_path)

# --- Find MIDI files ---
midi_files = []
for root, _, files in os.walk(midi_extract_path):
    for file in files:
        if file.lower().endswith(('.mid', '.midi')):
            midi_files.append(os.path.join(root, file))

# --- Prepare storage ---
spectrograms = []
input_tokens = []
target_tokens = []
failed_files = []

# --- EMOPIA-style 8-feature MIDI representation ---
def extract_emopia_features(midi_path):
    midi = MidiFile(midi_path)
    notes = []
    for inst in midi.instruments:
        for note in inst.notes:
            notes.append([
                1 if not inst.is_drum else 0,              # is_instrument
                note.pitch,                                # pitch
                note.velocity // 2,                        # quantized velocity (0–63)
                int(note.start * 100),                     # quantized start time
                int(note.end * 100),                       # quantized end time
                inst.program,                              # instrument program
                int(midi.tempo_changes[1][0]) if len(midi.tempo_changes) > 1 else 120,  # tempo
                0                                          # placeholder (e.g. key signature)
            ])
    notes.sort(key=lambda x: x[3])  # sort by start
    return notes

# --- Process each MIDI file ---
for midi_path in tqdm(midi_files[:100]):
    try:
        # 1. Convert MIDI to waveform
        midi = pretty_midi.PrettyMIDI(midi_path)
        audio = midi.fluidsynth(fs=22050)

        # 2. Compute log-mel spectrogram
        mel = librosa.feature.melspectrogram(y=audio, sr=22050, n_mels=64)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = mel_db[:, :1024]
        if mel_db.shape[1] < 1024:
            mel_db = np.pad(mel_db, ((0, 0), (0, 1024 - mel_db.shape[1])), mode='constant')
        spectrograms.append(mel_db.T)  # (1024, 64)

        # 3. Extract 8-feature MIDI tokens
        emopia_tokens = extract_emopia_features(midi_path)
        if len(emopia_tokens) < 2:
            print(f"Too few tokens in: {midi_path}")
            failed_files.append(midi_path)
            continue

        # 4. Construct input/target sequences
        max_len = 1024
        sos = [[0] * 8]
        eos = [[1] * 8]

        in_tokens = sos + emopia_tokens[:-1]
        tgt_tokens = emopia_tokens + eos

        in_tokens = (in_tokens + [[0]*8] * max_len)[:max_len]
        tgt_tokens = (tgt_tokens + [[0]*8] * max_len)[:max_len]

        input_tokens.append(in_tokens)
        target_tokens.append(tgt_tokens)

    except Exception as e:
        print(f"Error processing {midi_path}: {e}")
        failed_files.append(midi_path)
        continue

# --- Save failed MIDI file paths ---
with open(failed_log_path, "w") as f:
    for path in failed_files:
        f.write(path + "\n")

# --- Check before saving ---
if not spectrograms or not input_tokens or not target_tokens:
    raise ValueError("No valid MIDI data processed. Check failed_midis.txt for issues.")

# --- Save dataset ---
X = np.array(spectrograms, dtype=np.float32)  # (N, 1024, 64)
Y = np.stack([
    np.stack([np.array(tgt), np.array(inp)], axis=-1)
    for tgt, inp in zip(target_tokens, input_tokens)
])  # (N, 1024, 8, 2)

np.savez_compressed(dataset_save_path, x=X, y=Y)
print(f"Saved to {dataset_save_path} with shapes: X.shape={X.shape}, Y.shape={Y.shape}")
