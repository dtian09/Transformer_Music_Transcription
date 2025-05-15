from utils import write_midi
import numpy as np
import pickle
from miditoolkit import MidiFile, Instrument, Note

# Load dataset and dictionary
data = np.load("co-representation/emopia_data.npz", allow_pickle=True)
with open("co-representation/dictionary.pkl", "rb") as f:
    _, idx_to_token = pickle.load(f)

#tokens = data["y"][0]
tokens = data["y"][1]

for row in tokens:
  print(row)

#write_midi(tokens, 'sample.mid', idx_to_token)
write_midi(tokens, 'sample1.mid', idx_to_token)

print(idx_to_token)