import numpy as np
import pickle
import json

# Load EMOPHIA data and dictionary
emopia_data = np.load("co-representation/emopia_data.npz", allow_pickle=True)
#with open("co-representation/dictionary.pkl", "rb") as f:
#    dictionary = pickle.load(f)

with open("co-representation/dictionary.pkl", "rb") as f:
    token_to_idx, idx_to_token = pickle.load(f)

# Inspect keys and content
#emopia_keys = list(emopia_data.keys())
#print(emopia_keys)
'''
input2 = emopia_data['x'][2]
target2 = emopia_data['y'][2]

print('input2\n')
for row in input2:
  print(row)

print("########")
print('target2\n')
for row2 in target2:
  print(row2)
'''
#print("dictionary: "+str(dictionary))

for key in token_to_idx.keys():
  print(f"{key}: {token_to_idx[key]}")
print("\n##########\n")
vocab_size=1
for key in idx_to_token.keys():
  print(f"{key}: {idx_to_token[key]}")
  print(f"length: {len(idx_to_token[key])}")
  vocab_size = vocab_size * len(idx_to_token[key])
print(f"vocal_size: {vocab_size}")
