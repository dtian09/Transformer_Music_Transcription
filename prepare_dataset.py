import numpy as np
import pickle
import json

# Load EMOPHIA data and dictionary
emopia_data = np.load("co-representation/emopia_data.npz", allow_pickle=True)
with open("co-representation/dictionary.pkl", "rb") as f:
    dictionary = pickle.load(f)

# Inspect keys and content
emopia_keys = list(emopia_data.keys())
print(emopia_keys)
input0 = emopia_data['x'][0]
target0 = emopia_data['y'][0]
token_dict = dictionary

print("input0: "+str(input0))
print("target0: "+str(target0))
print("dictionary: "+str(dictionary))
