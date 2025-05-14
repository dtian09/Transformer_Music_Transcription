import numpy as np
import pickle
import json

# Load EMOPHIA data and dictionary
emopia_data = np.load("co-representation/emopia_data.npz", allow_pickle=True)
with open("co-representation/dictionary.pkl", "rb") as f:
    dictionary = pickle.load(f)

# Inspect keys and content
emopia_keys = list(emopia_data.keys())
example_input = emopia_data[emopia_keys[0]]
token_dict = dictionary

print("example_input: "+str(example_input))
print(f"token_dict: "+str(token_dict))
