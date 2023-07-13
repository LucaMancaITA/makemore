
import json
import random
from utils import build_dataset


# Hyperparameters
with open("config/config.json", "r", encoding="utf-8") as config:
    config = json.load(config)
block_size = config["block_size"]

# Read all the words
words = open("data/names.txt", "r").read().splitlines()
print(f"Number of words: {len(words)}")

# Build the vocabolary of characters and mapping to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

# Set the random seed
random.seed(42)
random.shuffle(words)

# Generate train, validation and test set
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))
Xtr, Ytr = build_dataset(words[:n1], block_size, stoi)
Xdev, Ydev = build_dataset(words[n1:n2], block_size, stoi)
Xte, Yte = build_dataset(words[n2:], block_size, stoi)
print("\nDataset info:")
print(f"Training samples:     {Xtr.shape[0]}")
print(f"Validation samples:   {Xdev.shape[0]}")
print(f"Testing samples:      {Xte.shape[0]}")