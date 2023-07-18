
# Imports
import json
import torch
from torch.nn import Functional as F
from utils import build_datasets


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    block_size = model.get_block_size()
    for _ in range(max_new_tokens):
        # If the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        # Forward the model to get the logits for the index in the sequence
        logits, _ = model(idx_cond)
        # Pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # oOptionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
        # Apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # Either sample from the distribution or take the most likely element
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)
        # Append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

def print_samples(filename, num=10, device="cpu"):
    """Samples from the model and pretty prints the decoded samples."""
    X_init = torch.zeros(num, 1, dtype=torch.long).to(device)
    top_k = -1
    train_dataset, val_dataset, test_dataset = build_datasets(filename)
    steps = train_dataset.get_output_length() - 1 # -1 because we already start with <START> token (index 0)
    X_samp = generate(model, X_init, steps, top_k=top_k, do_sample=True).to('cpu')
    train_samples, test_samples, new_samples = [], [], []
    for i in range(X_samp.size(0)):
        # Get the i'th row of sampled integers, as python list
        row = X_samp[i, 1:].tolist() # note: we need to crop out the first <START> token
        # Token 0 is the <STOP> token, so we crop the output sequence at that point
        crop_index = row.index(0) if 0 in row else len(row)
        row = row[:crop_index]
        word_samp = train_dataset.decode(row)
        # Separately track samples that we have and have not seen before
        if train_dataset.contains(word_samp) or val_dataset.contains(word_samp):
            train_samples.append(word_samp)
        elif test_dataset.contains(word_samp):
            test_samples.append(word_samp)
        else:
            new_samples.append(word_samp)
    print('-'*80)
    for lst, desc in [(train_samples, 'in train'), (test_samples, 'in test'), (new_samples, 'new')]:
        print(f"{len(lst)} samples that are {desc}:")
        for word in lst:
            print(word)
    print('-'*80)


if __name__ == "__main__":

    # Torch generator
    g = torch.Generator().manual_seed(42)

    # Constants
    with open("config/config.json", "r", encoding="utf-8") as config:
        config = json.load(config)

    filename = "data/names.txt"

    # Load the model
    model = None

    # Sample from the model
    #print_samples(filename, num=10, device="cpu")