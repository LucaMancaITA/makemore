
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader


class CharDataset(Dataset):

    def __init__(self, words, chars, max_word_length):
        self.words = words
        self.chars = chars
        self.max_word_length = max_word_length
        self.stoi = {ch:i+1 for i, ch in enumerate(chars)}
        self.itos = {i:s for s, i in self.stoi.items()} # inverse mapping

    def __len__(self):
        return len(self.words)

    def contains(self, word):
        return word in self.words

    def get_vocab_size(self):
        return len(self.chars) + 1

    def get_output_length(self):
        return self.max_word_length + 1 # <START> token followed by words

    def encode(self, word):
        ix = torch.tensor([self.stoi[w] for w in word], dtype=torch.long)
        return ix

    def decode(self, ix):
        word = ''.join(self.itos[i] for i in ix)
        return word

    def __getitem__(self, idx):
        word = self.words[idx]
        encoding = self.encode(word)
        x = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        y = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        x[1:1+len(encoding)] = encoding
        y[:len(encoding)] = encoding
        y[len(encoding)+1:] = -1 # mask the loss at the inactive locations
        return x, y


def build_datasets(input_file):
    """Build the training, validation and test set.

    Args:
        input_file (str): path to where the data is stored.

    Returns:
        torch.Dataset: train, val and test dataset.
    """
    # Preprocessing of the input text file
    with open(input_file, 'r') as f:
        data = f.read()
    words = data.splitlines()
    words = [w.strip() for w in words] # get rid of any white space
    words = [w for w in words if w] # get rid of any empty strings
    chars = sorted(list(set(''.join(words)))) # all the possible characters
    max_word_length = max(len(w) for w in words)
    print(f"Number of examples in the dataset: {len(words)}")
    print(f"Max word length: {max_word_length}")
    print(f"Number of unique characters in the vocabulary: {len(chars)}")
    print("Vocabulary:")
    print(''.join(chars))

    # Training, validation and test set
    train_set_size = int(0.8*len(words))
    test_set_size = int(0.9*len(words))
    rp = torch.randperm(len(words)).tolist()
    train_words = [words[i] for i in rp[:train_set_size]]
    val_words = [words[i] for i in rp[train_set_size:test_set_size]]
    test_words = [words[i] for i in rp[test_set_size:]]
    print("Split up the dataset into:\n" \
        f"{len(train_words)} training examples\n" \
        f"{len(val_words)} validation examples\n" \
        f"{len(test_words)} test examples")

    # Wrap in dataset objects
    train_dataset = CharDataset(train_words, chars, max_word_length)
    val_dataset = CharDataset(val_words, chars, max_word_length)
    test_dataset = CharDataset(test_words, chars, max_word_length)

    return train_dataset, val_dataset, test_dataset


@torch.inference_mode()
def evaluate(model, dataset, batch_size=50, max_batches=None):
    """Model evaluation.

    Args:
        model (Object): PyTorch model.
        dataset (Object): PyTorch dataloader.
        batch_size (int, optional): dataset batch size. Defaults to 50.
        max_batches (int, optional): max batches to evaluate. Defaults to None.

    Returns:
        float: evaluation loss.
    """
    model.eval()
    loader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=0)
    losses = []
    for i, batch in enumerate(loader):
        X, Y = batch
        logits, loss = model(X, Y)
        losses.append(loss.item())
        if max_batches is not None and i >= max_batches:
            break
    mean_loss = torch.tensor(losses).mean().item()
    model.train() # reset model back to training mode
    return mean_loss
