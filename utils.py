
import torch

def build_dataset(words, block_size, char2int):
    """Generate a torch dataset.

    Args:
        words (list): list of all the available words.
        block_size (int): number of characters used as input to predict the
                          next one.
        char2int (dict): dict with char mapping to int.

    Returns:
        torch.tensor: tensor dataset of inputs and outputs.
    """
    X, Y = [], []
    for w in words:
      context = [0] * block_size
      for ch in w + '.':
        ix = char2int[ch]
        X.append(context)
        Y.append(ix)
        context = context[1:] + [ix]

    X = torch.tensor(X)
    Y = torch.tensor(Y)

    return X, Y