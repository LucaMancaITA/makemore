
# Imports
import json
import time
import torch
from torch.utils.data import DataLoader
from utils import build_datasets
from nets.bigram import Bigram


# Hyperparameters
with open("config/config.json", "r", encoding="utf-8") as config:
    config = json.load(config)
seed = config["seed"]
vocab_size = config["vocab_size"]
batch_size = config["batch_size"]
max_epochs = config["max_epochs"]
learning_rate = config["learning_rate"]
weight_decay = config["weight_decay"]

# System inits
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Generate train, validation and test set
filename = "data/names.txt"
train_dataset, val_dataset, test_dataset = build_datasets(filename)

# Train dataloader
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)

# Create the model
model = Bigram(vocab_size=vocab_size)

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
    betas=(0.9, 0.99),
    eps=1e-8
)

# Training loop
print("\nTraining loop ...")
for step in range(max_epochs):

    t0 = time.time()

    # Read next batch
    batch = next(iter(train_dataloader))

    x_batch, y_batch = batch

    # Forward pass
    logits, loss = model(x_batch, y_batch)

    # Backward pass
    model.zero_grad(set_to_none=True)  # as p.grad = None for p in parameters
    loss.backward()
    optimizer.step()

    t1 = time.time()

    # Logging
    if step == 0 or step % 10 == 0:
        print(f"Step {step} | loss {loss.item():.4f}" \
              f"| step time {(t1-t0)*1000:.2f}ms")

    # Model evaluation --> TODO

    if step >= 500:
      break
