
# Imports
import os
import sys
import json
import time
import torch
from torch.utils.data import DataLoader
from utils import build_datasets, evaluate
from nets.bigram import Bigram
from torch.utils.tensorboard import SummaryWriter


# Hyperparameters
with open("config/training.json", "r", encoding="utf-8") as config:
    config = json.load(config)
seed = config["seed"]
work_dir = config["work_dir"]
input_file = config["input_file"]
vocab_size = config["vocab_size"]
architecture = config["architecture"]
resume = config["resume"]
batch_size = config["batch_size"]
max_epochs = config["max_epochs"]
learning_rate = config["learning_rate"]
weight_decay = config["weight_decay"]
save_step = config["save_step"]

# System inits
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Generate train, validation and test set
train_dataset, val_dataset, test_dataset = build_datasets(input_file)
os.makedirs(work_dir, exist_ok=True)
writer = SummaryWriter(log_dir=work_dir)

# Train dataloader
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)

# Create the model
if architecture == "bigram":
    model = Bigram(vocab_size=vocab_size)
else:
    print("Please insert a supported model architecture.\n" \
          "Supported models are: bigram.")
    sys.exit()

# Fine-tuning
if resume:
    try:
        model.load_state_dict(torch.load(os.path.join(work_dir, "model.pt")))
        print("Starting from checkpoint.")
    except:
        print("Unavailable model checkpoint.")
        sys.exit()
else:
    print("Training from scratch.")

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
    betas=(0.9, 0.99),
    eps=1e-8
)

# Training loop
best_loss = None
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

    # Model evaluation
    if step > 0 and step % save_step == 0:
        train_loss = evaluate(
            model=model,
            dataset=train_dataset,
            batch_size=100,
            max_batches=10)
        test_loss  = evaluate(
            model=model,
            dataset=test_dataset,
            batch_size=100,
            max_batches=10)
        writer.add_scalar("Loss/train", train_loss, step)
        writer.add_scalar("Loss/test", test_loss, step)
        writer.flush()
        print(f"step {step} train loss: {train_loss} test loss: {test_loss}")

        # Save the model to disk if it has improved
        if best_loss is None or test_loss < best_loss:
            out_path = os.path.join(work_dir, "model.pt")
            print(f"test loss {test_loss} is the best so far, saving model to {out_path}")
            torch.save(model.state_dict(), out_path)
            best_loss = test_loss
