"""
The training loop and the loaders for everything a finished run leaves behind.
"""

import numpy as np
import torch

import config


def train_model(model, train_loader, val_loaders, epochs, learning_rate, device,
                checkpoint_path, history_path, loss_fn, siamese=False, checkpoint_on=0,
                slow_prefixes=(), slow_lr_factor=1.0,
                freeze_prefixes=(), warmup_epochs=0, cosine=False):
    """Generic training loop for both Siamese models.

    Parameters whose names start with slow_prefixes train at
    learning_rate * slow_lr_factor, everything else at the full rate.
    freeze_prefixes stay frozen for the first warmup_epochs.
    cosine decays all rates to zero over the run, otherwise they stay constant.
    val_loaders is a list; the model is checkpointed on val_loaders[checkpoint_on].
    """
    slow = [p for n, p in model.named_parameters() if n.startswith(slow_prefixes)]
    fast = [p for n, p in model.named_parameters() if not n.startswith(slow_prefixes)]

    groups = [{"params": fast, "lr": learning_rate}]
    if slow:
        groups.append({"params": slow, "lr": learning_rate * slow_lr_factor})
    optimizer = torch.optim.Adam(groups)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) if cosine else None
    history = np.zeros((epochs, 3 + len(val_loaders)))
    best_val_loss = float("inf")

    def forward(batch):
        if siamese:
            pre, post, targets = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            return model(pre, post), targets
        inputs, targets = batch[0].to(device), batch[1].to(device)
        return model(inputs), targets

    def evaluate(loader):
        model.eval()
        loss_sum, n = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                logits, targets = forward(batch)
                loss = loss_fn(logits, targets)
                if torch.isfinite(loss):
                    loss_sum += loss.item() * targets.shape[0]
                    n += targets.shape[0]
        return loss_sum / n if n else float("nan")

    for epoch in range(epochs):
        if warmup_epochs:
            frozen = epoch < warmup_epochs
            for name, param in model.named_parameters():
                if name.startswith(freeze_prefixes):
                    param.requires_grad = not frozen

        model.train()
        train_loss_sum, n_train, n_negative = 0.0, 0, 0

        for batch in train_loader:
            optimizer.zero_grad()
            logits, targets = forward(batch)
            loss = loss_fn(logits, targets)
            if not torch.isfinite(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = targets.shape[0]
            train_loss_sum += loss.item() * batch_size
            n_train += batch_size
            n_negative += batch[-1].sum().item()

        if scheduler is not None:
            scheduler.step()

        train_loss = train_loss_sum / n_train
        pct_negative = 100 * n_negative / n_train
        val_losses = [evaluate(loader) for loader in val_loaders]
        history[epoch] = [train_loss, *val_losses, 100 - pct_negative, pct_negative]

        marker = ""
        if val_losses[checkpoint_on] < best_val_loss:
            best_val_loss = val_losses[checkpoint_on]
            torch.save(model.state_dict(), checkpoint_path)
            marker = "  best"

        losses = "  ".join(f"val{i + 1} {v:.4f}" for i, v in enumerate(val_losses))
        print(f"epoch {epoch + 1}/{epochs}  train {train_loss:.4f}  {losses}  "
              f"crops {100 - pct_negative:.1f}% pos{marker}")

    torch.save(history, history_path)
    return history


def load_checkpoint(model, filename, device=None):
    path = config.CKPT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"missing checkpoint {path}, see README.md")
    model.load_state_dict(torch.load(path, map_location=device or "cpu", weights_only=True))
    model.eval()
    print(f"loaded {path}")
    return model


def load_history(filename):
    path = config.HISTORY_DIR / filename
    if not path.exists():
        print(f"no saved history at {path}")
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def load_val_tiles(filename):
    path = config.VAL_TILES_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"missing {path}, run notebook 3 with DRAW_VAL_TILES = True")
    tiles = torch.load(path, map_location="cpu", weights_only=False)
    print(f"loaded {path} ({len(tiles)} tiles)")
    return tiles
