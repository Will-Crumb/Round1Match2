import argparse
import os
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import optuna

# ---------------------------
# Repro + Device
# ---------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # allow fast path
    torch.backends.cudnn.benchmark = True


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------
# Models
# ---------------------------
class MLP(nn.Module):
    def __init__(self, input_dim=28 * 28, hidden_dim=256, dropout=0.2, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class SmallCNN(nn.Module):
    def __init__(self, dropout=0.2, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 28x28 -> 28x28
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28 -> 14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 14x14
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14 -> 7
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------------------------
# Training / Evaluation
# ---------------------------
@dataclass
class TrainConfig:
    model_type: str = "cnn"  # "cnn" or "mlp"
    batch_size: int = 128
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.2
    seed: int = 42
    val_split: float = 0.1
    num_workers: int = 4
    data_dir: str = "./data"
    save_path: str = "./fashion_mnist_optuna.pt"


def accuracy(logits, targets):
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


def get_dataloaders(cfg: TrainConfig):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),  # Precomputed for Fashion-MNIST
    ])

    full_train = datasets.FashionMNIST(cfg.data_dir, train=True, download=True, transform=transform)

    val_len = int(len(full_train) * cfg.val_split)
    train_len = len(full_train) - val_len
    train_ds, val_ds = random_split(full_train, [train_len, val_len], generator=torch.Generator().manual_seed(cfg.seed))

    test_ds = datasets.FashionMNIST(cfg.data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


def build_model(cfg: TrainConfig):
    if cfg.model_type == "mlp":
        return MLP(dropout=cfg.dropout)
    elif cfg.model_type == "cnn":
        return SmallCNN(dropout=cfg.dropout)
    else:
        raise ValueError(f"Unknown model_type: {cfg.model_type}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        running_acc += accuracy(logits.detach(), labels.detach()) * imgs.size(0)

    n = len(loader.dataset)
    return running_loss / n, running_acc / n


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            logits = model(imgs)
            loss = criterion(logits, labels)
            running_loss += loss.item() * imgs.size(0)
            running_acc += accuracy(logits, labels) * imgs.size(0)

    n = len(loader.dataset)
    return running_loss / n, running_acc / n


def train_eval(cfg: TrainConfig):
    set_seed(cfg.seed)
    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    model = build_model(cfg).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:02d}/{cfg.epochs} | Train loss {tr_loss:.4f} acc {tr_acc:.4f} | Val loss {va_loss:.4f} acc {va_acc:.4f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        torch.save({"model_state_dict": best_state, "config": cfg.__dict__}, cfg.save_path)
        print(f"Saved best model to: {cfg.save_path} (val_acc={best_val_acc:.4f})")

    # Final test evaluation with best state
    if best_state is not None:
        model.load_state_dict(best_state)
    te_loss, te_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test loss {te_loss:.4f} acc {te_acc:.4f}")

def objective(trial):
    cfg = TrainConfig(
        model_type="cnn",
        batch_size=trial.suggest_categorical("batch_size", [64, 128, 256]),
        epochs=5,
        lr=trial.suggest_loguniform("lr", 1e-4, 1e-1),
        weight_decay=trial.suggest_loguniform("weight_decay", 1e-6, 1e-2),
        dropout=trial.suggest_uniform("dropout", 0.1, 0.5),
        seed=42
    )
    set_seed(cfg.seed)
    device = get_device()
    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val_acc = 0.0
    for epoch in range(cfg.epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        _, va_acc = evaluate(model, val_loader,criterion, device)
        best_val_acc = max(best_val_acc, va_acc)

    return best_val_acc

def run_optuna(n_trials, save_path="./fashion_mnist_optuna.pt"):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("Best hyperparameters:", study.best_params)
    print("Best validation accurancy:", study.best_value)

    # Retrain final model with best params
    best_params = study.best_params
    cfg = TrainConfig(
        model_type="cnn",
        batch_size=best_params["batch_size"],
        epochs=10,
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
        dropout=best_params["dropout"],
        seed=42,
        save_path=save_path
    )
    train_eval(cfg)

# ---------------------------
# CLI
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Fashion-MNIST baseline (PyTorch)")
    p.add_argument("--model", type=str, default="cnn", choices=["cnn", "mlp"], help="Model type")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--save_path", type=str, default="./fashion_mnist_optuna.pt")
    p.add_argument("--optimize", type=bool, default="true")
    p.add_argument("--n_trials", type=int, default=5, help="Number of Optuna trials")
    return p.parse_args()


def main():
    args = parse_args()

    if args.optimize:
        run_optuna(n_trials=args.n_trials)
    else:
        cfg = TrainConfig(
            model_type=args.model,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            seed=args.seed,
            val_split=args.val_split,
            num_workers=args.num_workers,
            data_dir=args.data_dir,
            save_path=args.save_path,
        )
        train_eval(cfg)

if __name__ == "__main__":
    main()
