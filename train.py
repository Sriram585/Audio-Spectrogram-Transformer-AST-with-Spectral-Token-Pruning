import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import download_esc50, ESC50SpectrogramDataset
from model import AudioSpectrogramTransformer

def train_and_evaluate(keep_ratio: float, train_loader, val_loader, epochs: int = 18):
    device = torch.device("cpu")
    model = AudioSpectrogramTransformer(num_classes=10, keep_ratio=keep_ratio).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    mode_name = "Baseline AST (100% Tokens)" if keep_ratio == 1.0 else f"Pruned AST ({int(keep_ratio*100)}% Tokens)"
    print(f"\n--- Training {mode_name} ---")
    start_time = time.perf_counter()

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for specs, labels in train_loader:
            optimizer.zero_grad()
            logits = model(specs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        # Validation pass
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for specs, labels in val_loader:
                preds = model(specs).argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = (correct / total) * 100.0
        best_acc = max(best_acc, val_acc)

        if epoch % 3 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")

    total_time = time.perf_counter() - start_time
    os.makedirs("saved_models", exist_ok=True)
    save_path = f"saved_models/ast_keep_{int(keep_ratio*100)}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Finished {mode_name} in {total_time:.1f}s | Best Val Acc: {best_acc:.2f}% | Saved to {save_path}")
    return best_acc, total_time

if __name__ == "__main__":
    torch.manual_seed(42)
    root = download_esc50("data")

    # Standard 5-fold split: Folds 1-4 (320 clips) for training, Fold 5 (80 clips) for validation
    train_ds = ESC50SpectrogramDataset(root, folds=(1, 2, 3, 4), esc10_only=True)
    val_ds = ESC50SpectrogramDataset(root, folds=(5,), esc10_only=True)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    acc_base, time_base = train_and_evaluate(keep_ratio=1.0, train_loader=train_loader, val_loader=val_loader)
    acc_pruned, time_pruned = train_and_evaluate(keep_ratio=0.6, train_loader=train_loader, val_loader=val_loader)

    print("\n================ SUMMARY ================")
    print(f"Baseline AST (160 tokens) : {acc_base:.2f}% accuracy | Train Time: {time_base:.1f}s")
    print(f"Pruned AST   ( 96 tokens) : {acc_pruned:.2f}% accuracy | Train Time: {time_pruned:.1f}s")