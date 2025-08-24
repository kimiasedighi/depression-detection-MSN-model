import os
import torch
import argparse
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from dataset import PoseDataset
from msn_body import MSNBody


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)
        data = data.permute(0, 1, 2, 3)  # [B, C, T, J]

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            data = data.permute(0, 1, 2, 3)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    return acc, all_preds, all_labels


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = PoseDataset(args.data_dir)
    indices = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=42)

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False)

    model = MSNBody(in_channels=3, num_joints=11, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, _, _ = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {train_loss:.4f} - Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            print("✅ Saved best model!")

    # Final evaluation on test set
    model.load_state_dict(torch.load(args.save_path))
    test_acc, preds, labels = evaluate(model, test_loader, device)
    print(f"\n🎯 Final Test Accuracy: {test_acc:.2f}%")
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=["Healthy", "Depressed"]))
    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MSN model on pose data")
    parser.add_argument('--data_dir', type=str, default="./processed_data", help="Path to processed .pt data")
    parser.add_argument('--save_path', type=str, default="best_msn_model.pth", help="Path to save best model")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=20)

    args = parser.parse_args()
    main(args)
