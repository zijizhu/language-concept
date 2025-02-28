import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm

from data import SUNDataset
from model import CLIPConcept

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        predicted = torch.argmax(outputs, dim=-1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(train_loader), correct / total

def validate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            predicted = torch.argmax(outputs, dim=-1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(test_loader), correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--data-dir', type=str, default='datasets')

    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = SUNDataset(args.data_dir, split='train')
    val_dataset = SUNDataset(args.data_dir, split='val')
    test_dataset = SUNDataset(args.data_dir, split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = CLIPConcept(
        query_features=torch.load('data/SUN/sun_attr_features_multi_prompt_mean.pt'),
        num_classes=717,
        device=device
    )
    criterion = nn.CrossEntropyLoss()
    for params in model.clip.parameters():
        params.requires_grad = False

    optimizer = optim.Adam(model.linear.parameters(), lr=args.lr)
    model.to(device=device)
    criterion.to(device=device)
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, "model.pth")
    print("Model saved as model.pth")

if __name__ == "__main__":
    main()