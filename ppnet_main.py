import argparse
import sys
from pathlib import Path
from collections import defaultdict
import logging

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from tqdm import tqdm

from data import SUNDataset, CUBConceptDataset
from models.ppnet import PPNet, Criterion


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    train_losses = defaultdict(float)
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        logits, cosine_scores, cosine_activations, activations = model(images)
        loss, loss_dict = criterion(logits, cosine_scores, labels, model.prototypes)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        model.normalize_prototypes()

        for loss_name, loss_value in loss_dict.items():
            train_losses[loss_name] += loss_dict[loss_name].item()

        predicted = torch.argmax(logits, dim=-1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    for loss_name, loss_value in train_losses.items():
        train_losses[loss_name] = loss_value / len(train_loader)
    return train_losses, correct / total


def validate(model, test_loader, criterion, device):
    model.eval()
    val_losses = defaultdict(float)
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            logits, cosine_scores, cosine_activations, activations = model(images)
            loss, loss_dict = criterion(logits, cosine_scores, labels, model.prototypes)

            for loss_name, loss_value in loss_dict.items():
                val_losses[loss_name] += loss_dict[loss_name].item()

            predicted = torch.argmax(logits, dim=-1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    for loss_name, loss_value in val_losses.items():
        val_losses[loss_name] = loss_value / len(test_loader)
    return val_losses, correct / total


def load_data(dataset_name: str, data_dir: str, batch_size: int):
    assert dataset_name in ['CUB', 'SUN']
    if dataset_name == 'SUN':
        train_dataset = SUNDataset(data_dir, split='train')
        val_dataset = SUNDataset(data_dir, split='val')
        test_dataset = SUNDataset(data_dir, split='test')
        num_classes = 717
    else:
        transforms = Compose([
            Resize(224, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        train_dataset = CUBConceptDataset(
            Path(data_dir) / 'cub200_cropped' / 'train_cropped_augmented',
            transforms=transforms
        )
        test_dataset = CUBConceptDataset(
            Path(data_dir) / 'cub200_cropped' / 'test_cropped',
            transforms=transforms
        )
        num_classes = 200
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, num_classes


def get_warmup_optimizer(model: nn.Module):
    optimizer = optim.Adam([
        {'params': list(model.adapter.parameters()) + [model.prototypes], 'lr': 3e-3},
        {'params': model.classifier.parameters(), 'lr': 1e-06}
    ])

    for params in model.backbone.parameters():
        params.requires_grad = False

    return optimizer


def get_full_optimizer(model: nn.Module):
    optimizer = optim.Adam([
        {'params': model.backbone.parameters(), 'lr': 1e-4},
        {'params': list(model.adapter.parameters()) + [model.prototypes], 'lr': 3e-3},
        {'params': model.classifier.parameters(), 'lr': 1e-06}
    ])

    for params in model.parameters():
        params.requires_grad = True

    return optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=8, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=80, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--data-dir', type=str, default='datasets')
    parser.add_argument('--dataset', type=str, default='CUB', choices=['CUB', 'SUN'])

    parser.add_argument('--clst-coef', type=float, default=-0.8)
    parser.add_argument('--sep-coef', type=float, default=0.08)
    parser.add_argument('--orth-coef', type=float, default=1e-4)

    parser.add_argument('--name', type=str, required=True)

    args = parser.parse_args()

    log_dir = Path('logs') / args.name
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler((log_dir / "train.log").as_posix()),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    
    logger = logging.getLogger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, num_classes = load_data(args.dataset, args.data_dir, args.batch_size)

    model = PPNet(num_classes=num_classes)

    criterion = Criterion(
        clst_coef=args.clst_coef,
        sep_coef=args.sep_coef,
        orth_coef=args.orth_coef,
        num_classes=num_classes
    )

    optimizer = get_warmup_optimizer(model)
    lr_scheduler = None

    model.to(device=device)
    criterion.to(device=device)

    for epoch in range(args.epochs):
        if epoch == 3:
            optimizer = get_full_optimizer(model)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        train_losses, train_acc = train(model, train_loader, criterion, optimizer, device)

        val_losses, val_acc = validate(model, test_loader, criterion, device)

        for loss_name, loss_value in train_losses.items():
            logger.info(f"Train {loss_name}: {loss_value:.4f}")
        logger.info(f"Train Acc: {train_acc:.4f}")

        for loss_name, loss_value in val_losses.items():
            logger.info(f"Val {loss_name}: {loss_value:.4f}")
        logger.info(f"Val Acc: {val_acc:.4f}")

        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, f"logs/{args.name}/model.pth")
        logger.info("Model saved as model.pth")

        if lr_scheduler is not None:
            lr_scheduler.step()


if __name__ == "__main__":
    main()