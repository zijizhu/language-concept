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
from models.original import construct_OursNet
import torch.nn.functional as F
import train_and_test as tnt

coefs = {
    'crs_ent': 1,
    'orth': 1e-4,
    'clst': 0.8,
    'sep': -0.08,
}


def load_data(dataset_name: str, data_dir: str, batch_size: int):
    assert dataset_name in ['CUB', 'SUN']
    if dataset_name == 'SUN':
        train_dataset = SUNDataset(data_dir, split='train')
        val_dataset = SUNDataset(data_dir, split='val')
        test_dataset = SUNDataset(data_dir, split='test')
        num_classes = 717
    else:
        transforms = Compose([
            Resize((224, 224,)),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
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
        {'params': model.add_on_layers.parameters(), 'lr': 3e-3, 'weight_decay': 1e-3},
        {'params': [model.prototype_vectors], 'lr': 3e-3},
        {'params': model.activation_weight, 'lr': 1e-06}
    ])

    for params in model.features.parameters():
        params.requires_grad = False

    return optimizer


def get_full_optimizer(model: nn.Module):
    optimizer = optim.Adam([
        {'params': model.features.parameters(), 'lr': 1e-4, 'weight_decay': 1e-3},
        {'params': model.add_on_layers.parameters(), 'lr': 3e-3, 'weight_decay': 1e-3},
        {'params': [model.prototype_vectors], 'lr': 3e-3},
        {'params': model.activation_weight, 'lr': 1e-06}
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

    parser.add_argument('--clst-coef', type=float, default=0.8)
    parser.add_argument('--sep-coef', type=float, default=-0.08)
    parser.add_argument('--ortho-coef', type=float, default=1e-4)

    parser.add_argument('--joint-start-epoch', type=int, default=3)
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

    model = construct_OursNet("densenet161", add_on_layers_type="regular")

    optimizer = get_warmup_optimizer(model)
    lr_scheduler = None

    model.to(device=device)

    logger.info("Start warmup...")
    for epoch in range(args.epochs):
        if epoch == args.joint_start_epoch:
            logger.info("Start fine-tuning...")
            optimizer = get_full_optimizer(model)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        _, train_losses = tnt.train(model=model, epoch=epoch, dataloader=train_loader, optimizer=optimizer, tb_writer=None, iteration=None, coefs=coefs)

        for loss_name, loss_value in train_losses.items():
            logger.info(f"Train {loss_name}: {loss_value:.4f}")

        _, val_losses = tnt.test(model=model, epoch=epoch, dataloader=test_loader, tb_writer=None, iteration=None, coefs=coefs)

        for loss_name, loss_value in val_losses.items():
            logger.info(f"Val {loss_name}: {loss_value:.4f}")

        torch.save(
            dict(
                state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                hparams=vars(args),
            ),
            f"logs/{args.name}/model.pth"
        )
        logger.info("Model saved as model.pth")

        if lr_scheduler is not None:
            lr_scheduler.step()


if __name__ == "__main__":
    main()