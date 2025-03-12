import logging
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import nn
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from cub_dataset import CUBDataset
from models.backbone import DINOv2BackboneExpanded
from models.dino import ProtoDINO, PCA
from utils.config import load_config_and_logging


@torch.no_grad()
def projection(model: nn.Module,
               dataloader: DataLoader,
               num_classes: int,
               num_prototypes: int,
               top_k_percent=30,
               device: torch.device | str = 'cuda'):
    """
    Finds the most similar image to each prototype by iterating through the training set.
    Extracts a dynamic bounding box around the top K% of activations for each prototype.

    Args:
        model: ProtoPNet model with an activation map output of shape B * C * K * H * W.
        dataloader: DataLoader providing the training dataset.
        num_classes (int): Number of classes (C).
        num_prototypes (int): Number of prototypes (K).
        top_k_percent (float): Percentage of top activations to consider for bounding box (e.g., 5 for top 5%).
        device (str): Device to perform computation on, typically 'cuda' or 'cpu'.

    Returns:
        best_images (torch.Tensor): Best-matching images for each prototype, shape (C, K, H, W).
        best_bboxes (list): Bounding boxes around top activations for each prototype, containing tensors of shape (C, K).
    """
    max_similarities = torch.full((num_classes, num_prototypes), -float('inf')).to(device)
    source_images = np.zeros((num_classes, num_prototypes, 224, 224, 3,), dtype=int)
    bboxes = torch.zeros((num_classes, num_prototypes, 4,)).to(device=device, dtype=torch.long)
    center_coords = torch.zeros((num_classes, num_prototypes, 2,)).to(device=device, dtype=torch.long)
    sample_indices = torch.zeros((num_classes, num_prototypes,)).to(device=device, dtype=torch.long)

    model.eval()  # Set model to evaluation mode
    model.to(device)
    with torch.no_grad():
        for images, batch_labels, indices in tqdm(dataloader):
            images, batch_labels = images.to(device), batch_labels.to(device)

            image_paths = [dataloader.dataset.samples[idx][0] for idx in indices]
            raw_images = [Image.open(im_path).convert("RGB").resize((224, 224,)) for im_path in image_paths]
            # Forward pass to get activation maps
            _, batch_activations = model.push_forward(images)  # type: torch.Tensor  # Expected shape: B * C * K * H * W

            batch_activations = F.interpolate(batch_activations, size=(224, 224,), mode='bilinear')

            # Pooling over spatial dimensions H and W
            B, CK, H, W = batch_activations.shape
            C, K = num_classes, CK // num_classes
            batch_activations = batch_activations.reshape(B, C, K, H, W)
            pooled_activations = batch_activations.view(B, C, K, -1).max(dim=-1).values  # Shape: B * C * K

            for i, (activation_maps, max_activations, label, idx) in enumerate(zip(batch_activations, pooled_activations, batch_labels, indices)):  # Loop over batch
                for j in range(num_prototypes):  # Loop over prototypes
                    similarity = max_activations[label, j]  # Similarity score for the label-specific prototype

                    # Update best match if current similarity is higher
                    if similarity > max_similarities[label, j]:
                        max_similarities[label, j] = similarity
                        source_images[label, j] = np.array(raw_images[i])  # Store the entire image
                        sample_indices[label, j] = idx

                        activation_map = activation_maps[label, j]  # Shape: H*W

                        # Save center coordinate of the prototypical patches
                        center = torch.unravel_index(activation_map.argmax(), (H, W,))
                        center_coords[label, j] = torch.stack(center)

                        # Get the activation map for the current image, class, and prototype
                        threshold_value = torch.quantile(activation_map, 1 - top_k_percent / 100)
                        activation_map = activation_map.reshape(H, W)
                        binary_map = (activation_map >= threshold_value).float()

                        # Find bounding box around the top activations
                        non_zero_indices = torch.nonzero(binary_map, as_tuple=False)
                        if non_zero_indices.numel() > 0:
                            y_min, x_min = torch.min(non_zero_indices, dim=0).values
                            y_max, x_max = torch.max(non_zero_indices, dim=0).values
                            bboxes[label][j] = torch.tensor((y_min.item(), x_min.item(), y_max.item(), x_max.item(),))
    np.savez(
        Path("prototypical_parts.npz"),
        max_similarities=max_similarities.detach().cpu().numpy(),
        source_images=source_images,
        center_coords=center_coords.detach().cpu().numpy(),
        bboxes=bboxes.detach().cpu().numpy(),
        sample_indices=sample_indices.detach().cpu().numpy()
    )

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg, log_dir, args = load_config_and_logging(name="eval", return_args=True)

    logger = logging.getLogger(__name__)

    L.seed_everything(cfg.seed)

    normalize = T.Normalize(mean=(0.485, 0.456, 0.406,), std=(0.229, 0.224, 0.225,))
    transforms = T.Compose([
        T.Resize((224, 224,)),
        T.ToTensor(),
        normalize
    ])

    n_classes = 200
    dataset_dir = Path("datasets") / "cub200_cropped"
    annotations_path = Path("datasets") / "CUB_200_2011"

    dataset_test = CUBDataset((dataset_dir / "test_cropped").as_posix(),
                              transforms=transforms)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=8, num_workers=8, shuffle=True)

    if "dinov2" in cfg.model.name:
        backbone = DINOv2BackboneExpanded(
            name=cfg.model.name,
            n_splits=cfg.model.n_splits,
            mode=cfg.model.tuning,
            freeze_norm_layer=cfg.model.get("freeze_norm", True)
        )
        dim = backbone.dim
    else:
        raise NotImplementedError("Backbone must be one of dinov2 or clip.")

    assert cfg.model.fg_extractor in ["PCA"]
    fg_extractor = PCA(bg_class=n_classes, **cfg.model.fg_extractor_args)

    net = ProtoDINO(
        backbone=backbone,
        dim=dim,
        fg_extractor=fg_extractor,
        n_prototypes=cfg.model.n_prototypes,
        n_classes=n_classes,
        gamma=cfg.model.get("gamma", 0.99),
        temperature=cfg.model.temperature,
        cls_head="attribute" if cfg.get("concept_learning", False) else cfg.model.cls_head,
        sa_init=cfg.model.sa_init,
        use_sinkhorn=cfg.model.get("use_sinkhorn", True),
        norm_prototypes=cfg.model.get("norm_prototypes", False),
        use_norm=cfg.model.get("use_norm", "both")
    )
    state_dict = torch.load(log_dir / "proto_best.pth", map_location="cpu")
    net.load_state_dict(state_dict=state_dict, strict=False)

    net.optimizing_prototypes = False
    net.initializing = False
    net.eval()
    net.to(device)

    projection(
        net,
        dataloader_test,
        num_classes=n_classes,
        num_prototypes=cfg.model.n_prototypes,
        top_k_percent=10,
        device=device,
    )
