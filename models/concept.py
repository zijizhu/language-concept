import torch
import clip
from torch import nn
import torch.nn.functional as F
from torchvision.models import densenet161, DenseNet161_Weights


class CLIPConcept(nn.Module):
    def __init__(
            self,
            # query_features: torch.Tensor | None = None,
            # query_words: list[str] = [],
            num_classes: int = 200,
            num_concepts: int = 112,
            k: int = 3,
            dim: int = 64,
            # device: str | torch.device = 'cuda',
            clip_model: str = 'ViT-B/16',
            score_aggregation: bool = True
    ):
        super().__init__()
        backbone = densenet161(weights=DenseNet161_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.k = k
        self.num_classes = num_classes
        self.num_concepts = num_concepts
        self.dim = dim

        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels=backbone.classifier.in_features, out_channels=dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.prototypes = nn.Parameter(torch.randn(num_concepts * k, dim, 1, 1))

        nn.init.trunc_normal_(self.prototypes, std=0.02)

        self.classifier = nn.Linear(num_concepts * k, num_classes, bias=False)

    def forward(self, images: torch.Tensor):
        features = self.backbone(images)  # shape: [batch_size, dim, w, h]
        features = self.adapter(features)

        cosine_activations = cosine_conv2d(features, self.prototypes)
        activations = project2basis(features, self.prototypes)

        cosine_scores = F.adaptive_max_pool2d(cosine_activations, (1, 1,)).squeeze()  # shape: [batch_size, num_concept * k]
        prototype_logits = F.adaptive_max_pool2d(activations, (1, 1,)).squeeze()  # shape: [batch_size, num_concept * k]

        concept_logits = prototype_logits.reshape(-1, self.num_concepts, self.k).max(dim=-1).values
        logits = self.classifier(prototype_logits)

        return logits, cosine_scores, concept_logits, cosine_activations, activations

    def normalize_prototypes(self):
        self.prototypes.data = F.normalize(self.prototypes, p=2, dim=1).data


def cosine_conv2d(x: torch.Tensor, weight: torch.Tensor):
    x = F.normalize(x, p=2, dim=1)
    weight = F.normalize(weight, p=2, dim=1)
    return F.conv2d(input=x, weight=weight)


def project2basis(x: torch.Tensor, weight: torch.Tensor):
    weight = F.normalize(weight, p=2, dim=1)
    return F.conv2d(input=x, weight=weight)


class Criterion(nn.Module):
    def __init__(self, bce_coef: float, clst_coef: float, sep_coef: float, num_concepts: int = 112):
        super().__init__()
        self.num_concepts = num_concepts
        self.xe = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()

        self.bce_coef = bce_coef
        self.clst_coef = clst_coef
        self.sep_coef = sep_coef

    def forward(self, logits: torch.Tensor, cosine_scores: torch.Tensor, concept_logits: torch.Tensor, targets: torch.Tensor, concept_targets: torch.Tensor):
        loss_dict = dict(
            xe=self.xe(logits, targets),
            bce=self.bce_coef * self.bce(concept_logits, concept_targets),
            clst=self.clst_coef * self.clst_criterion(cosine_scores, concept_targets),
            sep=self.sep_coef * self.sep_criterion(cosine_scores, concept_targets)
        )
        return sum(loss_dict.values()), loss_dict

    def clst_criterion(self, cosine_scores: torch.Tensor, concept_targets: torch.Tensor):
        cosine_scores = cosine_scores.reshape(cosine_scores.size(0), self.num_concepts, -1).max(dim=-1).values
        max_dist = 64
        positives = concept_targets.float()
        inverted_cosine_scores = (max_dist - cosine_scores) * positives
        min_cosine_scores = max_dist - inverted_cosine_scores.max(dim=-1).values
        return min_cosine_scores.mean()

    def sep_criterion(self, cosine_scores: torch.Tensor, concept_targets: torch.Tensor):
        cosine_scores = cosine_scores.reshape(cosine_scores.size(0), self.num_concepts, -1).max(dim=-1).values

        max_dist = 64
        negatives = 1 - concept_targets.float()
        inverted_cosine_scores = (max_dist - cosine_scores) * negatives
        min_cosine_scores = max_dist - inverted_cosine_scores.max(dim=-1).values
        return min_cosine_scores.mean()

