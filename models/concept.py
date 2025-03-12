import torch
import clip
from torch import nn
import torch.nn.functional as F
from torchvision.models import densenet121, DenseNet121_Weights


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
        backbone = densenet121(weights=DenseNet121_Weights.DEFAULT)
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

        cosine_sims = cosine_conv2d(features, self.prototypes)
        activations = project2basis(features, self.prototypes)

        max_cosine_sims = F.adaptive_max_pool2d(cosine_sims, (1, 1,)).squeeze()
        concept_logits = F.adaptive_max_pool2d(activations, (1, 1,)).squeeze()
        logits = self.classifier(concept_logits)

        return logits, max_cosine_sims, concept_logits, cosine_sims, activations

    def _init_classifier(self):
        self.prototype_class_identity = torch.zeros(self.num_classes * self.k, self.num_classes)

        for j in range(self.num_classes * self.k):
            self.prototype_class_identity[j, j // self.k] = 1

        positive_loc = torch.t(self.prototype_class_identity)
        negative_loc = 1 - positive_loc

        positive_value = 1
        negative_value = -0.5
        self.classifier.weight.data.copy_(positive_value * positive_loc + negative_value * negative_loc)

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
    def __init__(self, clst_coef: float, sep_coef: float, num_concepts: int = 112):
        super().__init__()
        self.num_concepts = num_concepts
        self.xe = nn.CrossEntropyLoss()

        self.clst_coef = clst_coef
        self.sep_coef = sep_coef

    def forward(self, logits: torch.Tensor, cosine_logits: torch.Tensor, targets: torch.Tensor, concept_targets: torch.Tensor):
        loss_dict = dict(
            xe=self.xe(logits, targets),
            clst=self.clst_coef * self.clst_criterion(cosine_logits, concept_targets),
            sep=self.sep_coef * self.sep_criterion(cosine_logits, concept_targets)
        )
        return sum(loss_dict.values()), loss_dict

    def clst_criterion(self, cosine_logits: torch.Tensor, concept_targets: torch.Tensor):
        batch_size = cosine_logits.size(0)
        cosine_logits = cosine_logits.reshape(batch_size, self.num_concepts, -1)  # shape: [batch_size, num_classes, k]
        positive_indices = concept_targets.float().nonzero(as_tuple=True)
        return cosine_logits[positive_indices].min(dim=-1).values.mean()

    def sep_criterion(self, cosine_logits: torch.Tensor, concept_targets: torch.Tensor):
        batch_size = cosine_logits.size(0)
        cosine_logits = cosine_logits.reshape(batch_size, self.num_concepts, -1)  # shape: [batch_size, num_classes, k]
        positives = concept_targets.float()
        negative_indices = (1 - positives).nonzero(as_tuple=True)
        return cosine_logits[negative_indices].reshape(batch_size, -1).min(dim=-1).values.mean()
