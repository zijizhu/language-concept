import torch
import clip
from torch import nn
import torch.nn.functional as F


class ScoreAggregation(nn.Module):
    def __init__(self, init_val: float = 0.2, num_classes: int = 200, k: int = 5) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.full((num_classes, k,), init_val, dtype=torch.float32))
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor):
        n_classes, n_prototypes = self.weights.shape
        batch_size = x.size(0)
        sa_weights = F.softmax(self.weights, dim=-1) * n_prototypes

        x = x.reshape(batch_size, self.num_classes, -1)
        x = x * sa_weights  # B C K
        x = x.sum(-1)  # B C
        return x


class CLIPConcept(nn.Module):
    def __init__(
        self,
        # query_features: torch.Tensor | None = None,
        # query_words: list[str] = [],
        num_classes: int = 200,
        k: int = 3,
        dim: int = 64,
        # device: str | torch.device = 'cuda',
        clip_model: str = 'ViT-B/16',
        score_aggregation: bool = False
    ):
        super().__init__()
        self.clip, _ = clip.load(clip_model, jit=False)

        # if query_features is not None:
        #     self.register_buffer('query_features', query_features)
        # else:
        #     query_features = []
        #     with torch.no_grad():
        #         for qw in query_words:
        #             query = clip.tokenize([temp(qw) for temp in openai_imagenet_template]).to(device)
        #             feature = self.clip.encode_text(query)
        #             feature /= feature.norm(dim=-1, keepdim=True)
        #             feature = feature.mean(dim=0)
        #             feature /= feature.norm()
        #             query_features.append(feature.unsqueeze(0))
        #     self.register_buffer('query_features', torch.cat(query_features, dim=0))
        self.k = k
        self.num_classes = num_classes

        self.dim = dim

        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels=self.clip.visual.output_dim, out_channels=dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.prototypes = nn.Parameter(torch.randn(num_classes * k, dim, 1, 1))

        nn.init.trunc_normal_(self.prototypes, std=0.02)
        
        if score_aggregation:
            self.classifier = ScoreAggregation(num_classes=num_classes, k=k)
        else:
            self.classifier = nn.Linear(num_classes * k, num_classes, bias=False)
            self._init_classifier()

    def forward(self, images: torch.Tensor, return_attr_logits=False):
        features = self.clip.encode_image(images, return_all=True, csa=True).to(dtype=torch.float32)
        # print("features have nan:", torch.isnan(features).any())
        features = features[:, 1:]  # shape: [batch_size, n_patches, dim]

        dim, patch_size = features.size(-1), self.clip.visual.patch_size
        w = h = images.size(-1) // patch_size
        features = features.permute(0, 2, 1).reshape(-1, dim, w, h)
        features = self.adapter(features)
        # print("adapted features have nan:", torch.isnan(features).any())

        cosine_sims = cosine_conv2d(features, self.prototypes)
        activations = project2basis(features, self.prototypes)

        max_cosine_sims = F.adaptive_max_pool2d(cosine_sims, (1, 1,)).squeeze()
        logits = F.adaptive_max_pool2d(activations, (1, 1,)).squeeze()
        logits = self.classifier(logits)

        return logits, max_cosine_sims, cosine_sims, activations

    def _init_classifier(self):
        self.prototype_class_identity = torch.zeros(self.num_classes * self.k, self.num_classes)

        for j in range(self.num_classes * self.k):
            self.prototype_class_identity[j, j // self.k] = 1

        positive_loc = torch.t(self.prototype_class_identity)
        negative_loc = 1 - positive_loc

        positive_value = 1
        negative_value = -0.5
        self.classifier.weight.data.copy_(positive_value * positive_loc + negative_value * negative_loc)


def cosine_conv2d(x: torch.Tensor, weight: torch.Tensor):
    x = F.normalize(x, p=2, dim=1)
    weight = F.normalize(weight, p=2, dim=1)
    return F.conv2d(input=x, weight=weight)


def project2basis(x: torch.Tensor, weight: torch.Tensor):
    weight = F.normalize(weight, p=2, dim=1)
    return F.conv2d(input=x, weight=weight)


class Criterion(nn.Module):
    def __init__(self, clst_coef: float, sep_coef: float, num_classes: int = 200):
        super().__init__()
        self.num_classes = num_classes
        self.xe = nn.CrossEntropyLoss()

        self.clst_coef = clst_coef
        self.sep_coef = sep_coef

    def forward(self, logits: torch.Tensor, cosine_logits: torch.Tensor, targets: torch.Tensor):
        loss_dict = dict(
            xe=self.xe(logits, targets),
            # clst=self.clst_coef * self.clst_criterion(cosine_logits, targets),
            # sep=self.sep_coef * self.sep_criterion(cosine_logits, targets)
        )
        return sum(loss_dict.values()), loss_dict


    def clst_criterion(self, cosine_logits: torch.Tensor, targets: torch.Tensor):
        batch_size = cosine_logits.size(0)
        cosine_logits = -cosine_logits.reshape(batch_size, self.num_classes, -1)  # shape: [batch_size, num_classes, k]
        return torch.mean(cosine_logits[torch.arange(batch_size), targets].min(dim=-1).values)

    def sep_criterion(self, cosine_logits: torch.Tensor, targets: torch.Tensor):
        batch_size = cosine_logits.size(0)
        cosine_logits = cosine_logits.reshape(batch_size, self.num_classes, -1)  # shape: [batch_size, num_classes, k]
        positives = F.one_hot(targets, num_classes=self.num_classes)
        negative_indices = (1 - positives).nonzero(as_tuple=True)
        return cosine_logits[negative_indices].reshape(batch_size, -1).min(dim=-1).values.mean()
