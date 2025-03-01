import torch
import clip
from torch import nn
import torch.nn.functional as F


class CLIPConcept(nn.Module):
    def __init__(
            self,
            # query_features: torch.Tensor | None = None,
            # query_words: list[str] = [],
            num_classes: int = 200,
            k: int = 3,
            # device: str | torch.device = 'cuda',
            clip_model: str = 'ViT-B/16'
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

        self.prototypes = nn.Parameter(torch.randn(num_classes * k, self.clip.visual.output_dim, 1, 1))
        self.fc = nn.Linear(num_classes * k, num_classes, bias=False)

        # self.linear = nn.Linear(self.query_features.size(0), num_classes)

    def forward(self, images: torch.Tensor, return_attr_logits=False):
        image_features = self.clip.encode_image(images, return_all=True, csa=True).to(dtype=torch.float32)
        image_features = image_features[:, 1:]  # shape: [batch_size, n_patches, dim]
        dim, patch_size = image_features.size(-1), self.clip.visual.patch_size
        w = h = images.size(-1) // patch_size
        image_features = image_features.permute(0, 2, 1).reshape(-1, dim, w, h)

        cosine_sims = cosine_conv2d(image_features, self.prototypes)
        activations = project2basis(image_features, self.prototypes)

        max_cosine_sims = F.adaptive_max_pool2d(cosine_sims, (1, 1,)).squeeze()
        logits = F.adaptive_max_pool2d(activations, (1, 1,)).squeeze()
        logits = self.fc(logits)

        return logits, max_cosine_sims, cosine_sims, activations

    def _init_fc(self):
        self.prototype_class_identity = torch.zeros(self.num_classes * self.k, self.num_classes)

        for j in range(self.num_classes * self.k):
            self.prototype_class_identity[j, j // self.k] = 1

        positive_loc = torch.t(self.prototype_class_identity)
        negative_loc = 1 - positive_loc

        positive_value = 1
        negative_value = -0.5
        self.fc.weight.data.copy_(
            positive_value * positive_loc
            + negative_value * negative_loc)


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
        return (self.xe(logits, targets)
                + self.clst_coef * self.clst_criterion(cosine_logits, targets)
                + self.sep_coef * self.sep_criterion(cosine_logits, targets))

    def clst_criterion(self, cosine_logits: torch.Tensor, targets: torch.Tensor):
        batch_size = cosine_logits.size(0)
        cosine_logits = -cosine_logits.reshape(batch_size, self.num_classes, -1)  # shape: [batch_size, num_classes, k]
        return torch.mean(cosine_logits[torch.arange(batch_size), targets].min(dim=-1).values)

    def sep_criterion(self, cosine_logits: torch.Tensor, targets: torch.Tensor):
        batch_size = cosine_logits.size(0)
        cosine_logits = cosine_logits.reshape(batch_size, self.num_classes, -1)  # shape: [batch_size, num_classes, k]
        positives = F.one_hot(targets, num_classes=self.num_classes)
        negative_indices = (1 - positives).nonzero(as_tuple=True)
        return cosine_logits[negative_indices].reshape(batch_size, -1, 3).min(dim=-1).values.min(dim=-1).values.mean()
