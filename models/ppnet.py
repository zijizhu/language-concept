import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import densenet161, DenseNet161_Weights


class ScoreAggregation(nn.Module):
    def __init__(self, init_val: float = 0.2, num_classes: int = 200, k: int = 10) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.full((num_classes, k,), init_val))
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor):
        n_classes, n_prototypes = self.weights.shape
        batch_size = x.size(0)
        sa_weights = F.softmax(self.weights, dim=-1) * n_prototypes

        x = x.reshape(batch_size, self.num_classes, -1)
        x = x * sa_weights
        x = x.sum(-1)
        return x


class PPNet(nn.Module):
    def __init__(
            self,
            num_classes: int = 200,
            k: int = 10,
            dim: int = 64,
            score_aggregation: bool = True
    ):
        super().__init__()
        backbone = densenet161(weights=DenseNet161_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.k = k
        self.num_classes = num_classes
        self.dim = dim

        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels=backbone.classifier.in_features, out_channels=dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.prototypes = nn.Parameter(torch.randn(num_classes * k, dim, 1, 1))

        nn.init.trunc_normal_(self.prototypes, std=0.02)

        if score_aggregation:
            self.classifier = ScoreAggregation(num_classes=num_classes, k=k)
        else:
            self.classifier = nn.Linear(num_classes * k, num_classes, bias=False)

        for layer in self.adapter.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')

                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, images: torch.Tensor):
        features = self.backbone(images)  # shape: [batch_size, dim, w, h]
        features = self.adapter(features)

        cosine_activations = cosine_conv2d(features, self.prototypes)
        activations = project2basis(features, self.prototypes)

        cosine_scores = F.adaptive_max_pool2d(cosine_activations, (1, 1,)).squeeze()  # shape: [batch_size, num_concept * k]
        prototype_logits = F.adaptive_max_pool2d(activations, (1, 1,)).squeeze()  # shape: [batch_size, num_concept * k]

        cosine_min_distances = self.global_min_pooling(-cosine_activations)

        logits = self.classifier(prototype_logits)

        # return logits, cosine_scores, cosine_activations, activations
        return logits, cosine_min_distances, cosine_activations, activations


    def normalize_prototypes(self):
        self.prototypes.data = F.normalize(self.prototypes, p=2, dim=1).data

    def global_min_pooling(self, distances):
        min_distances = -F.max_pool2d(-distances,
                                      kernel_size=(distances.size()[2],
                                                   distances.size()[3]))
        min_distances = min_distances.view(-1, self.num_prototypes)

        return min_distances



def cosine_conv2d(x: torch.Tensor, weight: torch.Tensor):
    x = F.normalize(x, p=2, dim=1)
    weight = F.normalize(weight, p=2, dim=1)
    return F.conv2d(input=x, weight=weight)


def project2basis(x: torch.Tensor, weight: torch.Tensor):
    weight = F.normalize(weight, p=2, dim=1)
    return F.conv2d(input=x, weight=weight)


class Criterion(nn.Module):
    def __init__(self, clst_coef: float, sep_coef: float, ortho_coef: float, k: int = 10, num_classes: int = 200):
        super().__init__()
        self.num_classes = num_classes
        self.k = k
        self.xe = nn.CrossEntropyLoss()

        self.clst_coef = clst_coef
        self.sep_coef = sep_coef
        self.ortho_coef = ortho_coef

        self.num_prototypes = 2000
        self.prototype_class_identity = torch.zeros(2000, self.num_classes)
        self.num_prototypes_per_class = self.num_prototypes // self.num_classes
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // self.num_prototypes_per_class] = 1


    def forward(self, logits: torch.Tensor, cosine_scores: torch.Tensor, targets: torch.Tensor, prototypes: torch.Tensor):
        loss_dict = dict(
            xe=self.xe(logits, targets),
            # clst=self.clst_coef * self.clst_criterion(cosine_scores, targets)
            clst=self.clst_coef * self.get_clst_loss(cosine_scores, targets)
        )
        # if self.sep_coef > 0:
        #     loss_dict['sep'] = self.sep_coef * self.sep_criterion(cosine_scores, targets)
        if self.sep_coef != 0:
            loss_dict['sep'] = self.sep_coef * self.get_sep_loss(cosine_scores, targets)
        if self.ortho_coef != 0:
            loss_dict['ortho'] = self.ortho_coef * self.ortho_criterion(prototypes)

        return sum(loss_dict.values()), loss_dict

    def clst_criterion(self, cosine_scores: torch.Tensor, targets: torch.Tensor):
        cosine_scores = cosine_scores.reshape(cosine_scores.size(0), self.num_classes, -1).max(dim=-1).values
        max_dist = 64
        positives = F.one_hot(targets, num_classes=self.num_classes).float()
        inverted_cosine_scores = (max_dist - cosine_scores) * positives
        min_cosine_scores = max_dist - inverted_cosine_scores.max(dim=-1).values
        return min_cosine_scores.mean()

    def sep_criterion(self, cosine_scores: torch.Tensor, targets: torch.Tensor):
        cosine_scores = cosine_scores.reshape(cosine_scores.size(0), self.num_classes, -1).max(dim=-1).values

        max_dist = 64
        negatives = 1 - F.one_hot(targets, num_classes=self.num_classes).float()
        inverted_cosine_scores = (max_dist - cosine_scores) * negatives
        min_cosine_scores = max_dist - inverted_cosine_scores.max(dim=-1).values
        return min_cosine_scores.mean()

    def ortho_criterion(self, prototypes: torch.Tensor):
        cur_basis_matrix = torch.squeeze(prototypes)
        subspace_basis_matrix = cur_basis_matrix.reshape(self.num_classes, 10, -1)
        subspace_basis_matrix_T = torch.transpose(subspace_basis_matrix, 1, 2)
        ortho_operator = torch.matmul(subspace_basis_matrix, subspace_basis_matrix_T)
        I_operator = torch.eye(subspace_basis_matrix.size(1), subspace_basis_matrix.size(1)).to(device=prototypes.device)
        difference_value = ortho_operator - I_operator
        ortho_cost = torch.sum(torch.relu(torch.norm(difference_value, p=1, dim=[1, 2]) - 0))

        return ortho_cost

    def get_clst_loss(self, min_distances, label):
        max_dist = 64
        prototypes_of_correct_class = torch.t(self.prototype_class_identity[:, label]).to(device=min_distances.device)
        inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
        cluster_cost = torch.mean(max_dist - inverted_distances)

        return cluster_cost

    def get_sep_loss(self, min_distances, label):
        max_dist = 64
        prototypes_of_correct_class = torch.t(self.prototype_class_identity[:, label]).to(device=min_distances.device)
        prototypes_of_wrong_class = 1 - prototypes_of_correct_class
        inverted_distances_to_nontarget_prototypes, _ = \
            torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
        separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

        return separation_cost

