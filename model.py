import torch
import clip
from torch import nn
import torch.nn.functional as F

from promts import openai_imagenet_template

class CLIPConcept(nn.Module):
    def __init__(self, query_features: torch.Tensor | None = None, query_words: list[str] = [], num_classes: int = 200, device: str | torch.device = 'cuda', clip_model: str = 'ViT-B/16'):
        super().__init__()
        self.clip, _ = clip.load(clip_model, jit=False)

        if query_features is not None:
            self.register_buffer('query_features', query_features)
        else:
            query_features = []
            with torch.no_grad():
                for qw in query_words:
                    query = clip.tokenize([temp(qw) for temp in openai_imagenet_template]).to(device)
                    feature = self.clip.encode_text(query)
                    feature /= feature.norm(dim=-1, keepdim=True)
                    feature = feature.mean(dim=0)
                    feature /= feature.norm()
                    query_features.append(feature.unsqueeze(0))
            self.register_buffer('query_features', torch.cat(query_features, dim=0))
        
        self.linear = nn.Linear(self.query_features.size(0), num_classes)

    def forward(self, images: torch.Tensor, return_attr_logits=False):
        image_features = self.clip.encode_image(images, return_all=True, csa=True).to(dtype=torch.float32)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features = image_features[:, 1:]
        attr_logits = image_features @ self.query_features.T

        path_size = self.clip.visual.patch_size

        w = h = images.size(-1) // path_size
        attr_dim = attr_logits.size(-1)
        attr_logits = attr_logits.permute(0, 2, 1).reshape(-1, attr_dim, w, h)

        attr_logits_pooled = F.adaptive_max_pool2d(attr_logits, (1,1,)).squeeze()

        logits = self.linear(attr_logits_pooled)

        if return_attr_logits:
            return logits, attr_logits

        return logits
