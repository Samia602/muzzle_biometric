import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import (
    resnet18,
    ResNet18_Weights
)


class EmbeddingNet(nn.Module):

    def __init__(self, embedding_dim=512):

        super().__init__()

        backbone = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        input_features = backbone.fc.in_features

        backbone.fc = nn.Identity()

        self.backbone = backbone

        self.embedding = nn.Linear(
            input_features,
            embedding_dim
        )

    def forward(self, x):

        x = self.backbone(x)

        x = self.embedding(x)

        x = F.normalize(
            x,
            p=2,
            dim=1
        )

        return x


class ArcFaceHead(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_classes,
        scale=30.0,
        margin=0.5
    ):

        super().__init__()

        self.weight = nn.Parameter(
            torch.FloatTensor(
                num_classes,
                embedding_dim
            )
        )

        nn.init.xavier_uniform_(
            self.weight
        )

        self.scale = scale
        self.margin = margin

    def forward(
        self,
        embeddings,
        labels
    ):

        embeddings = F.normalize(
            embeddings,
            p=2,
            dim=1
        )

        weights = F.normalize(
            self.weight,
            p=2,
            dim=1
        )

        cosine = F.linear(
            embeddings,
            weights
        )

        theta = torch.acos(
            torch.clamp(
                cosine,
                -1.0 + 1e-7,
                1.0 - 1e-7
            )
        )

        target_cosine = torch.cos(
            theta + self.margin
        )

        one_hot = torch.zeros_like(
            cosine
        )

        one_hot.scatter_(
            1,
            labels.view(-1, 1),
            1.0
        )

        output = (
            cosine * (1.0 - one_hot)
            + target_cosine * one_hot
        )

        output *= self.scale

        return output