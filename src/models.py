import torch
from torch import nn

import numpy as np
from itertools import pairwise
from torchvision import models


class SimpleMLP(torch.nn.Module):
    """3 layer MLP for colored MNIST classification"""

    def __init__(self, input_size=28 * 28 * 3, num_classes=10, hidden_dims=(128, 128)):
        super(SimpleMLP, self).__init__()

        dimensions = [input_size, *hidden_dims]
        feature_extractor = []
        feature_extractor.append(nn.Flatten())
        for d1, d2 in pairwise(dimensions):
            feature_extractor.append(torch.nn.Linear(d1, d2))
            feature_extractor.append(torch.nn.ReLU())
        self.feature_extractor = torch.nn.Sequential(*feature_extractor)
        self.features_dim = hidden_dims[-1]
        self.fc = torch.nn.Linear(self.features_dim, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.fc(features)


def get_model(model_name="mlp", num_classes=2, device="cuda", **kwargs):
    default_kwargs = {"mlp_hidden_dims": (256, 256), "input_shape": (3, 28, 28)}
    default_kwargs.update(kwargs)

    if model_name == "mlp":
        print(f'mlp hidden dims: {default_kwargs["mlp_hidden_dims"]}')
        model = SimpleMLP(
            input_size=np.prod(default_kwargs["input_shape"]),
            hidden_dims=default_kwargs["mlp_hidden_dims"],
            num_classes=num_classes,
        )
        return model.to(device)

    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    else:
        raise ValueError('Wrong model name')
        
    model = model.to(device)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes).to(device)
    return model
