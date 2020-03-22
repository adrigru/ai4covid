import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import densenet121


_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class COVID19Classifier(nn.Module):
    def __init__(self, model_fn):
        super().__init__()
        self.model = model_fn(pretrained=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(self.num_features, 42)

    def forward(self, x):
        x = self.model.features(x)
        x = F.relu(x, inplace=True)
        x = self.pool(x).view(x.size(0), -1)
        x = self.model.classifier(x)
        x = torch.sigmoid(x)
        return x


def predict(model, image):
    model.eval()
    image = _transform(image)
    image = torch.unsqueeze(image, dim=0)
    with torch.no_grad():
        prediction = model(image).item()
    return prediction


def load_model(ckpt_path, device=None):
    model = COVID19Classifier(densenet121)
    model.model.classifier = nn.Linear(model.num_features, 1)
    ckpt_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt_dict['state_dict'])
    return model


def generate_heatmap(model, image):
    model.eval()
    image = _transform(image)
    image = torch.unsqueeze(image, dim=0)
    weights = list(model.model.features.parameters())[-2]
    with torch.no_grad():
        features = model.model.features(image)
        heatmap = torch.zeros(features.shape[-2:])
        for i, w in enumerate(weights):
            heatmap += w * features[0, i, :, :]
    return heatmap
