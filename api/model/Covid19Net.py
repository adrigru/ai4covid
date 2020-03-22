import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.models import densenet121

sns.set()


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


class COVID19DataSet(Dataset):
    def __init__(self, metadata, image_dir, transform=None):
        self.image_names = image_dir + metadata.filename.values
        self.labels = metadata.has_covid19.values
        self.transform = transform

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor([label])

    def __len__(self):
        return len(self.image_names)


def predict(data_loader, model):
    model.eval()
    y_true = torch.FloatTensor()
    y_pred = torch.FloatTensor()
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x
            batch_y_pred = model(batch_x).cpu()
            y_true = torch.cat([y_true, batch_y])
            y_pred = torch.cat([y_pred, batch_y_pred])
    return y_pred


def load_image(image):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # TODO adjust the class for inference throw away metedata and so on...
    test = COVID19DataSet(metadata_val, image, transform=transform)
    return DataLoader(dataset=test, batch_size=len(test), num_workers=1)


def load_model():
    model = COVID19Classifier(densenet121)
    ckpt_dict = torch.load('/model.pth.tar')
    model.load_state_dict(ckpt_dict['state_dict'])
    model.model.classifier = nn.Linear(model.num_features, 1)
    return model
