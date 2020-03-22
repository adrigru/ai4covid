from torchvision.models import densenet121
from PIL import Image
import torchvision.transforms as transforms
from torch.nn import DataParallel
import cv2
import torch
import matplotlib.pyplot as plt

class_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
               'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
               'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']

# TODO adjust to work in our case

class HeatmapGenerator():

    # ---- Initialize heatmap generator
    # ---- pathModel - path to the trained densenet model
    # ---- nnArchitecture - architecture name DENSE-NET121, DENSE-NET169, DENSE-NET201
    # ---- nnClassCount - class count, 14 for chxray-14

    def __init__(self, pathModel, nnClassCount, transCrop):

        # ---- Initialize the network
        model = densenet121(nnClassCount).cuda()

        if use_gpu:
            model = DataParallel(model).cuda()
        else:
            model = DataParallel(model)

        modelCheckpoint = torch.load(pathModel)
        model.load_state_dict(modelCheckpoint['state_dict'])

        self.model = model
        self.model.eval()

        # ---- Initialize the weights
        self.weights = list(self.model.module.densenet121.features.parameters())[-2]

        # ---- Initialize the image transform
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []
        transformList.append(transforms.Resize((transCrop, transCrop)))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)
        self.transformSequence = transforms.Compose(transformList)

    # --------------------------------------------------------------------------------

    def generate(self, pathImageFile, pathOutputFile, transCrop):

        # ---- Load image, transform, convert
        with torch.no_grad():

            imageData = Image.open(pathImageFile).convert('RGB')
            imageData = self.transformSequence(imageData)
            imageData = imageData.unsqueeze_(0)
            if use_gpu:
                imageData = imageData.cuda()
            l = self.model(imageData)
            output = self.model.module.densenet121.features(imageData)
            label = class_names[torch.max(l, 1)[1]]
            # ---- Generate heatmap
            heatmap = None
            for i in range(0, len(self.weights)):
                map = output[0, i, :, :]
                if i == 0:
                    heatmap = self.weights[i] * map
                else:
                    heatmap += self.weights[i] * map
                npHeatmap = heatmap.cpu().data.numpy()

        # ---- Blend original and heatmap

        imgOriginal = cv2.imread(pathImageFile, 1)
        imgOriginal = cv2.resize(imgOriginal, (transCrop, transCrop))

        cam = npHeatmap / np.max(npHeatmap)
        cam = cv2.resize(cam, (transCrop, transCrop))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

        img = cv2.addWeighted(imgOriginal, 1, heatmap, 0.35, 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.title(label)
        plt.imshow(img)
        plt.plot()
        plt.axis('off')
        plt.savefig(pathOutputFile)
        plt.show()