from data import MVTecDataset, mvtec_classes, DEFAULT_SIZE
from models import PatchCore
from utils import backbones, dataset_scale_factor, tsne, pca

# For inference 
import numpy as np
from pathlib import Path 
from PIL import Image
import torch 
from torchvision import transforms
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns  

ALL_CLASSES = mvtec_classes()


def run_model(
        classes: list = ALL_CLASSES,
        backbone: str = 'WideResNet50'
) -> None:

    f_coreset = 0.1

    # Vanilla or Clip version
    vanilla = backbone == "WideResNet50"

    results = {}    # key = class, Value = [image-level ROC AUC, pixel-level ROC AUC]
    if vanilla:
        size = DEFAULT_SIZE
    elif backbone == 'ResNet50':    # RN50
        size = 224
    elif backbone == 'ResNet50-4':  # RN50x4
        size = 288
    elif backbone == 'ResNet50-16': # RN50x16
        size = 384
    elif backbone == 'ResNet101':   # RN50x101
        size = 224
    else:
        raise Exception('You can use the following nets: ResNet50, ResNet50-4, ResNet50-16, ResNet50-64, ResNet101')

    print(f'Running PatchCore...')
    for cls in classes:
        train_dl, test_dl = MVTecDataset(cls, size=size, vanilla=vanilla).get_dataloaders()
        patch_core = PatchCore(f_coreset, vanilla=vanilla, backbone=backbones[backbone], image_size=size)

        print(f'\nClass {cls}:')
        print(f'Training...')
        patch_core.fit(train_dl, scale=dataset_scale_factor[backbone])

        print(f'Testing...')
        image_rocauc, pixel_rocauc = patch_core.evaluate(test_dl)

        print(f'Results:')
        results[cls] = [float(image_rocauc), float(pixel_rocauc)]
        print(f'- Image-level ROC AUC = {image_rocauc:.3f}')
        print(f'- Iixel-level ROC AUC = {pixel_rocauc:.3f}\n')

    # Save global results and statistics
    image_results = [v[0] for k, v in results.items()]
    average_image_rocauc = sum(image_results) / len(image_results)
    pixel_resuts = [v[1] for k, v in results.items()]
    average_pixel_rocauc = sum(pixel_resuts) / len(pixel_resuts)

    print(f'- Average image-level ROC AUC = {average_image_rocauc:.3f}\n')
    print(f'- Average pixel-level ROC AUC = {average_pixel_rocauc:.3f}\n')

    return patch_core


if __name__ == "__main__":
    classes = "bottle"
    model = run_model(backbone='WideResNet50',
                      classes=[classes])
    
    # Inference image 
    DATA_PATH = f"datasets/{classes}"
    DEFAULT_SIZE = 224
    DEFAULT_RESIZE = 256
    IMAGENET_MEAN = torch.tensor([.485, .456, .406])  
    IMAGENET_STD = torch.tensor([.229, .224, .225]) 

    transform = transforms.Compose([         # Image transform
                transforms.Resize(DEFAULT_RESIZE, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(DEFAULT_SIZE),  
                transforms.ToTensor(),  
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD), 
            ])
    
    oks = list(Path(DATA_PATH).glob("test/good/*.png"))
    ngs = [p for p in Path(DATA_PATH).glob("test/*/*.png") if "good" not in str(p)]
    
    paths = oks + ngs
    labels = []
    for p in paths:
        if "good" in str(p):
            labels.append(0)
        elif "broken_large" in str(p):
            labels.append(1)
        elif "broken_small" in str(p):
            labels.append(2)
        elif "contamination" in str(p):
            labels.append(3)

    features, predictions = [], []
    for p in paths:
        # Image Load 
        img = Image.open(str(p)).convert("RGB")
        tensor_img = transform(img)

        # Extract features
        feature_maps = model.forward(tensor_img.unsqueeze(0))
        avg = torch.nn.AvgPool2d(3, stride=1)
        fmap_size = feature_maps[0].shape[-2]         # Feature map sizes h, w
        resize = torch.nn.AdaptiveAvgPool2d(fmap_size)
        resized_maps = [resize(avg(fmap)) for fmap in feature_maps]
        concated_feature = torch.cat(resized_maps, 1)
        concated_feature = concated_feature.flatten()
        features.append(concated_feature.detach().cpu().numpy())

        # Prediction 
        preds = model.predict(tensor_img.unsqueeze(0))
        predictions.append(preds)
    
    features = np.array(features)
    pcaed_features = pca(features)

    kmeans = KMeans(n_clusters=4)
    full_cls = kmeans.fit_predict(features)
    pcaed_cls = kmeans.fit_predict(pcaed_features)

    tsne_features = tsne(pcaed_features)
    x, y = tsne_features[:, 0], tsne_features[:, 1]
    sns.scatterplot(x=x,
                    y=y,
                    hue=labels,
                    style=full_cls,
                    palette="tab10"
                    )
    plt.legend(labels=["ok", "large", "small", "cont"])
    plt.savefig("full_scatter.jpg")
    plt.clf()

    sns.scatterplot(x=x,
                    y=y,
                    hue=labels,
                    style=pcaed_cls,
                    palette="tab10"
                    )
    plt.legend(labels=["ok", "large", "small", "cont"])
    plt.savefig("pcaed_scatter.jpg")

    print("here")