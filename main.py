from data import MVTecDataset, mvtec_classes, DEFAULT_SIZE
from models import PatchCore
from utils import backbones, dataset_scale_factor, tsne, pca, compute_mask, extract_features

# For inference 
from pathlib import Path 
import numpy as np
from PIL import Image
import torch 
import cv2
from torch.nn import functional as F 
from torchvision import transforms


ALL_CLASSES = mvtec_classes()

def main(classes: list = ALL_CLASSES,
         backbone: str = 'WideResNet50',
         checkpoint_path: str = "",
         mode: str = "train"
) -> None:

    assert mode in ["train", "test", "predict"]
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

        if mode == "predict":
            patch_core.load(checkpoint_path=checkpoint_path)
            return patch_core
        
        if mode == "test": 
            patch_core.load(checkpoint_path=checkpoint_path)

        if mode == "train":
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

    if mode=="train":
        patch_core.save(save_path=checkpoint_path)

    return patch_core

def predict(model,
            threshold: int
            ):
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

    features = []
    for p in paths:
        # Image Load 
        img = Image.open(str(p)).convert("RGB")
        resized_img = img.resize((DEFAULT_SIZE, DEFAULT_SIZE))
        tensor_img = transform(img)

        # Extract features
        feature = extract_features(model=model,
                                  image=tensor_img,
                                  output_size=DEFAULT_SIZE)
        # Prediction 
        preds = model.predict(tensor_img.unsqueeze(0))
        anomaly_score, anomaly_map = preds
        anomaly_score = anomaly_score.item()

        # Make binary mask 
        pred_mask = compute_mask(anomaly_map=anomaly_map.numpy(),
                                 threshold=threshold) 
        pred_mask = pred_mask if anomaly_score > 0.9 else np.zeros_like(pred_mask)

        # Display anomaly mask on origin image 
        color_mask = np.zeros(pred_mask.shape + (3,), 
                              dtype=np.uint8)
        color_mask[:,:,-1] = (pred_mask * 255)
        displayed_img = cv2.addWeighted(color_mask, 0.4, np.array(resized_img), 0.6, 0)

        # Feature extract 
        feature = feature * pred_mask
        features.append(feature)
        
        cv2.imwrite(f"results/{p.parent.name}_{p.stem}_{anomaly_score:.3f}.jpg", displayed_img)
    
    return features, labels 

if __name__ == "__main__":
    classes = "bottle"
    weight_dir = Path("checkpoints")
    checkpoint_path = weight_dir / "weight.pt"
    mode = "predict"

    if mode == "train":
        model = main(backbone='WideResNet50',
                    classes=[classes],
                    save_path=str(checkpoint_path),
                    mode=mode
                    )
    elif mode == "test":
        model = main(backbone='WideResNet50',
                    classes=[classes],
                    checkpoint_path=str(checkpoint_path),
                    mode=mode
                    )
    elif mode == "predict":
        model = main(backbone="WideResNet50",
                     classes=[classes],
                     checkpoint_path=str(checkpoint_path),
                     mode=mode
                     )
        features, labels = predict(model,
                                   threshold=180)
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        feature_array = np.array([f.flatten().numpy() for f in features])
        X_train, X_test, y_train, y_test = train_test_split(feature_array, labels, test_size=0.1)
        clf = RandomForestClassifier(max_depth=5, random_state=5)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

    print("here")