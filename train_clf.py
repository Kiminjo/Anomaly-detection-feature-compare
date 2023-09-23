from data import  TrainDataset, ValidDataset, DEFAULT_SIZE
from models import load_model
from utils import compute_mask, extract_features
from utils import SplittedMaskInfo, backbones, dataset_scale_factor

import torch
from torch.utils.data import DataLoader
from torch.nn.functional import interpolate
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import accuracy_score, f1_score
from typing import List, Tuple
import random 
import time

random.seed(21)

def patchcore_predict(model,
                      loader: DataLoader,
                      anomaly_score_threshold: float = 0.9,
                      mask_threshold: int = 180
                      ) -> Tuple[List[torch.tensor], List[np.array], List[float]]:
    features, prediction_masks, anomaly_scores = [], [], []
    print("PatchCore Inference...")
    for img, _, _ in tqdm(loader):
        # Extract features
        feature = extract_features(model=model,
                                   image=img,
                                   output_size=DEFAULT_SIZE)
        # Prediction 
        preds = model.predict(img)
        anomaly_score, anomaly_map = preds
        anomaly_score = anomaly_score.item()

        # Make binary mask 
        pred_mask = compute_mask(anomaly_map=anomaly_map.numpy(),
                                 threshold=mask_threshold) 
        pred_mask = pred_mask if anomaly_score > anomaly_score_threshold else np.zeros_like(pred_mask)

        features.append(feature)    
        prediction_masks.append(pred_mask)
        anomaly_scores.append(anomaly_score)
    return features, prediction_masks, anomaly_scores

def patchcore_inference(loader: DataLoader,
                        backbone: str = "WideResNet50",
                        checkpoint: str = ""
                        ):
    # 1. Load patchcore model 
    patchcore_model = load_model(backbone=backbone,
                                 checkpoint_path=checkpoint)
    
    # 2. Get feature, anomaly score and prediction mask 
    features, prediction_masks, anomaly_scores = patchcore_predict(model=patchcore_model,
                                                                           loader=loader)
    
    return features, prediction_masks, anomaly_scores

def split_mask(mask: np.array,
               area_threshold: int = 50
               ): 
    object_infos = [] 
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= area_threshold:
            object_mask = np.uint8(labels == label)
            x, y, w, h, _ = stats[label]
            bounding_box = (x, y, x + w, y + h)

            object_infos.append({"mask": object_mask, "bbox": bounding_box})
    return object_infos

def get_objects_feature(feature: List[torch.tensor],
                       mask_infos: List[List[np.array]]
                       ): 
    output_features = []
    for mask_info in mask_infos: 
        x1, y1, x2, y2 = mask_info["bbox"]
        box_feature = feature * mask_info["mask"]
        cropped_feature = box_feature[:,:,y1:y2,x1:x2]
        object_feature = interpolate(cropped_feature,
                                     size=(DEFAULT_SIZE, DEFAULT_SIZE),
                                     mode="bilinear")
        object_feature = torch.mean(object_feature, dim=(2, 3))
        output_features.append(object_feature.flatten().numpy())
    return output_features


if __name__=='__main__':
    data_dir = Path("datasets/coreimg/bottom")
    defects = ["정상", "밑면01", "밑면02", "밑면03"]
    
    # Componet 0. Get Train and Valid Loader 
    train_data_paths = [str(p) for p in (data_dir / "정상").glob("**/*.bmp")] + [str(p) for p in (data_dir / "정상").glob("**/*.png")]
    valid_data_paths, valid_labels = [], []
    for defect_idx, defect in enumerate(defects): 
        img_paths = [str(p) for p in (data_dir / defect).glob("**/*.bmp")] + [str(p) for p in (data_dir / defect).glob("**/*.png")]
        labels = [defect_idx] * len(img_paths)

        valid_data_paths += img_paths
        valid_labels += labels 
    patchcore_labels = [1 if l !=0 else 0 for l in valid_labels]

    trainset = TrainDataset(data_path=train_data_paths)
    validset = ValidDataset(data_path=valid_data_paths,
                            labels=patchcore_labels)
    
    train_loader = DataLoader(trainset,
                              num_workers=8,
                              shuffle=True)
    valid_loader = DataLoader(validset,
                              num_workers=8,
                              shuffle=False)
    
    # Component 1. Load Model and Inference 
    patchcore_checkpoint = "checkpoints/coreimg_bottom.pt"
    features, prediction_masks, anomaly_scores = patchcore_inference(loader=valid_loader,
                                                                     checkpoint=patchcore_checkpoint)
    
    # Component 2. Split Mask per objects
    # Component 3. Get Feature of object 
    rf_checkpoint = "checkpoints/coreimg_bottom.pkl"
    X, y = [], []
    print("Feature Process...")
    for feature, pred_mask, anomaly_score, cls_label in tqdm(zip(features, prediction_masks, anomaly_scores, valid_labels), total=len(features)): 
        pred_mask_infos: List[SplittedMaskInfo] = split_mask(pred_mask)
        object_features: List[torch.tensor] = get_objects_feature(feature=feature, 
                                                                  mask_infos=pred_mask_infos)
        if len(object_features) > 0:
            X += object_features
            y += [cls_label] * len(object_features)
    X = np.vstack(X)
    y = np.array(y)

    # Component 4. Train Random Forest Classifier 
    print("Random Forest Training...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # clf = RandomForestClassifier(n_estimators=500)
    clf = RandomForestClassifier(n_estimators=300)
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    print(f"Random forest training time: {end - start}")
    rf_result = clf.predict(X_test)
    acc = accuracy_score(y_true=y_test,
                         y_pred = rf_result)
    f1 = f1_score(y_true=y_test,
                  y_pred=rf_result,
                  average="macro")
    print(f"accuracy: {acc:.3f}")
    print(f"f1 score: {f1:.3f}")
    print(y_test)
    print(rf_result)

    joblib.dump(clf, rf_checkpoint)