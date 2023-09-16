
from data import  InferenceDataset, DEFAULT_SIZE
from models import load_model
from utils import compute_mask, extract_features
from utils import SplittedMaskInfo

import torch
from torch.utils.data import DataLoader
from torch.nn.functional import interpolate
from pathlib import Path
import numpy as np
import cv2 
from typing import List, Tuple

def patchcore_predict(model,
                      loader: DataLoader,
                      anomaly_score_threshold: float = 0.9,
                      mask_threshold: int = 180
                      ) -> Tuple[List[torch.tensor], List[np.array], List[float]]:
    features, prediction_masks, anomaly_scores = [], [], []
    for img in loader:
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

        # # Display anomaly mask on origin image 
        # color_mask = np.zeros(pred_mask.shape + (3,), 
        #                       dtype=np.uint8)
        # color_mask[:,:,-1] = (pred_mask * 255)
        # displayed_img = cv2.addWeighted(color_mask, 0.4, np.array(resized_img), 0.6, 0)

        features.append(feature)    
        prediction_masks.append(pred_mask)
        anomaly_scores.append(anomaly_score)
    return features, prediction_masks, anomaly_scores

def patchcore_inference(backbone: str = "WideResNet50",
                        checkpoint: str = ""
                        ):
    # 1. Load image 
    data_dir = Path("datasets/bottle/test/broken_large")
    data_paths = [str(p) for p in data_dir.glob("*.png")]

    inference_dataset = InferenceDataset(data_path=data_paths)
    inference_dataloader = DataLoader(inference_dataset,
                                      batch_size=1,
                                      num_workers=8,
                                      shuffle=False)
    
    # 2. Load patchcore model 
    patchcore_model = load_model(backbone=backbone,
                                 checkpoint_path=checkpoint)
    
    # 3. Get feature, anomaly score and prediction mask 
    features, prediction_masks, anomaly_scores= patchcore_predict(model=patchcore_model,
                                                                  loader=inference_dataloader)
    
    return features, prediction_masks, anomaly_scores

def split_mask(mask: np.array,
               area_threshold: int = 50
               ): 
    object_infos = [] 
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

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
        output_features.append(object_feature.flatten())
    return output_features



if __name__=='__main__':
    checkpoint = "checkpoints/weight.pt"
    
    # Component 1. Inference using PathCore 
    print("patchcore inference...") 
    features, prediction_masks, anomaly_scores = patchcore_inference(checkpoint=checkpoint)

    # Component 2. Split mask per objects 
    # Component 3. Get Feature of object
    for feature, pred_mask, anomaly_score in zip(features, prediction_masks, anomaly_scores): 
        pred_mask_infos: List[SplittedMaskInfo] = split_mask(pred_mask)
        object_features: List[torch.tensor] = get_objects_feature(feature=feature, 
                                                                 mask_infos=pred_mask_infos)
        print("here")


        


    