import os
import os.path as osp
from typing import Dict, List
import warnings
from argparse import ArgumentParser

import cv2
import mmcv
import numpy as np
import torch
import json

from mmpose.apis import init_pose_model, vis_pose_result
from mmpose.apis.inference import inference_top_down_pose_model_batch
from mmpose.datasets import DatasetInfo


def crawl_image_directory(img_root: str) -> List[str]:
    """Crawls all subdirectories of img_root and returns a list of all image files."""
    image_files = []
    for root, _, files in os.walk(img_root):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    return image_files


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose_config", type=str, required=True)
    parser.add_argument("--pose_checkpoint", type=str, required=True)
    parser.add_argument("--img_root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    
    args = parser.parse_args()
    



    pose_model = init_pose_model(args.pose_config, args.pose_checkpoint, device="cpu")
    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    dataset_info = DatasetInfo(dataset_info)
    return_heatmap = False
    output_layer_names = None
    
    if torch.cuda.is_available():
        pose_model = pose_model.cuda()

    all_images = crawl_image_directory(args.img_root)
    
    
    with open("results.jsonl", "w") as f:
        for i in range(0, len(all_images), args.batch_size):
            batch_images = all_images[i:i+args.batch_size]
        
            pose_results, returned_outputs = inference_top_down_pose_model_batch(
                pose_model,
                batch_images,
                person_results=None,
                bbox_thr=None,
                format='xywh',
                dataset=dataset,
                dataset_info=dataset_info,
                return_heatmap=return_heatmap,
                outputs=output_layer_names
            )
            
            for idx, image in enumerate(batch_images):
                results = {
                    "image_name": image,
                    "pose_results": pose_results[idx]['keypoints'].tolist()
                }
                f.write(json.dumps(results) + "\n")
                