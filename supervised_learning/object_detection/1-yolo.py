#!/usr/bin/env python3
"""
This is some documentation
"""
import keras
import numpy as np


class Yolo:
    def process_outputs(self, outputs, image_size):
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size

        for output in outputs:
            grid_height, grid_width, anchor_boxes = output.shape[:3]

            # Extract t_x, t_y, t_w, t_h, box_confidence, and class_probs
            t_xy = output[..., :2]
            t_wh = output[..., 2:4]
            box_confidence = output[..., 4:5]
            class_probs = output[..., 5:]

            # Calculate box coordinates relative to the original image size
            bx_by = t_xy / np.array([grid_width, grid_height])
            bw_bh = np.exp(t_wh)
            
            # Anchor boxes, assume they are normalized to the grid cell
            # This will typically be derived from anchor box sizes specified during model training
            anchors = np.array([[0.5, 0.5], [1.0, 1.0], [2.0, 2.0]])
            
            bw_bh = bw_bh * anchors
            x1y1 = (bx_by - (bw_bh / 2)) * np.array([image_width, image_height])
            x2y2 = (bx_by + (bw_bh / 2)) * np.array([image_width, image_height])
            box = np.concatenate([x1y1, x2y2], axis=-1)

            boxes.append(box)
            box_confidences.append(1 / (1 + np.exp(-box_confidence)))
            box_class_probs.append(1 / (1 + np.exp(-class_probs)))

        return boxes, box_confidences, box_class_probs
