#!/usr/bin/env python3
"""
This is some documentation
"""
import keras
import numpy as np


class Yolo:
    def __init__(self, model, anchors, class_names):
        self.model = model
        self.anchors = anchors
        self.class_names = class_names
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        boxes = []
        box_confidences = []
        box_class_probs = []
        
        image_height, image_width = image_size
        
        for output in outputs:
            grid_height, grid_width, anchor_boxes, _ = output.shape
            tx = output[..., 0]
            ty = output[..., 1]
            tw = output[..., 2]
            th = output[..., 3]
            box_confidence = self.sigmoid(output[..., 4, np.newaxis])
            class_probs = self.sigmoid(output[..., 5:])
            
            cx = (self.sigmoid(tx) + np.arange(grid_width).reshape(1, grid_width, 1)) / grid_width
            cy = (self.sigmoid(ty) + np.arange(grid_height).reshape(grid_height, 1, 1)) / grid_height
            
            pw = self.anchors[:, 0].reshape(1, 1, anchor_boxes)
            ph = self.anchors[:, 1].reshape(1, 1, anchor_boxes)
            
            bw = pw * np.exp(tw) / self.model.input.shape[1].value
            bh = ph * np.exp(th) / self.model.input.shape[2].value
            
            x1 = (cx - bw / 2) * image_width
            y1 = (cy - bh / 2) * image_height
            x2 = (cx + bw / 2) * image_width
            y2 = (cy + bh / 2) * image_height
            
            boxes.append(np.stack((x1, y1, x2, y2), axis=-1))
            box_confidences.append(box_confidence)
            box_class_probs.append(class_probs)
        
        return (boxes, box_confidences, box_class_probs)
