#!/usr/bin/env python3
"""
This is some documentation
"""
import keras
import numpy as np


class Yolo:
    """
    Clase utilizando Yolo v3 para ejecutar detección de objetos
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        self.model = keras.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        boxes, box_confidences, box_class_probs = [], [], []

        for i in range(3):
            t_x, t_y, t_w, t_h = outputs[i][:4]  # Bounding box coordinates
            box_confidence = outputs[i][..., 4:5]     # Box confidence
            class_probs = outputs[i][..., 5:]         # Class probabilities

            # Convert bounding box coordinates to absolute values
            x1 = (t_x - t_w / 2) * image_size[1]
            y1 = (t_y - t_h / 2) * image_size[0]
            x2 = (t_x + t_w / 2) * image_size[1]
            y2 = (t_y + t_h / 2) * image_size[0]

            # Append processed components to lists
            boxes.append(np.stack([x1, y1, x2, y2], axis=-1))
            box_confidences.append(box_confidence)
            box_class_probs.append(class_probs)

        return boxes, box_confidences, box_class_probs
