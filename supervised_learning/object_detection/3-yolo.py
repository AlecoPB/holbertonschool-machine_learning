#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np
from keras.models import load_model


class Yolo:
    """
    YOLO Object Detection Model
    """

    def __init__(self, model_path, classes_path,
                 class_threshold, nms_threshold, anchors):
        """
        Initialize YOLO with model, classes, and thresholds.
        """
        self.model = load_model(model_path)
        self.class_names = self._read_classes(classes_path)
        self.class_threshold = class_threshold
        self.nms_threshold = nms_threshold
        self.anchors = anchors

    def _read_classes(self, path):
        """
        Read and return class names from a file.
        """
        with open(path, 'r') as file:
            return [line.strip() for line in file if line.strip()]

    def process_outputs(self, outputs, image_size):
        """
        Process raw model outputs into bounding boxes
        """
        image_height, image_width = image_size
        boxes, confidences, class_probs = [], [], []

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            # Generate grid coordinates
            grid_x, grid_y = np.meshgrid(np.arange(grid_width),
                                         np.arange(grid_height))
            grid_x = grid_x.reshape(1, grid_height, grid_width, 1)
            grid_y = grid_y.reshape(1, grid_height, grid_width, 1)

            # Decode bounding box predictions
            center_x = ((1 / (1 + np.exp(-output[..., 0])) + grid_x)
                        / grid_width * image_width)
            center_y = ((1 / (1 + np.exp(-output[..., 1])) + grid_y)
                        / grid_height * image_height)
            width = (self.anchors[i][:, 0] * np.exp(output[..., 2])
                     / self.model.input.shape[1] * image_width)
            height = (self.anchors[i][:, 1] * np.exp(output[..., 3])
                      / self.model.input.shape[2] * image_height)

            x1, y1 = center_x - width / 2, center_y - height / 2
            x2, y2 = center_x + width / 2, center_y + height / 2

            boxes.append(np.stack([x1, y1, x2, y2], axis=-1))
            confidences.append(1 / (1 + np.exp(-output[..., 4:5])))
            class_probs.append(1 / (1 + np.exp(-output[..., 5:])))

        return boxes, confidences, class_probs

    def filter_boxes(self, boxes, confidences, class_probs):
        """
        Filter boxes based on class and confidence thresholds.
        """
        filtered_boxes, box_scores, box_classes = [], [], []

        for i in range(len(confidences)):
            scores = confidences[i] * class_probs[i]
            max_scores = scores.max(axis=-1)
            valid_indices = np.where(max_scores > self.class_threshold)

            filtered_boxes.append(boxes[i][valid_indices])
            box_scores.extend(max_scores[valid_indices])
            box_classes.extend(scores[valid_indices].argmax(axis=-1))

        return (np.concatenate(filtered_boxes), np.array(box_classes),
                np.array(box_scores))

    def non_max_suppression(self, boxes, classes, scores):
        """
        Apply Non-Max Suppression to filter overlapping boxes.
        """
        selected_indices = []

        for class_id in np.unique(classes):
            indices = np.where(classes == class_id)[0]
            sorted_indices = indices[np.argsort(-scores[indices])]

            while len(sorted_indices) > 0:
                current = sorted_indices[0]
                selected_indices.append(current)

                if len(sorted_indices) == 1:
                    break

                ious = np.array([IoU(boxes[current],
                                     boxes[i]) for i in sorted_indices[1:]])
                sorted_indices = sorted_indices[1:][ious < self.nms_threshold]

        selected_indices = np.array(selected_indices)
        return (boxes[selected_indices], classes[selected_indices],
                scores[selected_indices])


def IoU(box1, box2):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    """
    x1_inter, y1_inter = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2_inter, y2_inter = min(box1[2], box2[2]), min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area != 0 else 0
