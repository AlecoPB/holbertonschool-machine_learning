#!/usr/bin/env python3
"""
This is some documentation
"""
import os
from glob import iglob
import numpy as np
import cv2
from tensorflow import keras as K


class Yolo:
    """
    Implements the YOLOv3 object detection algorithm.
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initializes the Yolo class.

        Args:
            model_path (str): Path to the YOLO model.
            classes_path (str): Path to the file with class names.
            class_t (float): Confidence threshold for filtering boxes.
            nms_t (float): IoU threshold for non-max suppression.
            anchors (numpy.ndarray): Anchor box dimensions.
        """
        self.model = K.models.load_model(model_path)
        self.class_names = self._load_classes(classes_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    @staticmethod
    def _load_classes(path):
        """
        Reads class names from a file.

        Args:
            path (str): File path to class names.

        Returns:
            list: List of class names.
        """
        with open(path, 'r') as file:
            return [line.strip() for line in file if line.strip()]

    @staticmethod
    def sigmoid(x):
        """
        Applies the sigmoid function.

        Args:
            x (numpy.ndarray): Input array.

        Returns:
            numpy.ndarray: Sigmoid-transformed array.
        """
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Processes model outputs into bounding boxes, confidences, and class probabilities.

        Args:
            outputs (list): List of model output arrays.
            image_size (tuple): Original image size (height, width).

        Returns:
            tuple: Processed boxes, confidences, and class probabilities.
        """
        image_height, image_width = image_size
        boxes, confidences, class_probs = [], [], []

        for i, output in enumerate(outputs):
            grid_height, grid_width = output.shape[:2]

            # Extract box parameters
            t_x, t_y, t_w, t_h = output[..., 0], output[..., 1], output[..., 2], output[..., 3]

            # Generate grid cell coordinates
            c_x, c_y = np.meshgrid(np.arange(grid_width), np.arange(grid_height))
            c_x, c_y = c_x[..., np.newaxis], c_y[..., np.newaxis]

            # Decode box coordinates
            bx = (self.sigmoid(t_x) + c_x) / grid_width
            by = (self.sigmoid(t_y) + c_y) / grid_height
            bw = (np.exp(t_w) * self.anchors[i, :, 0]) / self.model.input.shape[1]
            bh = (np.exp(t_h) * self.anchors[i, :, 1]) / self.model.input.shape[2]

            # Convert to image scale
            x1 = (bx - bw / 2) * image_width
            y1 = (by - bh / 2) * image_height
            x2 = (bx + bw / 2) * image_width
            y2 = (by + bh / 2) * image_height

            boxes.append(np.stack([x1, y1, x2, y2], axis=-1))
            confidences.append(self.sigmoid(output[..., 4:5]))
            class_probs.append(self.sigmoid(output[..., 5:]))

        return boxes, confidences, class_probs

    def filter_boxes(self, boxes, confidences, class_probs):
        """
        Filters boxes based on confidence and class thresholds.

        Args:
            boxes (list): List of bounding boxes.
            confidences (list): List of box confidences.
            class_probs (list): List of class probabilities.

        Returns:
            tuple: Filtered boxes, class IDs, and scores.
        """
        filtered_boxes, box_classes, box_scores = [], [], []

        for box, confidence, class_prob in zip(boxes, confidences, class_probs):
            scores = confidence * class_prob
            class_ids = np.argmax(scores, axis=-1)
            max_scores = np.max(scores, axis=-1)

            mask = max_scores >= self.class_t

            filtered_boxes.append(box[mask])
            box_classes.append(class_ids[mask])
            box_scores.append(max_scores[mask])

        return (
            np.concatenate(filtered_boxes, axis=0),
            np.concatenate(box_classes, axis=0),
            np.concatenate(box_scores, axis=0),
        )

    @staticmethod
    def iou(box1, boxes):
        """
        Computes IoU between a box and multiple boxes.

        Args:
            box1 (numpy.ndarray): Single bounding box.
            boxes (numpy.ndarray): Array of bounding boxes.

        Returns:
            numpy.ndarray: IoU values.
        """
        x1, y1, x2, y2 = box1
        box1_area = (x2 - x1) * (y2 - y1)

        x1s, y1s, x2s, y2s = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        box_areas = (x2s - x1s) * (y2s - y1s)

        inter_x1 = np.maximum(x1, x1s)
        inter_y1 = np.maximum(y1, y1s)
        inter_x2 = np.minimum(x2, x2s)
        inter_y2 = np.minimum(y2, y2s)

        inter_area = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)
        union_area = box1_area + box_areas - inter_area

        return inter_area / union_area

    def non_max_suppression(self, boxes, classes, scores):
        """
        Applies Non-Max Suppression (NMS).

        Args:
            boxes (numpy.ndarray): Bounding boxes.
            classes (numpy.ndarray): Class IDs for each box.
            scores (numpy.ndarray): Confidence scores for each box.

        Returns:
            tuple: Filtered boxes, class IDs, and scores.
        """
        unique_classes = np.unique(classes)
        final_boxes, final_classes, final_scores = [], [], []

        for cls in unique_classes:
            cls_mask = classes == cls
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]

            sorted_indices = np.argsort(-cls_scores)
            cls_boxes = cls_boxes[sorted_indices]
            cls_scores = cls_scores[sorted_indices]

            while len(cls_boxes) > 0:
                box = cls_boxes[0]
                score = cls_scores[0]

                final_boxes.append(box)
                final_classes.append(cls)
                final_scores.append(score)

                if len(cls_boxes) == 1:
                    break

                ious = self.iou(box, cls_boxes[1:])
                cls_boxes = cls_boxes[1:][ious < self.nms_t]
                cls_scores = cls_scores[1:][ious < self.nms_t]

        return (
            np.array(final_boxes),
            np.array(final_classes),
            np.array(final_scores),
        )

    @staticmethod
    def load_images(folder_path):
        """
        Loads images from a folder.

        Args:
            folder_path (str): Path to folder containing images.

        Returns:
            tuple: Loaded images and their paths.
        """
        images, image_paths = [], []

        for path in iglob(os.path.join(folder_path, '*.jpg')):
            image = cv2.imread(path)
            if image is not None:
                images.append(image)
                image_paths.append(path)

        return images, image_paths

    def preprocess_images(self, images):
        """
        Preprocesses images for YOLO model.

        Args:
            images (list): List of images.

        Returns:
            tuple: Preprocessed images and original shapes.
        """
        input_h, input_w = self.model.input.shape[1:3]
        pimages, image_shapes = [], []

        for img in images:
            resized = cv2.resize(img, (input_w, input_h), interpolation=cv2.INTER_CUBIC)
            pimages.append(resized / 255.0)
            image_shapes.append(img.shape[:2])

        return np.array(pimages), np.array(image_shapes)
