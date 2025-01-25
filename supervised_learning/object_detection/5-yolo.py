#!/usr/bin/env python3
"""
This script implements the Yolo class, which provides methods for
object detection using the YOLO v3 algorithm.
"""
import tensorflow as tf
import numpy as np


class Yolo:
    """
    A class for object detection using the YOLO v3 algorithm.
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initializes the Yolo object with a pre-trained model, class labels,
        thresholds, and anchor boxes.

        Args:
            model_path (str): Path to the saved YOLO model.
            classes_path (str): Path to the file containing class labels.
            class_t (float): Threshold for filtering boxes based on confidence.
            nms_t (float): IoU threshold for Non-Max Suppression (NMS).
            anchors (numpy.ndarray): Array of anchor box dimensions.
        """
        self.model = tf.keras.models.load_model(model_path)
        with open(classes_path, 'r') as file:
            self.class_names = [line.strip() for line in file.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """Applies the sigmoid function to the input."""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Processes the outputs of the YOLO model into bounding box coordinates,
        confidence scores, and class probabilities.

        Args:
            outputs (list of numpy.ndarrays): Predictions from YOLO layers.
            image_size (numpy.ndarray): The original image dimensions [height, width].

        Returns:
            tuple: (boxes, box_confidences, box_class_probs) where:
                - boxes is a list of arrays containing bounding box coordinates.
                - box_confidences is a list of arrays containing confidence scores.
                - box_class_probs is a list of arrays containing class probabilities.
        """
        image_height, image_width = image_size
        boxes, confidences, class_probs = [], [], []

        for i, output in enumerate(outputs):
            grid_h, grid_w = output.shape[:2]

            # Extract box parameters
            t_x, t_y, t_w, t_h = output[..., 0], output[..., 1], output[..., 2], output[..., 3]
            c_x, c_y = np.meshgrid(np.arange(grid_w), np.arange(grid_h))

            c_x = np.expand_dims(c_x, axis=-1)
            c_y = np.expand_dims(c_y, axis=-1)

            # Normalize and scale bounding box predictions
            bx = (self.sigmoid(t_x) + c_x) / grid_w
            by = (self.sigmoid(t_y) + c_y) / grid_h
            bw = (np.exp(t_w) * self.anchors[i, :, 0]) / self.model.input.shape[1]
            bh = (np.exp(t_h) * self.anchors[i, :, 1]) / self.model.input.shape[2]

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
        Filters bounding boxes based on confidence and class scores.

        Args:
            boxes (list): Processed bounding box coordinates.
            confidences (list): Confidence scores for each box.
            class_probs (list): Class probabilities for each box.

        Returns:
            tuple: (filtered_boxes, box_classes, box_scores) where:
                - filtered_boxes contains the filtered bounding box coordinates.
                - box_classes contains the predicted class indices.
                - box_scores contains the corresponding scores.
        """
        filtered_boxes, box_classes, box_scores = [], [], []

        for box, confidence, class_prob in zip(boxes, confidences, class_probs):
            scores = confidence * class_prob
            class_indices = np.argmax(scores, axis=-1)
            class_scores = np.max(scores, axis=-1)
            mask = class_scores >= self.class_t

            filtered_boxes.append(box[mask])
            box_classes.append(class_indices[mask])
            box_scores.append(class_scores[mask])

        return (np.concatenate(filtered_boxes, axis=0),
                np.concatenate(box_classes, axis=0),
                np.concatenate(box_scores, axis=0))

    def iou(self, box1, boxes):
        """
        Calculates the Intersection over Union (IoU) between one box and
        an array of boxes.

        Args:
            box1 (numpy.ndarray): A single box [x1, y1, x2, y2].
            boxes (numpy.ndarray): Multiple boxes to compare against.

        Returns:
            numpy.ndarray: IoU values for each comparison.
        """
        x1, y1, x2, y2 = box1
        area1 = (x2 - x1) * (y2 - y1)

        x1s, y1s, x2s, y2s = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2s - x1s) * (y2s - y1s)

        inter_x1 = np.maximum(x1, x1s)
        inter_y1 = np.maximum(y1, y1s)
        inter_x2 = np.minimum(x2, x2s)
        inter_y2 = np.minimum(y2, y2s)
        inter_area = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)

        union_area = area1 + areas - inter_area
        return inter_area / union_area

    def non_max_suppression(self, boxes, classes, scores):
        """
        Applies Non-Max Suppression (NMS) to remove overlapping boxes.

        Args:
            boxes (numpy.ndarray): All bounding boxes.
            classes (numpy.ndarray): Predicted classes for each box.
            scores (numpy.ndarray): Scores for each box.

        Returns:
            tuple: (final_boxes, final_classes, final_scores).
        """
        final_boxes, final_classes, final_scores = [], [], []
        unique_classes = np.unique(classes)

        for cls in unique_classes:
            indices = np.where(classes == cls)
            cls_boxes = boxes[indices]
            cls_scores = scores[indices]

            order = np.argsort(cls_scores)[::-1]
            cls_boxes, cls_scores = cls_boxes[order], cls_scores[order]

            while cls_boxes.size > 0:
                best_box = cls_boxes[0]
                best_score = cls_scores[0]

                final_boxes.append(best_box)
                final_classes.append(cls)
                final_scores.append(best_score)

                if len(cls_boxes) == 1:
                    break

                ious = self.iou(best_box, cls_boxes[1:])
                keep = np.where(ious < self.nms_t)[0]
                cls_boxes = cls_boxes[1:][keep]
                cls_scores = cls_scores[1:][keep]

        return (np.array(final_boxes),
                np.array(final_classes),
                np.array(final_scores))
