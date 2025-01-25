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
        with open(classes_path, 'r') as file:
            self.class_names = [line.strip() for line in file]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
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
        Processes model outputs into bounding boxes,
        confidences, and class probabilities.

        Args:
            outputs (list): List of model output arrays.
            image_size (numpy.ndarray): Original image
            size [image_height, image_width].

        Returns:
            tuple: Processed boxes, confidences, and class probabilities.
        """
        image_height, image_width = image_size
        boxes, box_confidences, box_class_probs = [], [], []

        for i, output in enumerate(outputs):
            grid_height, grid_width = output.shape[:2]

            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            c_x, c_y = np.meshgrid(np.arange(grid_width),
                                   np.arange(grid_height))
            c_x = c_x[..., np.newaxis]
            c_y = c_y[..., np.newaxis]

            bx = (self.sigmoid(t_x) + c_x) / grid_width
            by = (self.sigmoid(t_y) + c_y) / grid_height
            bw = ((np.exp(t_w) * self.anchors[i, :, 0])
                  / self.model.input.shape[1])
            bh = ((np.exp(t_h) * self.anchors[i, :, 1])
                  / self.model.input.shape[2])

            x1 = (bx - bw / 2) * image_width
            y1 = (by - bh / 2) * image_height
            x2 = (bx + bw / 2) * image_width
            y2 = (by + bh / 2) * image_height

            boxes.append(np.stack([x1, y1, x2, y2], axis=-1))
            box_confidences.append(self.sigmoid(output[..., 4:5]))
            box_class_probs.append(self.sigmoid(output[..., 5:]))

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters boxes based on confidence and class thresholds.

        Args:
            boxes (list): List of bounding boxes.
            box_confidences (list): List of box confidences.
            box_class_probs (list): List of class probabilities.

        Returns:
            tuple: Filtered boxes, class IDs, and scores.
        """
        filtered_boxes, box_classes, box_scores = [], [], []

        for box, box_confidence, box_class_prob in zip(boxes,
                                                       box_confidences,
                                                       box_class_probs):
            box_score = box_confidence * box_class_prob
            box_class = np.argmax(box_score, axis=-1)
            box_score = np.max(box_score, axis=-1)

            mask = box_score >= self.class_t
            filtered_boxes.append(box[mask])
            box_classes.append(box_class[mask])
            box_scores.append(box_score[mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def iou(self, box1, boxes):
        """
        Calculates the Intersection Over Union (IoU)
        between a box and an array of boxes.

        Args:
            box1 (numpy.ndarray): Single bounding box.
            boxes (numpy.ndarray): Array of bounding boxes.

        Returns:
            numpy.ndarray: IoU values.
        """
        x1, y1, x2, y2 = box1
        box1_area = (x2 - x1) * (y2 - y1)

        x1s, y1s, x2s, y2s = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        boxes_area = (x2s - x1s) * (y2s - y1s)

        inter_x1 = np.maximum(x1, x1s)
        inter_y1 = np.maximum(y1, y1s)
        inter_x2 = np.minimum(x2, x2s)
        inter_y2 = np.minimum(y2, y2s)

        inter_area = (np.maximum(inter_x2 - inter_x1, 0)
                      * np.maximum(inter_y2 - inter_y1, 0))
        union_area = box1_area + boxes_area - inter_area

        return inter_area / union_area

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Applies Non-Max Suppression (NMS).

        Args:
            filtered_boxes (numpy.ndarray): Bounding boxes.
            box_classes (numpy.ndarray): Class IDs for each box.
            box_scores (numpy.ndarray): Confidence scores for each box.

        Returns:
            tuple: Filtered boxes, class IDs, and scores.
        """
        unique_classes = np.unique(box_classes)
        box_predictions, predicted_box_classes = [], []
        predicted_box_scores = []

        for cls in unique_classes:
            cls_mask = box_classes == cls
            cls_boxes = filtered_boxes[cls_mask]
            cls_scores = box_scores[cls_mask]

            sorted_indices = np.argsort(-cls_scores)
            cls_boxes = cls_boxes[sorted_indices]
            cls_scores = cls_scores[sorted_indices]

            while len(cls_boxes) > 0:
                box_predictions.append(cls_boxes[0])
                predicted_box_classes.append(cls)
                predicted_box_scores.append(cls_scores[0])

                if len(cls_boxes) == 1:
                    break

                ious = self.iou(cls_boxes[0], cls_boxes[1:])
                cls_boxes = cls_boxes[1:][ious < self.nms_t]
                cls_scores = cls_scores[1:][ious < self.nms_t]

        return (
            np.array(box_predictions),
            np.array(predicted_box_classes),
            np.array(predicted_box_scores),
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
        pimages = []
        image_shapes = []
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        for img in images:
            # Resize image with inter-cubic interpolation
            resized_img = cv2.resize(
                img, (input_h, input_w), interpolation=cv2.INTER_CUBIC)

            # Rescale pixel values from [0, 255] to [0, 1]
            pimages.append(resized_img / 255.0)

            # Add image shape to shapes array
            orig_h, orig_w = img.shape[:2]
            image_shapes.append([orig_h, orig_w])

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)
        return pimages, image_shapes
