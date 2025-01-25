#!/usr/bin/env python3
"""
This module implements the Yolo class for the YOLO algorithm.
"""

import os
from glob import iglob
from tensorflow import keras as K
import numpy as np
import cv2


class Yolo:
    """
    This class implements the YOLO v3 algorithm for object detection.
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initializes the Yolo class.

        Args:
            model_path (str): Path to the YOLO model.
            classes_path (str): Path to the file containing class names.
            class_t (float): Threshold for box score during the initial filtering.
            nms_t (float): IOU threshold for non-max suppression.
            anchors (numpy.ndarray): The anchor boxes.
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as file:
            self.class_names = [line.strip() for line in file]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """Computes the sigmoid function."""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Processes the outputs from the YOLO model to extract bounding boxes,
        box confidences, and class probabilities for detected objects.

        Parameters:
        - outputs: A list of numpy.ndarrays containing predictions from YOLO.
        - image_size: A numpy.ndarray containing the original image size
            [image_height, image_width].

        Returns:
        - boxes: A list of numpy.ndarrays with processed bounding boxes for each output.
        - box_confidences: A list of numpy.ndarrays with box confidences for each output.
        - box_class_probs: A list of numpy.ndarrays with class probabilities for each output.
        """
        image_height, image_width = image_size
        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            grid_height, grid_width = output.shape[:2]

            # Extract box parameters (output coordinates)
            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            # Create grid cell coordinates for width and height
            c_x, c_y = np.meshgrid(np.arange(grid_width), np.arange(grid_height))

            # Expand dimensions to match t_x & t_y
            c_x = np.expand_dims(c_x, axis=-1)
            c_y = np.expand_dims(c_y, axis=-1)

            # Calculate bounding box coordinates
            # Apply sigmoid activation and offset by grid cell location, then normalize
            bx = (self.sigmoid(t_x) + c_x) / grid_width
            by = (self.sigmoid(t_y) + c_y) / grid_height
            # Apply exponential and scale by anchor dimensions
            bw = (np.exp(t_w) * self.anchors[i, :, 0]) / self.model.input.shape[1]
            bh = (np.exp(t_h) * self.anchors[i, :, 1]) / self.model.input.shape[2]

            # Convert to original image scale
            x1 = (bx - bw / 2) * image_width
            y1 = (by - bh / 2) * image_height
            x2 = (bx + bw / 2) * image_width
            y2 = (by + bh / 2) * image_height

            # Stack coordinates to form final box coordinates
            boxes.append(np.stack([x1, y1, x2, y2], axis=-1))

            # Extract sigmoid-normalized box confidence and class probabilities
            box_confidences.append(self.sigmoid(output[..., 4:5]))
            box_class_probs.append(self.sigmoid(output[..., 5:]))

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters boxes based on box confidences and class probabilities.

        Parameters:
        - boxes: List of numpy.ndarrays with processed bounding boxes for each output.
        - box_confidences: List of numpy.ndarrays with box confidences for each output.
        - box_class_probs: List of numpy.ndarrays with class probabilities for each output.

        Returns:
        - filtered_boxes: A numpy.ndarray with filtered bounding boxes.
        - box_classes: A numpy.ndarray with the predicted class number for each box in filtered_boxes.
        - box_scores: A numpy.ndarray with the box scores for each box in filtered_boxes.
        """
        filtered _boxes = []
        box_classes = []
        box_scores = []

        for box, box_confidence, box_class_prob in zip(boxes, box_confidences, box_class_probs):
            # Calculate box scores from confidences and class probabilities
            box_score = box_confidence * box_class_prob

            # Identify the class (index) with the highest score for each box
            box_class = np.argmax(box_score, axis=-1)

            # Retain only the highest score for each box
            box_score = np.max(box_score, axis=-1)

            # Create a mask for boxes with scores above the threshold
            filter_mask = box_score >= self.class_t

            # Filter each list using the mask
            filtered_boxes.append(box[filter_mask])
            box_classes.append(box_class[filter_mask])
            box_scores.append(box_score[filter_mask])

        # Convert the resulting lists into numpy arrays
        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def iou(self, box1, boxes):
        """
        Computes the Intersection Over Union (IoU) between a box and an array of boxes.

        Parameters:
        - box1: A numpy.ndarray of shape (4,) representing the first box.
        - boxes: A numpy.ndarray of shape (?, 4) representing the other boxes.

        Returns:
        - iou_scores: A numpy.ndarray of shape (?) containing the IoU scores.
        """
        x1, y1, x2, y2 = box1
        box1_area = (x2 - x1) * (y2 - y1)

        # Extract dimensions for all other boxes to compare
        x1s = boxes[:, 0]
        y1s = boxes[:, 1]
        x2s = boxes[:, 2]
        y2s = boxes[:, 3]

        boxes_area = (x2s - x1s) * (y2s - y1s)

        inter_x1 = np.maximum(x1, x1s)
        inter_y1 = np.maximum(y1, y1s)
        inter_x2 = np.minimum(x2, x2s)
        inter_y2 = np.minimum(y2, y2s)

        inter_area = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)
        union_area = box1_area + boxes_area - inter_area

        iou_scores = inter_area / union_area
        return iou_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Applies Non-Max Suppression (NMS) to filter the bounding boxes.

        Parameters:
        - filtered_boxes: A numpy.ndarray of shape (?, 4) containing all filtered bounding boxes.
        - box_classes: A numpy.ndarray of shape (?,) containing the class number for each box in filtered_boxes.
        - box_scores: A numpy.ndarray of shape (?) containing the box scores for each box in filtered_boxes.

        Returns:
        - box_predictions: A numpy.ndarray of shape (?, 4) containing all predicted bounding boxes ordered by class and box score.
        - predicted_box_classes: A numpy.ndarray of shape (?,) containing the class number for box_predictions ordered by class and box score.
        - predicted_box_scores: A numpy.ndarray of shape (?) containing the box scores for box_predictions ordered by class and box score.
        """
        unique_classes = np.unique(box_classes)
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for cls in unique_classes:
            # Sort the boxes by their unique class
            cls_indices = np.where(box_classes == cls)
            cls_boxes = filtered_boxes[cls_indices]
            cls_scores = box_scores[cls_indices]

            # Sort the boxes by their scores (in descending order)
            sorted_indices = np.argsort(cls_scores)[::-1]
            cls_boxes = cls_boxes[sorted_indices]
            cls_scores = cls_scores[sorted_indices]

            while len(cls_boxes) > 0:
                # Select the box with the highest score
                box = cls_boxes[0]
                score = cls_scores[0]

                box_predictions.append(box)
                predicted_box_classes.append(cls)
                predicted_box_scores.append(score)

                # If this was the last box, no need to continue
                if len(cls_boxes) == 1:
                    break

                # Calculate IoU between the selected box and the rest
                ious = self.iou(box, cls_boxes[1:])
                # Select boxes with IoU lower than the threshold
                remaining_indices = np.where(ious < self.nms_t)[0]

                # Exclude the box we just added to the output
                cls_boxes = cls _boxes[1:][remaining_indices]
                cls_scores = cls_scores[1:][remaining_indices]

        box_predictions = np.array(box_predictions)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores

    @staticmethod
    def load_images(folder_path):
        """
        Loads images from a specified folder.

        Parameters:
        - folder_path: A string representing the path to the folder containing all images to load.

        Returns:
        - images: A list of images as numpy.ndarrays.
        - image_paths: A list of paths to the individual images in images.
        """
        image_paths = []
        images = []
        # Iterate over .jpg image files
        for path in iglob(os.path.join(folder_path, '*.jpg')):
            image = cv2.imread(path)
            if image is not None:
                images.append(image)
                image_paths.append(path)

        return images, image_paths

    def preprocess_images(self, images):
        """
        Resizes and rescales images for the Darknet model.

        Parameters:
        - images: A list of images as numpy.ndarrays.

        Returns:
        - pimages: A numpy.ndarray of shape (ni, input_h, input_w, 3) containing all preprocessed images.
        - image_shapes: A numpy.ndarray of shape (ni, 2) containing the original height and width of the images.
        """
        pimages = []
        image_shapes = []
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        for img in images:
            # Resize image using cubic interpolation
            resized_img = cv2.resize(img, (input_h, input_w), interpolation=cv2.INTER_CUBIC)

            # Rescale pixel values from [0, 255] to [0, 1]
            pimages.append(resized_img / 255.0)

            # Store original image dimensions
            orig_h, orig_w = img.shape[:2]
            image_shapes.append([orig_h, orig_w])

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)
        return pimages, image_shapes