#!/usr/bin/env python3
"""
This module provides the Yolo class for object detection using Yolo v3.
"""

import numpy as np
import keras


class Yolo:
    """
    Yolo class for object detection using Yolo v3.

    Attributes:
        model (keras.Model): The YOLO model loaded from the given path.
        class_names (list): List of class names loaded from the given path.
        class_t (float): Threshold for class score.
        nms_t (float): Non-max suppression threshold.
        anchors (list): List of anchor boxes.
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initializes the Yolo object with the given parameters.

        Args:
            model_path (str): Path to the YOLO model file.
            classes_path (str): Path to the file containing class names.
            class_t (float): Threshold for class score.
            nms_t (float): Non-max suppression threshold.
            anchors (list): List of anchor boxes.
        """
        self.model = keras.models.load_model(model_path, compile=False)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = np.array(anchors)

    def process_outputs(self, outputs, image_size):
        """
        Processes the outputs from the Darknet model.

        Args:
            outputs (list): List of numpy.ndarrays
            containing the predictions from the model.
            image_size (numpy.ndarray): Image's original
            size [image_height, image_width].

        Returns:
            tuple: (boxes, box_confidences, box_class_probs)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size

        for output, anchors in zip(outputs, self.anchors):
            grid_height, grid_width, anchor_boxes = output.shape[:3]
            box = output[..., :4]
            box_confidence = 1 / (1 + np.exp(-output[..., 4, np.newaxis]))
            box_class_prob = 1 / (1 + np.exp(-output[..., 5:]))

            tx, ty, tw, th = box[..., 0], box[..., 1], box[..., 2], box[..., 3]

            # Create the grid for bx and by
            cx = np.tile(np.arange(grid_width),
                         (grid_height, 1)).reshape((grid_height,
                                                    grid_width,
                                                    1))
            cy = np.tile(np.arange(grid_height),
                         (grid_width, 1)).T.reshape((grid_height,
                                                     grid_width,
                                                     1))

            # Normalize the bx and by
            bx = (1 / (1 + np.exp(-tx))) + cx
            by = (1 / (1 + np.exp(-ty))) + cy

            # Normalize the bw and bh using anchors
            bw = np.exp(tw) * anchors[:, 0].reshape((1, 1, len(anchors)))
            bh = np.exp(th) * anchors[:, 1].reshape((1, 1, len(anchors)))

            bx /= grid_width
            by /= grid_height
            bw /= self.model.input.shape[1]
            bh /= self.model.input.shape[2]

            # Convert to x1, y1, x2, y2
            x1 = (bx - (bw / 2)) * image_width
            y1 = (by - (bh / 2)) * image_height
            x2 = (bx + (bw / 2)) * image_width
            y2 = (by + (bh / 2)) * image_height

            box[..., 0], box[..., 1], box[..., 2], box[..., 3] = x1, y1, x2, y2

            boxes.append(box)
            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters the boxes based on box scores.
        Returns:
            tuple: (filtered_boxes, box_classes, box_scores)
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            box = boxes[i].reshape(-1, 4)
            box_confidence = box_confidences[i].reshape(-1)
            box_class_prob =\
                box_class_probs[i].reshape(-1, box_class_probs[i].shape[-1])

            box_scores_current = box_confidence[:, np.newaxis] * box_class_prob
            box_classes_current = np.argmax(box_scores_current, axis=1)
            box_class_scores_current = np.max(box_scores_current, axis=1)

            filtering_mask = box_class_scores_current >= self.class_t

            filtered_boxes.append(box[filtering_mask])
            box_classes.append(box_classes_current[filtering_mask])
            box_scores.append(box_class_scores_current[filtering_mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Performs Non-max suppression (NMS) on filtered boxes.

        Args:
            filtered_boxes (numpy.ndarray): Array of shape (?, 4) containing all of the filtered bounding boxes.
            box_classes (numpy.ndarray): Array of shape (?,) containing the class number for each box in filtered_boxes.
            box_scores (numpy.ndarray): Array of shape (?) containing the box scores for each box in filtered_boxes.

        Returns:
            tuple: (box_predictions, predicted_box_classes, predicted_box_scores)
                box_predictions (numpy.ndarray): Array of shape (?, 4) containing all of the predicted bounding boxes ordered by class and box score.
                predicted_box_classes (numpy.ndarray): Array of shape (?,) containing the class number for each box in box_predictions ordered by class and box score.
                predicted_box_scores (numpy.ndarray): Array of shape (?) containing the box scores for each box in box_predictions ordered by class and box score.
        """
        # Sort boxes based on scores and classes in descending order
        idxs = np.argsort(-box_scores)
        filtered_boxes = filtered_boxes[idxs]
        box_classes = box_classes[idxs]
        box_scores = box_scores[idxs]

        # Initialize lists to store results
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        while len(filtered_boxes) > 0:
            # Pick the box with the highest score
            best_box = filtered_boxes[0]
            best_class = box_classes[0]
            best_score = box_scores[0]

            box_predictions.append(best_box)
            predicted_box_classes.append(best_class)
            predicted_box_scores.append(best_score)

            # Compute IoU (Intersection over Union) with remaining boxes
            iou = self.compute_iou(best_box, filtered_boxes[1:])
            
            # Find the indices of boxes with IoU less than nms_t
            iou_mask = iou < self.nms_t

            # Filter out boxes with high IoU
            filtered_boxes = filtered_boxes[1:][iou_mask]
            box_classes = box_classes[1:][iou_mask]
            box_scores = box_scores[1:][iou_mask]

        # Convert lists to numpy arrays
        box_predictions = np.array(box_predictions)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores

    def compute_iou(self, box, boxes):
        """
        Computes IoU (Intersection over Union) between a box and an array of boxes.

        Args:
            box (numpy.ndarray): Box to compare, shape (4,).
            boxes (numpy.ndarray): Array of boxes to compare against, shape (?, 4).

        Returns:
            numpy.ndarray: Array of IoU values, shape (?).
        """
        # Calculate intersection coordinates
        intersection_x1 = np.maximum(box[0], boxes[:, 0])
        intersection_y1 = np.maximum(box[1], boxes[:, 1])
        intersection_x2 = np.minimum(box[2], boxes[:, 2])
        intersection_y2 = np.minimum(box[3], boxes[:, 3])

        # Calculate intersection area
        intersection_area = np.maximum(0, intersection_x2 - intersection_x1 + 1) * np.maximum(0, intersection_y2 - intersection_y1 + 1)

        # Calculate union area
        box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
        boxes_area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
        union_area = box_area + boxes_area - intersection_area

        # Calculate IoU
        iou = intersection_area / union_area

        return iou

    def load_images(folder_path):
        """
        Loads images from a folder path.

        Args:
            folder_path (str): Path to the folder containing images.

        Returns:
            tuple: (images, image_paths)
                images (list of numpy.ndarray): List of images as numpy arrays.
                image_paths (list of str): List of paths to the individual images.
        """
        images = []
        image_paths = []

        # Iterate through all files in the folder
        for filename in os.listdir(folder_path):
            # Check if the file is an image file (common image extensions)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                # Construct the full path to the image file
                file_path = os.path.join(folder_path, filename)
                # Read the image using OpenCV
                image = cv2.imread(file_path)
                if image is not None:
                    # Append the loaded image and its path to the lists
                    images.append(image)
                    image_paths.append(file_path)

        return images, image_paths