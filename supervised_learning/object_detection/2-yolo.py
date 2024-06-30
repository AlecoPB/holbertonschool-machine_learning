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
            box_class_prob = box_class_probs[i].reshape(-1, box_class_probs[i].shape[-1])

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

