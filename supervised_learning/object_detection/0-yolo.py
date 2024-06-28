#!/usr/bin/env python3
"""
This is some documentation
"""
import keras


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
            boxes.append(outputs[i])
            box_confidences.append(outputs[i])
            box_class_probs.append(outputs[i])
        boxes.append(outputs[3])
        box_confidences.append(outputs[4])
        box_class_probs.append(outputs[5])
        return (boxes, box_confidences, box_class_probs)
