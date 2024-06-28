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
