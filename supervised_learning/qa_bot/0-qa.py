#!/usr/bin/env python3
"""
Question Answering
"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    #Load model from tf_hub
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    #Load pre-trained tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad')