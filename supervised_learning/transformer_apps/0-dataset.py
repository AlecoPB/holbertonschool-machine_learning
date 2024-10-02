#!/usr/bin/env python3

"""
Dataset module for loading and preparing a dataset for machine translation.
"""

import tensorflow_datasets as tfds
import transformers

class Dataset:
    """
    Dataset class to load and prep a dataset for machine translation.
    """

    def __init__(self):
        """
        Initializes the Dataset instance.
        """
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en', split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en', split='validation', as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for the dataset.

        Args:
            data: tf.data.Dataset whose examples are formatted as a tuple (pt, en).
                  pt is the tf.Tensor containing the Portuguese sentence.
                  en is the tf.Tensor containing the corresponding English sentence.

        Returns:
            tokenizer_pt: The Portuguese tokenizer.
            tokenizer_en: The English tokenizer.
        """
        tokenizer_pt = transformers.BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
        tokenizer_en = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

        # Optional: Set max vocab size, although BERT tokenizers are pretrained and fixed
        # tokenizer_pt.train_new_from_iterator((pt.numpy() for pt, en in data), vocab_size=2**13)
        # tokenizer_en.train_new_from_iterator((en.numpy() for pt, en in data), vocab_size=2**13)

        return tokenizer_pt, tokenizer_en