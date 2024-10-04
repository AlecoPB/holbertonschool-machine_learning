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
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True,)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for the dataset.
        """
        tokenizer_pt = transformers.BertTokenizer.\
            from_pretrained('neuralmind/bert-base-portuguese-cased', max_length = 2**13)
        tokenizer_en = transformers.BertTokenizer.\
            from_pretrained('bert-base-uncased', max_length = 2**13)

        return tokenizer_pt, tokenizer_en
