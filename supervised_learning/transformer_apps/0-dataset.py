#!/usr/bin/env python3
"""
Dataset module for loading and preparing a dataset for machine translation.
"""
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """
    Dataset class to load and prep a dataset for machine translation.z
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
        """
        tokenizer_pt = transformers.BertTokenizer.\
            from_pretrained('neuralmind/bert-base-portuguese-case')
        tokenizer_en = transformers.BertTokenizer.\
            from_pretrained('bert-base-uncased')

        def get_training_corpus_en():
            for _, en in data:
                yield en.numpy().decode('utf-8')

        def get_training_corpus_pt():
            for pt, _ in data:
                yield pt.numpy().decode('utf-8')

        tokenizer_pt = tokenizer_pt.\
            train_new_from_iterator(get_training_corpus_pt(), vocab_size=2**13)
        tokenizer_en = tokenizer_en.\
            train_new_from_iterator(get_training_corpus_en(), vocab_size=2**13)
        return tokenizer_pt, tokenizer_en
