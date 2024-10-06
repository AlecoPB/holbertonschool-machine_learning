#!/usr/bin/env python3
"""
This is some documentation
"""
import tensorflow_datasets as tfds
import transformers
import numpy as np


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
                                    split='validation', as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en =\
            self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """
        Generate sub-word tokenizers
        """

        tokenizer_pt = transformers.AutoTokenizer.\
            from_pretrained('neuralmind/bert-base-portuguese-cased')
        tokenizer_en = transformers.AutoTokenizer.\
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

    def encode(self, pt, en):
        """
        Encodes a translation into tokens.

        Parameters:
        - pt: tf.Tensor, the Portuguese sentence.
        - en: tf.Tensor, the English sentence.

        Returns:
        - pt_tokens: np.ndarray containing the Portuguese tokens.
        - en_tokens: np.ndarray containing the English tokens.
        """
    def encode(self, pt, en):
        """
        Instance method
        """
        if tf.is_tensor(pt):
            pt = pt.numpy().decode('utf-8')
        if tf.is_tensor(en):
            en = en.numpy().decode('utf-8')

        # nouveaux indexs pour les tokens CLS et SEP
        nouveau_cls_id = 8192
        nouveau_sep_id = 8193

        # Exemple de tokenization manuelle avec vos propres IDs
        pt_tokens = ([nouveau_cls_id] +
                     self.tokenizer_pt.encode(pt, add_special_tokens=False) +
                     [nouveau_sep_id])
        en_tokens = ([nouveau_cls_id] +
                     self.tokenizer_en.encode(en, add_special_tokens=False) +
                     [nouveau_sep_id])
        # print("encode en", self.tokenizer_en.encode(en))
        return pt_tokens, en_tokens
