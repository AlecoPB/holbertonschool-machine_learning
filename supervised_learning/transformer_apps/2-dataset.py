#!/usr/bin/env python3
"""
This is some documentation
"""
import tensorflow_datasets as tfds
import transformers
import tensorflow as tf


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

    def tf_encode(self, pt, en):
        """
        TensorFlow wrapper for the encode instance method.

        Parameters:
        - pt: tf.Tensor containing the Portuguese sentence.
        - en: tf.Tensor containing the English sentence.

        Returns:
        - pt_tokens: tf.Tensor containing the tokenized Portuguese sentence.
        - en_tokens: tf.Tensor containing the tokenized English sentence.
        """
        # Apply encode using tf.py_function to make it TensorFlow-friendly
        pt_tokens, en_tokens = tf.py_function(self.encode, [pt, en], [tf.int64, tf.int64])

        # Set shape of the tensors after tokenization
        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])

        # pt_tokens = tf.strings.to_number(pt_tokens, out_type=tf.int64)
        # en_tokens = tf.strings.to_number(en_tokens, out_type=tf.int64)

        return pt_tokens, en_tokens
