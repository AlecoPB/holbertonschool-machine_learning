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

    def __init__(self, batch_size, max_len):
        """
        Class constructor
        """
        self.batch_size = batch_size
        self.max_len = max_len

        self.data_train = tfds.load(
            'ted_hrlr_translate/pt_to_en', split='train', as_supervised=True)
        self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='validation', as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = (
            self.tokenize_dataset(self.data_train)
        )

        #  map est une fonction tres utile dans TensorFlow qui permet
        #  d'appliquer une transformation à chaque élément d'un dataset
        # a la maniere de map python
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

        def mask_maxlen(pt, en):
            """
            filter out examples that have more than max_len tokens
            idk why its working
            """
            return tf.logical_and(
                tf.size(pt) <= self.max_len,
                tf.size(en) <= self.max_len
            )

        self.data_train = self.data_train.filter(mask_maxlen)

        # Caches the elements in this dataset.
        # The first time the dataset is iterated over, its elements
        # will be cached either in the specified file or in memory.
        # Subsequent iterations will use the cached data.

        self.data_train = self.data_train.cache()
        self.data_train = self.data_train.shuffle(buffer_size=20000)

        self.data_train = self.data_train.padded_batch(self.bacth_size)

        self.data_train = self.data_train.prefetch(
            tf.data.experimental.AUTOTUNE)

        self.data_valid = self.data_valid.filter(mask_maxlen)
        self.data_valid = self.data_valid.padded_batch(self.bacth_size)

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
        Encode the Portuguese and English sentences using the tokenizers.
        """
        if tf.is_tensor(pt):
            pt = pt.numpy().decode('utf-8')
        if tf.is_tensor(en):
            en = en.numpy().decode('utf-8')

        nouveau_cls_id = 8192
        nouveau_sep_id = 8193

        pt_tokens = ([nouveau_cls_id] +
                     self.tokenizer_pt.encode(pt, add_special_tokens=False) +
                     [nouveau_sep_id])
        en_tokens = ([nouveau_cls_id] +
                     self.tokenizer_en.encode(en, add_special_tokens=False) +
                     [nouveau_sep_id])
        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        TensorFlow wrapper for the encode instance method.
        """
        # Apply encode using tf.py_function
        encoder = tf.py_function(func=self.encode, inp=[pt, en],
                                 Tout=[tf.int64, tf.int64])

        # Set shape of the tensors after tokenization
        pt_tensor = tf.ensure_shape(encoder[0], [None])
        en_tensor = tf.ensure_shape(encoder[1], [None])

        print("pt_tensor type", type(pt_tensor))

        return pt_tensor, en_tensor
