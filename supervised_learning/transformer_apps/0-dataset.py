#!/usr/bin/env python3
"""
This is some documentation
"""
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """
    Dataset class
    """
    def __init__(self):
        self.data_train, self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split=['train', 'validation'],
            as_supervised=True
        )
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """
        Method to tokenize a dataset
        """
        # Load pre-trained BERT tokenizers
        tokenizer_pt = transformers.BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
        tokenizer_en = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

        # Tokenize the dataset
        def tokenize_pt_en(pt, en):
            """
            Portuguese tokenizer
            """
            pt = tokenizer_pt.encode(pt.numpy().decode('utf-8'), add_special_tokens=True, truncation=True, max_length=2**13)
            en = tokenizer_en.encode(en.numpy().decode('utf-8'), add_special_tokens=True, truncation=True, max_length=2**13)
            return pt, en

        def tf_tokenize_pt_en(pt, en):
            pt, en = tf.py_function(tokenize_pt_en, [pt, en], [tf.int32, tf.int32])
            pt.set_shape([None])
            en.set_shape([None])
            return pt, en

        data = data.map(tf_tokenize_pt_en, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return tokenizer_pt, tokenizer_en
