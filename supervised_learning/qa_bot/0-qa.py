#!/usr/bin/env python3
"""
Question Answering
"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """
    Finds the answer to a question within a given text
    """
    #Load model from tf_hub
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    #Load pre-trained tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad')

    #Tokenize the input text
    inputs = tokenizer.encode_plus(question,
                                   reference,
                                   return_tensors='tf',
                                   return_attention_mask=True)

    input_ids = inputs['input_ids']
    token_type_ids = inputs['token_type_ids']
    attention_mask = inputs['attention_mask']

    #Pass inputs to the model
    outputs = model([input_ids, attention_mask, token_type_ids])

    #Extract start and end logits
    start_logits = outputs['start_logits']
    end_logits = outputs['end_logits']
    # start_logits, end_logits = outputs[0], outputs[1]

    #Get the most likely start and end token positions
    start_idx = tf.argmax(start_logits, axis=1).numpy()[0]
    end_idx = tf.argmax(end_logits, axis=1).numpy()[0]

    #Error handling
    if start_idx > end_idx:
        return "Unable to find a valid answer."

    #Convert token inx back to the text
    tokens = input_ids[0].numpy()[start_idx:end_idx+1]
    answer = tokenizer.decode(tokens,
                              skip_special_tokens=True,
                              clean_up_tokenization_spaces=True)

    return answer
