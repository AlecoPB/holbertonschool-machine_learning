import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import glob
import os
from sklearn.metrics.pairwise import cosine_similarity


def semantic_search(corpus_path, sentence):

    def read_documents(directory_path, extension='*.md'):
        """
        Read documents from a directory
        """
        file_paths = glob.glob(os.path.join(directory_path, extension))
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                yield f.read()

    # 1. Load the Universal Sentence Encoder from TensorFlow Hub
    model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    # 2. Load the corpus of reference documents
    documents = list(read_documents(corpus_path))

    # 3. Encode the reference documents into vectors in one go
    doc_embeddings = model(documents).numpy()  # Convert to numpy arrays

    # 4. Encode the search sentence into a vector
    query_embedding = model([sentence]).numpy()

    # 5. Compute cosine similarities between the sentence and the documents
    cosine_similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

    # 6. Find the index of the most similar document
    most_similar_idx = np.argmax(cosine_similarities)

    # 7. Return the most similar document
    return documents[most_similar_idx]
