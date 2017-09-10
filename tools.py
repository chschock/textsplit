from sklearn.feature_extraction.text import CountVectorizer
from .algorithm import split_greedy, split_optimal

import nltk.data
sentence_analyser = nltk.data.load('tokenizers/punkt/english.pickle')

def sentences_iter(texts):
    for text in texts:
        yield sentence_analyser.tokenize(text)

def sentences_vectors_iter(sentenced_texts, wordvecs, vecr_kwargs=dict()):
    """
    Returns generator of pairs of a list of sentences and their vectorization
    matrix. Sentence vectors are just the sum of word vectors in a sentence.
    """
    vecr = CountVectorizer(vocabulary=wordvecs.index, **vecr_kwargs)

    for sentences in sentenced_texts:
        yield (sentences, vecr.transform(sentences).dot(wordvecs))

def split_texts_greedy(sentences_vectors_texts, penalty):
    """
    Takes an iterator over documents of pairs of sentences and their count
    vectorization.
    Returns generater with list of segments by splitting documents greedy
    with given `penalty`.
    """
    for sentences, vectors in sentences_vectors_texts:
        L = vectors.shape[0]
        seg = split_greedy(vectors, penalty)
        segmented_text = []
        for beg, end in zip([0] + seg.splits, seg.splits + [L]):
            segmented_text.append(sentences[beg:end])
        yield segmented_text

def split_texts_optimal(sentences_vectors_texts, penalty, seg_limit=None):
    """
    Takes an iterator over documents of pairs of sentences and their count
    vectorization.
    Returns generater with list of segments by splitting documents greedy
    with given `penalty`.
    """
    for i_text, (sentences, vectors) in enumerate(sentences_vectors_texts):
        L = vectors.shape[0]
        seg = split_optimal(vectors, penalty, seg_limit=seg_limit)
        segmented_text = []
        for beg, end in zip([0] + seg.splits, seg.splits + [L]):
            segmented_text.append(sentences[beg:end])
        if not seg.optimal:
            print('segmentation not optimal for document %d' % i_text)
        yield segmented_text
