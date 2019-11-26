import numpy as np
import re
import random
from .algorithm import split_greedy

def get_segments(text_particles, segmentation):
    """
    Reorganize text particles by aggregating them to arrays described by the
    provided `segmentation`.
    """
    segmented_text = []
    L = len(text_particles)
    for beg, end in zip([0] + segmentation.splits, segmentation.splits + [L]):
        segmented_text.append(text_particles[beg:end])
    return segmented_text

def get_penalty(docmats, segment_len):
    """
    Determine penalty for segments having length `segment_len` on average.
    This is achieved by stochastically rounding the expected number
    of splits per document `max_splits` and taking the minimal split_gain that
    occurs in split_greedy given `max_splits`.
    """
    penalties = []
    for docmat in docmats:
        avg_n_seg = docmat.shape[0] / segment_len
        max_splits = int(avg_n_seg) + (random.random() < avg_n_seg % 1) - 1
        if max_splits >= 1:
            seg = split_greedy(docmat, max_splits=max_splits)
            if seg.min_gain < np.inf:
                penalties.append(seg.min_gain)
    if len(penalties) > 0:
        return np.mean(penalties)
    raise ValueError('All documents too short for given segment_len.')


def P_k(splits_ref, splits_hyp, N):
    """
    Metric to evaluate reference splits against hypothesised splits.
    Lower is better.
    `N` is the text length.
    """
    k = round(N / (len(splits_ref) + 1) / 2 - 1)
    ref = np.array(splits_ref, dtype=np.int32)
    hyp = np.array(splits_hyp, dtype=np.int32)

    def is_split_between(splits, l, r):
        return np.sometrue(np.logical_and(splits - l >= 0, splits - r < 0))

    acc = 0
    for i in range(N-k):
        acc += is_split_between(ref, i, i+k) != is_split_between(hyp, i, i+k)

    return acc / (N-k)


class SimpleSentenceTokenizer:

    def __init__(self, breaking_chars='.!?'):
        assert len(breaking_chars) > 0
        self.breaking_chars = breaking_chars
        self.prog = re.compile(r".+?[{}]\W+".format(breaking_chars), re.DOTALL)

    def __call__(self, text):
        return self.prog.findall(text)
