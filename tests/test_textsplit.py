import unittest
import numpy as np
from ..textsplit import split_greedy, split_optimal, P_k
from ..textsplit import get_total, get_penalty, get_gains

DIM = 20

def getDoc(segment_len, n_seg):
    return np.vstack([np.tile(w, (segment_len, 1))
                      for w in np.random.random((n_seg, DIM))])

docA = getDoc(20, 10)
penaltyA = get_penalty([docA], 20)

class TestTextSplit(unittest.TestCase):

    def test_get_penalty(self):
        total, splits, gains = split_greedy(docA, min_gain=penaltyA)
        self.assertEqual(len(splits), 9)

    def test_split_greedy_min_gain(self):
        total, splits, gains = split_greedy(docA, min_gain=penaltyA)
        self.assertEqual(len(splits), len(gains))
        self.assertGreater(np.percentile(gains, 25), penalty)
        gains2 = get_gains(docA, splits)
        self.assertTrue(all(np.isclose(gains, gains2)))

    def test_split_greedy_max_splits(self):
        total, splits, gains = split_greedy(docA, max_splits=5)
        self.assertEqual(len(splits), len(gains))
        self.assertTrue(len(splits) == len(gains) == 5)

    def test_split_greedy_min_gain_max_splits(self):
        total, splits, gains = split_greedy(docA, min_gain=penaltyA, max_splits=5)
        self.assertEqual(len(splits), len(gains))
        self.assertEqual(len(splits), 5)
        self.assertGreater(np.percentile(gains, 25), penaltyA)

    def test_split_greedy_min_gain(self):
        total, splits, gains = split_greedy(docA, min_gain=penaltyA)
        self.assertEqual(len(splits), len(gains))
        self.assertGreater(np.percentile(gains, 25), penaltyA)

    def test_split_optimal(self):
        total, splits, gains = split_optimal(docA, penalty=penaltyA)
        self.assertEqual(len(splits), len(gains))
        self.assertGreater(np.min(gains), penaltyA)

    def test_split_optimal_vs_greedy(self):
        docs = [np.random.random((100, DIM)) for _ in range(100)]
        penalty = get_penalty(docs, 10)
        for i, doc in enumerate(docs):
            tot_o, splits_o, gains_o = split_optimal(doc, penalty=penalty)
            tot_g, splits_g, gains_g = split_greedy(doc, min_gain=penalty)
            self.assertAlmostEqual(tot_o, get_total(doc, splits_o, penalty), places=3)
            self.assertAlmostEqual(tot_g, get_total(doc, splits_g, penalty), places=3)
            self.assertGreaterEqual(tot_o + 0.001, tot_g)

    def test_split_optimal_with_seg_limit(self):
        docs = [np.random.random((100, DIM)) for _ in range(10)]
        penalty = get_penalty(docs, 20)
        for i, doc in enumerate(docs):
            total, splits, gains = split_optimal(doc, penalty=penalty)
            cuts = [0] + splits + [100]
            total2, splits2, gains2, optimal = split_optimal(
                doc, penalty=penalty, seg_limit=np.diff(cuts).max()+1)
            self.assertTrue(optimal)
            self.assertEqual(splits, splits2)
            self.assertAlmostEqual(total, total2)

    def test_P_k(self):
        docs = [np.random.random((100, DIM)) for _ in range(10)]
        penalty = get_penalty(docs, 10)
        for i, doc in enumerate(docs):
            tot_o, splits_o, gains_o = split_optimal(doc, penalty=penalty)
            tot_g, splits_g, gains_g = split_greedy(doc, min_gain=penalty)
            pk = P_k(splits_o, splits_g, len(doc))
            self.assertGreaterEqual(pk, 0)
            self.assertGreaterEqual(1, pk)
