import unittest
import numpy as np
from ..textsplit.algorithm import split_greedy, split_optimal, get_total, get_gains
from ..textsplit.tools import get_penalty, P_k

DIM = 20

def getDoc(segment_len, n_seg):
    return np.vstack([np.tile(w, (segment_len, 1))
                      for w in np.random.random((n_seg, DIM))])


docA = getDoc(20, 10)
penaltyA = get_penalty([docA], 20)  # get_penalty is deterministic here

class TestTextSplit(unittest.TestCase):

    def test_get_penalty(self):
        seg = split_greedy(docA, penalty=penaltyA)
        self.assertEqual(len(seg.splits), 9)

    def test_split_greedy_penalty(self):
        seg = split_greedy(docA, penalty=penaltyA)
        self.assertEqual(len(seg.splits), len(seg.gains))
        self.assertGreater(np.percentile(seg.gains, 25), penaltyA)
        gains2 = get_gains(docA, seg.splits)
        self.assertTrue(all(np.isclose(seg.gains, gains2)))

    def test_split_greedy_max_splits(self):
        seg = split_greedy(docA, max_splits=5)
        self.assertEqual(len(seg.splits), len(seg.gains))
        self.assertTrue(len(seg.splits) == len(seg.gains) == 5)

    def test_split_greedy_penalty_max_splits(self):
        seg = split_greedy(docA, penalty=penaltyA, max_splits=5)
        self.assertEqual(len(seg.splits), len(seg.gains))
        self.assertEqual(len(seg.splits), 5)
        self.assertGreater(np.percentile(seg.gains, 25), penaltyA)

    def test_split_optimal(self):
        seg = split_optimal(docA, penalty=penaltyA)
        self.assertEqual(len(seg.splits), len(seg.gains))
        print(len(seg.splits))
        self.assertGreater(np.min(seg.gains) + 0.00001, penaltyA)

    def test_split_optimal_vs_greedy(self):
        docs = [np.random.random((100, DIM)) for _ in range(100)]
        penalty = get_penalty(docs, 10)
        for i, doc in enumerate(docs):
            seg_o = split_optimal(doc, penalty=penalty)
            seg_g = split_greedy(doc, penalty=penalty)
            self.assertAlmostEqual(seg_o.total, get_total(doc, seg_o.splits, penalty), places=3)
            self.assertAlmostEqual(seg_g.total, get_total(doc, seg_g.splits, penalty), places=3)
            self.assertGreaterEqual(seg_o.total + 0.001, seg_g.total)

    def test_split_optimal_with_seg_limit(self):
        docs = [np.random.random((100, DIM)) for _ in range(10)]
        penalty = get_penalty(docs, 20)
        for i, doc in enumerate(docs):
            seg = split_optimal(doc, penalty=penalty)
            cuts = [0] + seg.splits + [100]
            seg2 = split_optimal(
                doc, penalty=penalty, seg_limit=np.diff(cuts).max()+1)
            self.assertTrue(seg2.optimal)
            self.assertEqual(seg.splits, seg2.splits)
            self.assertAlmostEqual(seg.total, seg2.total)

    def test_P_k(self):
        docs = [np.random.random((100, DIM)) for _ in range(10)]
        penalty = get_penalty(docs, 10)
        for i, doc in enumerate(docs):
            seg_o = split_optimal(doc, penalty=penalty)
            seg_g = split_greedy(doc, penalty=penalty)
            pk = P_k(seg_o.splits, seg_g.splits, len(doc))
            self.assertGreaterEqual(pk, 0)
            self.assertGreaterEqual(1, pk)
