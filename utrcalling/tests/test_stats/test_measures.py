"""Tests for utrcalling.core.stats.measures: ranks, length index and entropy of site use."""

import unittest

import numpy as np

from utrcalling.core.stats import measures


class TestSiteRanks(unittest.TestCase):
    def test_ranks_go_from_the_shortest_to_the_longest_site_of_each_utr(self):
        ranks = measures.site_ranks(["A", "B", "A", "A", "B"], [300, 20, 10, 150, 90])
        np.testing.assert_allclose(ranks, [1.0, 0.0, 0.0, 0.5, 1.0])

    def test_equal_sizes_share_their_mean_rank(self):
        ranks = measures.site_ranks(["A"] * 4, [10, 20, 20, 30])
        np.testing.assert_allclose(ranks, [0.0, 0.5, 0.5, 1.0])

    def test_single_site_utrs_have_no_rank(self):
        ranks = measures.site_ranks(["A", "B", "B"], [10, 20, 30])
        self.assertTrue(np.isnan(ranks[0]))
        np.testing.assert_allclose(ranks[1:], [0.0, 1.0])


class TestLengthIndexAndEntropy(unittest.TestCase):
    def test_length_index_is_the_share_weighted_mean_rank(self):
        shares = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.25, 0.5, 0.25], [0.5, 0.0, 0.5]])
        index = measures.length_index(shares, [0.0, 0.5, 1.0])
        np.testing.assert_allclose(index, [0.0, 1.0, 0.5, 0.5])

    def test_entropy_goes_from_one_site_to_equal_use(self):
        shares = np.array([[1.0, 0.0, 0.0], [1 / 3, 1 / 3, 1 / 3], [0.5, 0.5, 0.0]])
        entropy = measures.normalized_entropy(shares)
        np.testing.assert_allclose(entropy, [0.0, 1.0, np.log(2) / np.log(3)])

    def test_entropy_tells_apart_what_the_length_index_cannot(self):
        # The same mean rank, from the middle site or from both ends.
        shares = np.array([[0.0, 1.0, 0.0], [0.5, 0.0, 0.5]])
        index = measures.length_index(shares, [0.0, 0.5, 1.0])
        entropy = measures.normalized_entropy(shares)
        self.assertAlmostEqual(index[0], index[1])
        self.assertLess(entropy[0], entropy[1])
