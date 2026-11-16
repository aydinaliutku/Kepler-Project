import unittest

import numpy as np
import pandas as pd

from evaluate import FEATURES, held_out_split


class KeplerSplitTests(unittest.TestCase):
    def test_star_never_crosses_holdout_boundary(self):
        stars = np.repeat(np.arange(100), 2)
        labels = np.tile([0, 1], 100)
        samples = pd.DataFrame(np.zeros((len(stars), len(FEATURES))), columns=FEATURES)
        train, test = held_out_split(samples, labels, stars)
        self.assertFalse(set(stars[train]) & set(stars[test]))
        self.assertEqual(set(labels[test]), {0, 1})
        self.assertEqual(len(train) + len(test), len(stars))

    def test_vetting_fields_are_excluded(self):
        self.assertTrue(all(not name.startswith("koi_fpflag") for name in FEATURES))
        self.assertNotIn("koi_score", FEATURES)
        self.assertNotIn("kepid", FEATURES)
