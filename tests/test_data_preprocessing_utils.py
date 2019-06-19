from unittest import TestCase

import data.preprocessing.text.Utils as utils


class Test_data_preprocessing_utils(TestCase):
    def test_calc_max_len_from_text(self):
        texts = ['This is test1', 'This is another test', 'This is the last test']
        maxlen = utils.calc_max_len_from_text(texts)
        self.assertEqual(maxlen, 5)

    def test_calc_max_len_from_ids(self):
        sequences = [[100, 2, 19, 1, 0],[-1, 199, 90000, 4],[1, 2]]
        maxlen = utils.calc_max_len_from_ids(sequences)
        self.assertEqual(maxlen, 5)
