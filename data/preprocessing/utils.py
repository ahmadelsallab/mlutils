
def build_vocab(texts):
    pass


def calc_max_len_from_text(texts):
    """
    This is different from TextFeatures method, as this one just uses split without tokenization
    :param texts: list of lists (sequence of tokens)
    :type texts: list
    :return: maxlen
    :rtype: int
    """
    return max([len(text.split()) for text in texts])

def calc_max_len_from_ids(sequences):
    """
    Same as calc_max_len_from_text, but the input is list of tokens ids (not strings)
    :param texts: list of lists (sequence of tokens ids)
    :type texts: list
    :return: maxlen
    :rtype: int
    """
    return max([len(seq) for seq in sequences])


def normalize(text):
    pass


def pad(self, text, max_len):
    pass