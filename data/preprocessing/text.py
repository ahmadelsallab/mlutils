
class TextFeatures:
    def __init__(self, texts):
        pass

    def vectorize(self, text):
        raise NotImplementedError("vectorize to be implemented according to the exact representation method")

    def _build_vocab(self, text):
        pass

    def _fit_tokenizer(self, text):
        pass

    def _calc_max_len(self, texts):
        pass

    def normalize(self, text):
        pass

    def tokenize(self, text):
        pass

    def pad(self, text, max_len):
        pass


class SequenceFeatures(TextFeatures):
    def __init__(self, texts):
        super().__init__(texts=texts)

    def vectorize(self, text):
        pass


class BoWFeatures(TextFeatures):
    def __init__(self, texts):
        super().__init__(texts=texts)

    def vectorize(self, text):
        pass


class CountFeatures(TextFeatures):
    def __init__(self, texts):
        super().__init__(texts=texts)

    def vectorize(self, text):

        pass


class TFIDFFeatures(TextFeatures):
    def __init__(self, texts):
        super().__init__(texts=texts)

    def vectorize(self, text):
        pass

