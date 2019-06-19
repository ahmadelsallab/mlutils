from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
import numpy as np

from keras.preprocessing.text import Tokenizer as KerasTokenizer
from collections import Counter

class Utils:

    def __init__(self):
        """

        :param library: keras, nltk, spacy
        :type library:
        """
        pass

    @staticmethod
    def fit_tokenizer(texts, library='keras', vocab_size=None, oov_token=None):
        if library=='keras':
            if vocab_size:
                tokenizer = KerasTokenizer(nb_words=vocab_size, oov_token=oov_token)
            else:
                tokenizer = KerasTokenizer(oov_token=oov_token)
            tokenizer.fit_on_texts(texts)

        return tokenizer

    @staticmethod
    def str2seq(text, library='string'):
        """
        Abstraction of text to sequence
        :param text: string
        :type text: basestring
        :param library: tools to use to transfrom string to seqeuence: keras or string (split)
        :return:
        :rtype:
        """
        if library=='string':
            return text.split()
        elif library=='keras':
            return text_to_word_sequence(text)

    @classmethod
    def calc_max_len_from_text(cls, texts):
        """
        This is different from TextFeatures method, as this one just uses split without tokenization
        :param texts: list of strings
        :type texts: list

        :return: maxlen
        :rtype: int
        """
        return max([len(cls.str2seq(text)) for text in texts])

    @staticmethod
    def calc_max_len_from_ids(sequences):
        """
        Same as calc_max_len_from_text, but the input is list of tokens ids (not strings)
        :param texts: list of lists (sequence of tokens ids)
        :type texts: list
        :return: maxlen
        :rtype: int
        """
        return max([len(seq) for seq in sequences])

    @staticmethod
    def normalize(text):
        # TODO: Must lower

    @staticmethod
    def pad(texts, maxlen, library='keras'):
        """
        Append 0's to the end
        :param texts: list of lists (sequence of tokens)
        :type texts: list
        :param maxlen: max length of text in texts
        :type maxlen: int
        :param library: keras
        :return: padded sequence as numpy array
        :rtype: np.array
        """
        if library=='keras':
            return np.array(pad_sequences(texts,
                                          maxlen=maxlen,
                                          padding='post',
                                          truncating='post'))


    @staticmethod
    def load_embeddings(embeddings_file, str2int, vocab_size, embedding_dim):
        embeddings_index = {}
        f = open(embeddings_file)
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

        f.close()


        embedding_matrix = np.random.random((vocab_size+1, embedding_dim))

        for word, i in str2int.items():
          embedding_vector = embeddings_index.get(word)
          if embedding_vector is not None:
              # words not found in embedding index will be random.
              embedding_matrix[i] = embedding_vector
        return embedding_matrix

class Vocabulary:
    UNK_TOK = '_UNK_'
    UNK_ID = 0  # 0 index is reserved for the UNK in both Keras Tokenizer and Embedding

    def __init__(self, texts=None, library='string', tokenizer=None, vocab_size=None, oov_token=None, stop_words=None):
        if texts: self.build_vocab(texts, library, tokenizer, vocab_size, oov_token, stop_words)

    def build_vocab(self, texts, library='string', tokenizer=None, vocab_size=None, oov_token=None, stop_words=None):
        if library == 'string':
            # TODO: use normalize instead of lower
            words = [word.lower() for text in texts for word in Utils.str2seq(text, library=library)]
            word_counts = Counter(words)

            # Sort by most frequent. Not that .items() is a tuple, and counts is at index [1] of that tuple
            word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

            # Take most frequent as per vocab_size
            self.str2idx = {key:val for key,val in word_counts[:vocab_size]}

            # Insert UNK
            self.str2idx[oov_token] = UNK_ID


        elif library == 'keras':
            self.str2idx = tokenizer.word_index
            # oov is handled in the tokenizer

        self.idx2str = dict([(value, key) for (key, value) in self.str2idx.items()])

    @property
    def vocab(self):
        return self.str2idx.keys()



class TextFeatures:
    # Note that: it's highly recommended to use 'keras', since the tokenizer does few important things
    # 1. Splits the sentence words based on punctuations, not only spaces
    # 2. Can consider stop words
    # 3. Filters non important tokens when building vocab
    # 4. Considers lower in vocab and extraction
    def __init__(self, texts, vocab_size=None, library='keras', oov_token=None, stop_words=None):
        # TODO: stop words not in keras. Use spacy.

        if library == 'keras':
            self.tokenizer = Utils.fit_tokenizer(texts, vocab_size, oov_token)
        elif library == 'string':
            self.tokenizer = None

        self.vocab = Vocabulary(texts, library, self.tokenizer, vocab_size)
        self.library = library

    def text2features(self, texts, maxlen, pad=True):
        raise NotImplementedError("extract to be implemented according to the exact representation method")

    def features2text(self, features):
        if self.library == 'keras':
            return self.tokenizer.sequences_to_texts(features)
        elif self.library == 'string':
            return [' '.join([self.vocab.idx2str[idx] for idx in vec]) for vec in features]



class SequenceFeatures(TextFeatures):
    def __init__(self, texts):
        super().__init__(texts=texts)

    def text2features(self, texts, maxlen, pad=True):
        if self.library == 'keras':
            features = self.tokenizer.texts_to_sequences(texts)

        elif self.library == 'string':
            # Dont forget to lower as the vocab is built like that
            # TODO: use normalize instead of lower
            features = [[self.vocab.str2idx[word.lower()] for word in Utils.str2seq(text)] for text in texts]

        if pad:
            return Utils.pad(features, maxlen, self.library)
        else:
            return features

class BoWFeatures(TextFeatures):
    def __init__(self, texts):
        super().__init__(texts=texts)

    def text2features(self, texts):
        pass


class CountFeatures(TextFeatures):
    def __init__(self, texts):
        super().__init__(texts=texts)

    def text2features(self, texts):

        pass


class TFIDFFeatures(TextFeatures):
    def __init__(self, texts):
        super().__init__(texts=texts)

    def text2features(self, texts):
        pass

