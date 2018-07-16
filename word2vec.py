"""
2018.07.16

References:
    deborausujono/word2vecpy (https://github.com/deborausujono/word2vecpy)
    Xin Rong, word2vec Parameter Learning Explained (https://arxiv.org/abs/1411.2738)
"""

from collections import defaultdict
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Word2Vec:
    """
    Train word vectors.

    Args:
        method: Word2Vec algorithm (string) - 'sg' (Skip-Gram) or 'cbow' (CBOW)

    Attributes:
        train: actually progress training
        most_similar: return top n most similar words given input word (cosine similarity)
    """
    def __init__(self, method='sg'):
        self.method = method

    def _build_vocab(self, corpus, min_count=3):
        """
        Build vocabulary from corpus.

        Args:
            corpus: target corpus to build vocabulary (list or file)
            min_count: minimum count to be included in vocabulary

        Return:
            vocab: dictionary mapping word on (its index and distribution) pair
            token_count: word count dictionary
        """
        token_count = defaultdict(int)
        token_list = []
        # when input type of corpus is file
        if type(corpus) == str:
            with open(corpus, 'r') as f:
                for line in f.read().splitlines():
                    tokens = line.strip().split()
                    for token in tokens:
                        token = token.lower()
                        token_count[token] += 1

        for line in corpus:
            tokens = line.strip().split()
            for token in tokens:
                token = token.lower()
                token_count[token] += 1

        # Build vocabulary if its count exceed min_count
        for word in token_count.keys():
            if token_count[word] >= min_count:
                token_list.append(word)

        # count for calculating unigram distribution for negative sampling
        vocab = {word: (idx, token_count[word]) for idx, word in enumerate(token_list)}

        return vocab, token_count

    def _build_samples(self, vocab, corpus,
                       window_size=5, threshold=1e-5):
        """
        Build samples for training consist of (center, context) which are subsampled.

        Usage:
            1. Build vocabulary calling self._build_vocab(corpus).
            2. Build training samples (center word, context) using vocabulary.

        Args:
            vocab: dictionary of vocabulary and their indices
            corpus: target corpus to train word2vec
            window_size: window size for context

        Return:
            samples with (center, context) words. context is list.
        """
        tokens = []
        samples = []

        if type(corpus) == str:
            with open(corpus, 'r') as f:
                for line in f.read().splitlines():
                    tokens.clear()
                    words = [word.lower() for word in line.strip().split()]
                    for word in words:
                        # prob: subsampling probability
                        keep_prob = 1 - np.sqrt(threshold / self.token_count[word])
                        # if random probability is higher than prob, add to tokens
                        if np.random.random() > keep_prob:
                            tokens.append(vocab.get(word))
                    for idx, token in enumerate(tokens):
                        # dict.get return None if key is not in dict
                        if token is None:
                            continue
                        s = max(0, idx - window_size)
                        e = min(idx + window_size, len(tokens) - 1)
                        context = tokens[s:idx] + tokens[idx:e]
                        while None in context:
                            context.remove(None)
                        if not context:
                            continue
                        samples.append((token, context))

        for line in corpus:
            words = [word.lower() for word in line.strip().split()]
            tokens.clear()
            for word in words:
                keep_prob = 1 - np.sqrt(threshold / self.token_count[word])
                if np.random.random() > keep_prob:
                    tokens.append(vocab.get(word))
            for idx, token in enumerate(tokens):
                if token is None:
                    continue
                s = max(0, idx-window_size)
                e = min(idx+window_size, len(tokens)-1)
                context = tokens[s:idx] + tokens[idx:e]
                while None in context:
                    context.remove(None)
                if not context:
                    continue
                samples.append((token, context))
        return samples

    def _neg_sampler(self, negative, pos_samples):
        # self.prob: probability for negative sampling
        # if positive samples selected, repeat sampling.
        samples = np.random.choice(self.tokens, negative, False, self.prob)
        while np.any(np.array([sample in pos_samples for sample in samples])):
            samples = np.random.choice(self.tokens, negative, False, self.prob)
        return samples

    def train(self, corpus, dim=100,
              window_size=3, negative=5, power=0.75,
              min_count=3, threshold=1e-5, eta=0.01):
        """
        Train word embedding vector.
            embedding vector: self.W1

        Args:
            corpus: corpus to train (list or file)
            dim: embedding size
            window_size: window size
            negative: the number of negative samples
            power: multiplier for unigram distribution for negative sampling
            min_count: minimum value to count
            threshold: subsampling threshold, default 1e-5
            eta: learning rate, default 0.01
        """
        self.vocab, self.token_count = self._build_vocab(corpus, min_count)
        self.tokens, prob = zip(*self.vocab.values())  # token id / its unigram distribution
        prob = np.power(np.array(prob), power)
        self.prob = prob / prob.sum()  # probability for negative sampling
        self.samples = self._build_samples(self.vocab, corpus, window_size, threshold)
        vocab_size = len(self.vocab)

        self.W1 = np.random.uniform(-0.5, 0.5, size=(vocab_size, dim))  # target weights
        self.D = np.random.uniform(-0.5, 0.5, size=(dim, dim))          # hidden layer with tanh
        self.W2 = np.random.uniform(-0.5, 0.5, size=(vocab_size, dim))

        # training starts here
        # 'sg' (Skip-Gram) is a default algorithm
        for center_token, context in self.samples:

            if self.method == 'cbow':
                # mean value of context vectors
                W1_mean = np.mean(self.W1[context], axis=0)
                g_w = np.zeros(dim)

                if negative > 0:
                    neg_samples = self._neg_sampler(negative, context)
                    clf = [(center_token, 1)] + [(target, 0) for target in neg_samples]

                    for target, label in clf:
                        # feedforward
                        a = np.dot(W1_mean, self.D)
                        z = np.tanh(a)
                        u = np.dot(z, self.W2[target])
                        y = sigmoid(u)

                        # backpropagation
                        e = label - y                       # ()
                        self.W2[target] += eta * e * z      # (N,)
                        dz = e * self.W2[target]            # (N,)
                        da = dz * (1 - np.tanh(a) ** 2)     # (N,)
                        dD = np.outer(da, W1_mean)          # (N,N)
                        self.D += eta * dD                  # (N,N)
                        g_w += eta * da

                    for context_token in context:
                        self.W1[context_token] += g_w

                else:
                    clf = [(center_token, 1)]

                    for target, label in clf:
                        # feedforward
                        a = np.dot(W1_mean, self.D)
                        z = np.tanh(a)
                        u = np.dot(z, self.W2[target])
                        y = np.exp(u) / np.sum(np.exp(np.dot(z, self.W2.T)))

                        # backpropagation
                        e = label - y                       # ()
                        self.W2[target] += eta * e * z      # (N,)
                        dz = e * self.W2[target]            # (N,)
                        da = dz * (1 - np.tanh(a) ** 2)     # (N,)
                        dD = np.outer(da, W1_mean)          # (N,N)
                        self.D += eta * dD                  # (N,N)
                        g_w += eta * da

                    for context_token in context:
                        self.W1[context_token] += g_w

            elif self.method == 'sg':
                for context_token in context:
                    g_w = np.zeros(dim)

                    if negative > 0:
                        neg_samples = self._neg_sampler(negative, context)
                        clf = [(context_token, 1)] + [(target, 0) for target in neg_samples]

                        for target, label in clf:
                            # feedforward
                            a = np.dot(self.W1[center_token], self.D)
                            z = np.tanh(a)
                            u = np.dot(z, self.W2[target])
                            y = sigmoid(u)

                            # backpropagation
                            e = label - y  # ()
                            self.W2[context_token] += eta * e * z     # (N,)
                            dz = e * self.W2[context_token]           # (N,)
                            da = dz * (1 - np.tanh(a) ** 2)           # (N,)
                            dD = np.outer(da, self.W1[center_token])  # (N,N)
                            self.D += eta * dD                        # (N,N)
                            g_w += eta * da                           # (N,)

                        self.W1[center_token] += g_w                  # (N,) update W1

                    else:
                        clf = [(context_token, 1)]

                        for target, label in clf:
                            # feedforward
                            a = np.dot(self.W1[center_token], self.D)               # (N,)(N,N) --> (N,)
                            z = np.tanh(a)                                          # (N,)
                            u = np.dot(z, self.W2[target])                          # (N,)(N,) --> ()
                            y = np.exp(u) / np.sum(np.exp(np.dot(z, self.W2.T)))    # p(w_j|w_i) --> ()

                            # backpropagation
                            e = label - y                               # ()
                            self.W2[context_token] += eta * e * z       # (N,)
                            dz = e * self.W2[context_token]             # (N,)
                            da = dz * (1 - np.tanh(a)**2)               # (N,)
                            dD = np.outer(da, self.W1[center_token])    # (N,N)
                            self.D += eta * dD                          # (N,N)
                            g_w += eta * da                             # (N,)

                        self.W1[center_token] += g_w                    # (N,) update W1

    def most_similar(self, word, top_n=10):
        """
        Return most similar words with input word
        using cosine similarity between input word and other words.

        Args:
            word: query word
            top_n: top n most similar words

        Return:
            sorted (word, similarity) pair list (list of tuple) given top_n
        """
        tokens = self.vocab.keys()                                      # words (string)
        indices = list(zip(*self.vocab.values()))[0]                    # word indices
        word2idx = {token: idx for idx, token in zip(indices, tokens)}  # mapping word on index
        idx2word = {idx: token for idx, token in zip(indices, tokens)}  # mapping index on word

        word_index = word2idx[word]
        v_w1 = self.W1[word_index]

        word_sim = {}
        for i in range(len(self.vocab)):
            v_w2 = self.W1[i]
            dot_product = np.dot(v_w1, v_w2)
            norms = np.dot(np.linalg.norm(v_w1), np.linalg.norm(v_w2))
            sim = dot_product / norms

            if i != word_index:
                word = idx2word[i]
                word_sim[word] = sim

        word_sim_sorted = sorted(word_sim.items(), key=lambda x: x[1], reverse=True)

        return word_sim_sorted[:top_n]
