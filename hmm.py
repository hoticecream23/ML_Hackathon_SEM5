import numpy as np
import string
from collections import Counter

ALPH = string.ascii_lowercase
A2I = {a: i for i, a in enumerate(ALPH)}

class SimpleHMM:
    def __init__(self):
        self.transition = np.zeros((26, 26)) + 1e-8
        self.unigram = np.zeros(26) + 1e-8
        self.length_buckets = {}

    def train(self, words):
        for w in words:
            letters = list(w)
            for i, ch in enumerate(letters):
                self.unigram[A2I[ch]] += 1
                if i < len(letters) - 1:
                    self.transition[A2I[ch], A2I[letters[i + 1]]] += 1
            L = len(w)
            if L not in self.length_buckets:
                self.length_buckets[L] = Counter()
            for i, ch in enumerate(letters):
                self.length_buckets[L][(i, ch)] += 1

        self.unigram /= self.unigram.sum()
        self.transition /= self.transition.sum(axis=1, keepdims=True)
        self.emissions = {}
        for L, cnt in self.length_buckets.items():
            pos_dist = {}
            for (pos, ch), v in cnt.items():
                pos_dist.setdefault(pos, np.zeros(26))
                pos_dist[pos][A2I[ch]] += v
            for pos in pos_dist:
                pos_dist[pos] /= pos_dist[pos].sum()
            self.emissions[L] = pos_dist

    def letter_probs(self, masked_word, guessed_set):
        L = len(masked_word)
        probs = np.copy(self.unigram)
        if L in self.emissions:
            pos_dist = np.zeros(26)
            count = 0
            for i, ch in enumerate(masked_word):
                if ch == "_":
                    if i in self.emissions[L]:
                        pos_dist += self.emissions[L][i]
                    count += 1
            if count > 0:
                pos_dist /= count
                probs = 0.6 * probs + 0.4 * pos_dist
        for g in guessed_set:
            probs[A2I[g]] = 0
        probs /= probs.sum()
        return probs
