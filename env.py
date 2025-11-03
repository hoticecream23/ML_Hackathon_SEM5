import numpy as np
import string

class HangmanEnv:
    def __init__(self, words, max_wrong=6):
        self.words = words
        self.max_wrong = max_wrong
        self.reset()

    def reset(self):
        self.word = np.random.choice(self.words)
        self.masked = ["_"] * len(self.word)
        self.guessed = set()
        self.wrong = 0
        return self._get_state()

    def _get_state(self):
        return "".join(self.masked), self.guessed, self.wrong

    def step(self, guess):
        if guess in self.guessed:
            reward = -1.0         # repeat penalty
        else:
            self.guessed.add(guess)
            if guess in self.word:
        # small positive per correct letter (encourages revealing faster)
                hits = sum(1 for i,ch in enumerate(self.word) if ch == guess)
                for i,ch in enumerate(self.word):
                    if ch == guess:
                        self.masked[i] = ch
                reward = 1.0 * hits
            else:
                self.wrong += 1
                reward = -1.5       # wrong guess penalty
        # terminal rewards
            if "_" not in self.masked:
                reward += 10.0         # big win reward
                done = True
            elif self.wrong >= self.max_wrong:
                reward -= 5.0          # losing penalty
                done = True