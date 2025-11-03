import random, string

def load_corpus(path):
    with open(path) as f:
        words = [w.strip().lower() for w in f if w.strip()]
    words = [w for w in words if all(c in string.ascii_lowercase for c in w)]
    return words

def split_train_test(words, test_size=2000, seed=42):
    random.seed(seed)
    random.shuffle(words)
    return words[test_size:], words[:test_size]
