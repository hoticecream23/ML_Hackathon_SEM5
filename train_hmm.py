"""
Training script for the Hidden Markov Model.

This script:
1. Loads the corpus from data/corpus.txt
2. Trains the HMM on the corpus
3. Saves the trained model to results/hmm.pkl
4. Prints training summary
"""

import os
import sys
from hmm_model import HangmanHMM
from utils import load_corpus


def main():
    """Train the HMM and save it."""
    
    print("=" * 60)
    print("Hangman HMM Training")
    print("=" * 60)
    
    # Load corpus
    try:
        corpus_path = 'data/corpus.txt'
        words = load_corpus(corpus_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n{e}")
        sys.exit(1)
    
    # Print corpus statistics
    print(f"\nCorpus Statistics:")
    print(f"  Total words: {len(words)}")
    print(f"  Unique words: {len(set(words))}")
    
    word_lengths = [len(word) for word in words]
    print(f"  Word length range: {min(word_lengths)} - {max(word_lengths)}")
    print(f"  Average word length: {sum(word_lengths) / len(word_lengths):.2f}")
    
    # Count words by length
    from collections import Counter
    length_dist = Counter(word_lengths)
    print(f"  Most common lengths: {length_dist.most_common(5)}")
    
    # Initialize and train HMM
    print("\n" + "=" * 60)
    print("Training HMM...")
    print("=" * 60)
    
    # Use 10 hidden states to capture position patterns
    hmm = HangmanHMM(n_states=10)
    hmm.train(words)
    
    # Save the model
    os.makedirs('results', exist_ok=True)
    hmm.save('results/hmm.pkl')
    
    # Test the HMM with a few examples
    print("\n" + "=" * 60)
    print("Testing HMM predictions:")
    print("=" * 60)
    
    test_cases = [
        ("_____", set()),
        ("a____", {'a'}),
        ("_e___", {'e'}),
        ("__pp__", {'p'}),
    ]
    
    for masked_word, guessed in test_cases:
        probs = hmm.hmm_predict(masked_word, guessed)
        # Get top 5 predicted letters
        top_indices = probs.argsort()[-5:][::-1]
        top_letters = [chr(ord('a') + i) for i in top_indices]
        top_probs = [probs[i] for i in top_indices]
        
        print(f"\nMasked word: {masked_word}, Guessed: {guessed}")
        print(f"  Top predictions: {list(zip(top_letters, [f'{p:.3f}' for p in top_probs]))}")
    
    print("\n" + "=" * 60)
    print("HMM training complete!")
    print(f"Model saved to: results/hmm.pkl")
    print("=" * 60)


if __name__ == "__main__":
    main()