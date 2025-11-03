"""
Hidden Markov Model for Hangman letter prediction.

The HMM models word structure by treating:
- Hidden states: Position-based patterns in words (e.g., beginning, middle, end patterns)
- Emissions: Letters at each position

Design choice: We use a single HMM that models letter sequences in words.
The model learns transition probabilities between letter positions and 
emission probabilities for each letter at different positions.
"""

import numpy as np
import pickle
from typing import List, Set, Dict
from collections import defaultdict
from utils import ALPHABET, LETTER_TO_IDX, filter_words_by_pattern


class HangmanHMM:
    """
    Hidden Markov Model for predicting letters in Hangman.
    
    This HMM uses a simple but effective approach:
    - States represent positions in words (with length grouping)
    - Emissions are the letters at each position
    - We learn transition and emission probabilities from the corpus
    """
    
    def __init__(self, n_states: int = 15):
        """
        Initialize the HMM.
        
        Args:
            n_states: Number of hidden states (position patterns)
                     We use 10 states to capture different word patterns
        """
        self.n_states = n_states
        self.n_letters = 26
        
        # Model parameters
        self.start_prob = None  # Initial state distribution
        self.trans_prob = None  # Transition probabilities between states
        self.emit_prob = None   # Emission probabilities (state -> letter)
        
        # Length-specific models for better accuracy
        self.length_models = {}  # Dict mapping word length to letter frequencies
        
        # Overall letter frequencies as fallback
        self.letter_freq = np.ones(26) / 26
        
    def train(self, words: List[str]):
        """
        Train the HMM on a corpus of words.
        
        Args:
            words: List of words for training
        """
        print(f"Training HMM on {len(words)} words...")
        
        # Initialize parameters
        self.start_prob = np.ones(self.n_states) / self.n_states
        self.trans_prob = np.ones((self.n_states, self.n_states)) / self.n_states
        self.emit_prob = np.ones((self.n_states, self.n_letters)) * 0.01
        
        # Count letter frequencies for each state (position)
        state_letter_counts = defaultdict(lambda: defaultdict(int))
        state_counts = defaultdict(int)
        
        # Also build length-specific letter frequency models
        length_letter_counts = defaultdict(lambda: defaultdict(int))
        length_counts = defaultdict(int)
        
        for word in words:
            word_len = len(word)
            
            # Process each letter in the word
            for pos, letter in enumerate(word):
                if letter not in LETTER_TO_IDX:
                    continue
                
                letter_idx = LETTER_TO_IDX[letter]
                
                # Map position to state (divide word into n_states segments)
                state = int(pos * self.n_states / word_len)
                state = min(state, self.n_states - 1)
                
                # Update counts
                state_letter_counts[state][letter_idx] += 1
                state_counts[state] += 1
                
                # Update length-specific counts
                length_letter_counts[word_len][letter_idx] += 1
                length_counts[word_len] += 1
        
        # Compute emission probabilities from counts
        for state in range(self.n_states):
            if state_counts[state] > 0:
                for letter_idx in range(self.n_letters):
                    count = state_letter_counts[state][letter_idx]
                    self.emit_prob[state, letter_idx] = (count + 0.1) / (state_counts[state] + 2.6)
        
        # Normalize emission probabilities
        self.emit_prob = self.emit_prob / self.emit_prob.sum(axis=1, keepdims=True)
        
        # Build length-specific models
        for word_len in length_letter_counts:
            letter_probs = np.zeros(26)
            total = length_counts[word_len]
            for letter_idx in range(26):
                count = length_letter_counts[word_len][letter_idx]
                letter_probs[letter_idx] = (count + 0.1) / (total + 2.6)
            
            # Normalize
            letter_probs = letter_probs / letter_probs.sum()
            self.length_models[word_len] = letter_probs
        
        # Compute overall letter frequency
        total_letters = sum(len(word) for word in words)
        letter_counts = np.zeros(26)
        for word in words:
            for letter in word:
                if letter in LETTER_TO_IDX:
                    letter_counts[LETTER_TO_IDX[letter]] += 1
        
        self.letter_freq = (letter_counts + 0.1) / (total_letters + 2.6)
        self.letter_freq = self.letter_freq / self.letter_freq.sum()
        
        print(f"HMM training complete. Learned {len(self.length_models)} length-specific models.")
    
    def hmm_predict(self, masked_word: str, guessed_letters: Set[str]) -> np.ndarray:
        """
        Predict letter probabilities for the next guess.
        
        Args:
            masked_word: Current state of the word (e.g., "_a_e")
            guessed_letters: Set of already guessed letters
            
        Returns:
            Probability distribution over 26 letters
        """
        word_len = len(masked_word)
        
        # Start with length-specific model if available
        if word_len in self.length_models:
            probs = self.length_models[word_len].copy()
        else:
            # Fall back to overall letter frequency
            probs = self.letter_freq.copy()
        
        # Use position-based emission probabilities for known positions
        unknown_positions = [i for i, c in enumerate(masked_word) if c == '_']
        
        if len(unknown_positions) > 0:
            position_probs = np.zeros(26)
            
            for pos in unknown_positions:
                # Map position to state
                state = int(pos * self.n_states / word_len)
                state = min(state, self.n_states - 1)
                
                # Add emission probabilities for this state
                position_probs += self.emit_prob[state, :]
            
            # Average over all unknown positions
            position_probs = position_probs / len(unknown_positions)
            
            # Combine with length-specific model
            probs = 0.6 * probs + 0.4 * position_probs
        
        # Zero out already guessed letters
        for letter in guessed_letters:
            if letter in LETTER_TO_IDX:
                probs[LETTER_TO_IDX[letter]] = 0
        
        # Renormalize
        prob_sum = probs.sum()
        if prob_sum > 0:
            probs = probs / prob_sum
        else:
            # If all letters guessed, return uniform over remaining
            remaining = [i for i in range(26) if ALPHABET[i] not in guessed_letters]
            if remaining:
                probs = np.zeros(26)
                for i in remaining:
                    probs[i] = 1.0 / len(remaining)
            else:
                probs = np.ones(26) / 26
        
        return probs
    
    def save(self, filepath: str):
        """Save the trained HMM to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"HMM saved to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'HangmanHMM':
        """Load a trained HMM from a file."""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"HMM loaded from {filepath}")
        return model


def hmm_predict(hmm_model: HangmanHMM, masked_word: str, guessed_letters: Set[str]) -> np.ndarray:
    """
    Convenience function for HMM prediction.
    
    Args:
        hmm_model: Trained HMM model
        masked_word: Current masked word state
        guessed_letters: Set of guessed letters
        
    Returns:
        Probability distribution over 26 letters
    """
    return hmm_model.hmm_predict(masked_word, guessed_letters)