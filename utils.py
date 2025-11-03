"""
Utility functions for the Hangman project.
Includes text preprocessing, letter encoding/decoding, and corpus loading.
"""

import os
import numpy as np
from typing import List, Set, Tuple


# Constants
ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
LETTER_TO_IDX = {letter: idx for idx, letter in enumerate(ALPHABET)}
IDX_TO_LETTER = {idx: letter for idx, letter in enumerate(ALPHABET)}


def load_corpus(corpus_path: str = 'data/corpus.txt') -> List[str]:
    """
    Load and preprocess the corpus file.
    
    Args:
        corpus_path: Path to the corpus file
        
    Returns:
        List of cleaned words (lowercase, alphabetic only)
        
    Raises:
        FileNotFoundError: If corpus file doesn't exist
        ValueError: If corpus file is empty
    """
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(
            f"ERROR: {corpus_path} not found or empty — please provide corpus.txt"
        )
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        words = f.read().splitlines()
    
    # Clean words: lowercase, remove non-alphabetic characters
    cleaned_words = []
    for word in words:
        word = word.strip().lower()
        # Only keep words with alphabetic characters
        if word.isalpha() and len(word) > 0:
            cleaned_words.append(word)
    
    if len(cleaned_words) == 0:
        raise ValueError(
            f"ERROR: {corpus_path} is empty or contains no valid words"
        )
    
    print(f"Loaded {len(cleaned_words)} words from {corpus_path}")
    return cleaned_words


def preprocess_word(word: str) -> str:
    """Convert word to lowercase and ensure it's alphabetic."""
    return word.lower().strip()


def create_masked_word(word: str, guessed_letters: Set[str]) -> str:
    """
    Create a masked version of the word with underscores for unguessed letters.
    
    Args:
        word: The target word
        guessed_letters: Set of letters that have been guessed
        
    Returns:
        Masked word string (e.g., "a__le" for "apple" with guesses {a, l})
    """
    masked = ""
    for letter in word:
        if letter in guessed_letters:
            masked += letter
        else:
            masked += "_"
    return masked


def encode_letter(letter: str) -> int:
    """Convert a letter to its index (0-25)."""
    return LETTER_TO_IDX.get(letter.lower(), -1)


def decode_letter(idx: int) -> str:
    """Convert an index (0-25) to its letter."""
    return IDX_TO_LETTER.get(idx, '?')


def encode_masked_word(masked_word: str, max_length: int = 20) -> np.ndarray:
    """
    Encode a masked word as a numerical vector.
    Uses 27 dimensions per position: 26 for letters + 1 for unknown (_).
    
    Args:
        masked_word: String with underscores for unknown letters
        max_length: Maximum word length for padding
        
    Returns:
        One-hot encoded array of shape (max_length, 27)
    """
    encoding = np.zeros((max_length, 27))
    
    for i, char in enumerate(masked_word[:max_length]):
        if char == '_':
            encoding[i, 26] = 1  # Unknown character
        elif char in LETTER_TO_IDX:
            encoding[i, LETTER_TO_IDX[char]] = 1
    
    return encoding


def encode_guessed_letters(guessed_letters: Set[str]) -> np.ndarray:
    """
    Encode the set of guessed letters as a binary vector.
    
    Args:
        guessed_letters: Set of letters that have been guessed
        
    Returns:
        Binary array of length 26 (1 if guessed, 0 otherwise)
    """
    encoding = np.zeros(26)
    for letter in guessed_letters:
        if letter in LETTER_TO_IDX:
            encoding[LETTER_TO_IDX[letter]] = 1
    return encoding


def get_available_letters(guessed_letters: Set[str]) -> List[str]:
    """
    Get list of letters that haven't been guessed yet.
    
    Args:
        guessed_letters: Set of letters already guessed
        
    Returns:
        List of available letters
    """
    return [letter for letter in ALPHABET if letter not in guessed_letters]


def get_word_pattern(word: str, guessed_letters: Set[str]) -> str:
    """
    Get the pattern of a word based on guessed letters.
    Similar to create_masked_word but ensures consistency.
    
    Args:
        word: The target word
        guessed_letters: Set of guessed letters
        
    Returns:
        Pattern string with revealed and hidden letters
    """
    return create_masked_word(word, guessed_letters)


def filter_words_by_pattern(words: List[str], pattern: str, guessed_letters: Set[str]) -> List[str]:
    """
    Filter words that match a given pattern.
    
    Args:
        words: List of words to filter
        pattern: Pattern with underscores for unknown letters
        guessed_letters: Letters that have been guessed
        
    Returns:
        List of words matching the pattern
    """
    if len(pattern) == 0:
        return []
    
    matching_words = []
    pattern_length = len(pattern)
    
    for word in words:
        if len(word) != pattern_length:
            continue
        
        # Check if word matches pattern
        matches = True
        for i, char in enumerate(pattern):
            if char == '_':
                # This position is unknown, word letter must not be in guessed_letters
                if word[i] in guessed_letters:
                    matches = False
                    break
            else:
                # This position is known, must match exactly
                if word[i] != char:
                    matches = False
                    break
        
        if matches:
            matching_words.append(word)
    
    return matching_words


def split_corpus(words: List[str], train_ratio: float = 0.8, random_seed: int = 42) -> Tuple[List[str], List[str]]:
    """
    Split corpus into training and testing sets.
    
    Args:
        words: List of words
        train_ratio: Ratio of training data
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_words, test_words)
    """
    np.random.seed(random_seed)
    indices = np.random.permutation(len(words))
    split_idx = int(len(words) * train_ratio)
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    train_words = [words[i] for i in train_indices]
    test_words = [words[i] for i in test_indices]
    
    return train_words, test_words


def calculate_letter_frequency(words: List[str]) -> np.ndarray:
    """
    Calculate letter frequency distribution across all words.
    
    Args:
        words: List of words
        
    Returns:
        Probability distribution over 26 letters
    """
    letter_counts = np.zeros(26)
    
    for word in words:
        for letter in word:
            if letter in LETTER_TO_IDX:
                letter_counts[LETTER_TO_IDX[letter]] += 1
    
    # Normalize to get probabilities
    total = letter_counts.sum()
    if total > 0:
        return letter_counts / total
    else:
        return np.ones(26) / 26  # Uniform distribution if no data