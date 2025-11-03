"""
Hangman Game Environment (Gym-like interface).

This environment implements the Hangman game for reinforcement learning.
It follows the Gym interface conventions for compatibility with RL algorithms.

State: Includes masked word, guessed letters, remaining lives, and HMM probabilities
Actions: Guess any unguessed letter (0-25 for a-z)
Rewards:
    +10 for correct guess
    -5 for wrong guess
    -2 for repeated guess
    +50 for winning the word
    -50 for losing (running out of lives)
"""

import numpy as np
from typing import Tuple, Dict, Set
from utils import (
    ALPHABET, LETTER_TO_IDX, decode_letter, create_masked_word,
    encode_masked_word, encode_guessed_letters
)


class HangmanEnv:
    """
    Hangman environment for reinforcement learning.
    """
    
    def __init__(self, word: str, hmm_model=None, max_lives: int = 6):
        """
        Initialize the Hangman environment.
        
        Args:
            word: The target word to guess
            hmm_model: Trained HMM model for probability predictions
            max_lives: Maximum number of wrong guesses allowed
        """
        self.target_word = word.lower()
        self.hmm_model = hmm_model
        self.max_lives = max_lives
        
        # Game state
        self.guessed_letters: Set[str] = set()
        self.remaining_lives = max_lives
        self.masked_word = "_" * len(self.target_word)
        self.done = False
        self.won = False
        
        # Statistics
        self.total_guesses = 0
        self.wrong_guesses = 0
        self.repeated_guesses = 0
        
    def reset(self, word: str = None) -> Dict:
        """
        Reset the environment for a new game.
        
        Args:
            word: Optional new word (if None, keeps current word)
            
        Returns:
            Initial state dictionary
        """
        if word is not None:
            self.target_word = word.lower()
        
        self.guessed_letters = set()
        self.remaining_lives = self.max_lives
        self.masked_word = "_" * len(self.target_word)
        self.done = False
        self.won = False
        
        self.total_guesses = 0
        self.wrong_guesses = 0
        self.repeated_guesses = 0
        
        return self._get_state()
    
    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """
        Take a step in the environment by guessing a letter.
        
        Args:
            action: Letter index (0-25 for a-z)
            
        Returns:
            Tuple of (state, reward, done, info)
        """
        if self.done:
            # Game already finished
            return self._get_state(), 0, True, self._get_info()
        
        # Convert action to letter
        letter = decode_letter(action)
        
        # Initialize reward
        reward = 0
        
        # Check if letter was already guessed (repeated guess)
        if letter in self.guessed_letters:
            reward = -2  # Penalty for repeated guess
            self.repeated_guesses += 1
            self.total_guesses += 1
            return self._get_state(), reward, self.done, self._get_info()
        
        # Add letter to guessed set
        self.guessed_letters.add(letter)
        self.total_guesses += 1
        
        # Check if letter is in the word
        if letter in self.target_word:
            # Correct guess
            reward = 20
            self.masked_word = create_masked_word(self.target_word, self.guessed_letters)
            
            # Check if word is complete (won)
            if '_' not in self.masked_word:
                reward += 100  # Bonus for winning
                self.done = True
                self.won = True
        else:
            # Wrong guess
            reward = -10
            self.remaining_lives -= 1
            self.wrong_guesses += 1
            
            # Check if out of lives (lost)
            if self.remaining_lives <= 0:
                reward -= 100  # Penalty for losing
                self.done = True
                self.won = False
        
        return self._get_state(), reward, self.done, self._get_info()
    
    def _get_state(self) -> Dict:
        """
        Get the current state of the environment.
        
        Returns:
            Dictionary containing:
                - masked_word: Current masked word string
                - guessed_letters: Set of guessed letters
                - remaining_lives: Number of lives left
                - hmm_probs: HMM probability distribution (if HMM available)
                - encoded_masked_word: Numerical encoding of masked word
                - encoded_guessed: Binary encoding of guessed letters
        """
        state = {
            'masked_word': self.masked_word,
            'guessed_letters': self.guessed_letters.copy(),
            'remaining_lives': self.remaining_lives,
        }
        
        # Get HMM probabilities if model available
        if self.hmm_model is not None:
            hmm_probs = self.hmm_model.hmm_predict(self.masked_word, self.guessed_letters)
        else:
            # Default to uniform distribution over unguessed letters
            hmm_probs = np.ones(26)
            for letter in self.guessed_letters:
                if letter in LETTER_TO_IDX:
                    hmm_probs[LETTER_TO_IDX[letter]] = 0
            prob_sum = hmm_probs.sum()
            if prob_sum > 0:
                hmm_probs = hmm_probs / prob_sum
            else:
                hmm_probs = np.ones(26) / 26
        
        state['hmm_probs'] = hmm_probs
        
        # Encoded representations for neural networks
        state['encoded_masked_word'] = encode_masked_word(self.masked_word)
        state['encoded_guessed'] = encode_guessed_letters(self.guessed_letters)
        
        return state
    
    def _get_info(self) -> Dict:
        """
        Get additional information about the current state.
        
        Returns:
            Dictionary with game statistics
        """
        return {
            'total_guesses': self.total_guesses,
            'wrong_guesses': self.wrong_guesses,
            'repeated_guesses': self.repeated_guesses,
            'target_word': self.target_word,
            'won': self.won,
            'remaining_lives': self.remaining_lives,
        }
    
    def render(self):
        """Print the current state of the game."""
        print(f"Word: {self.masked_word}")
        print(f"Guessed: {sorted(list(self.guessed_letters))}")
        print(f"Lives: {self.remaining_lives}/{self.max_lives}")
        print(f"Guesses: {self.total_guesses} (Wrong: {self.wrong_guesses}, Repeated: {self.repeated_guesses})")
    
    def get_valid_actions(self) -> np.ndarray:
        """
        Get mask of valid actions (unguessed letters).
        
        Returns:
            Binary array where 1 indicates valid action, 0 invalid
        """
        valid = np.ones(26)
        for letter in self.guessed_letters:
            if letter in LETTER_TO_IDX:
                valid[LETTER_TO_IDX[letter]] = 0
        return valid


def create_state_vector(state: Dict, max_word_length: int = 20) -> np.ndarray:
    """
    Create a flat state vector from the state dictionary for neural networks.
    
    Args:
        state: State dictionary from environment
        max_word_length: Maximum word length for padding
        
    Returns:
        Flattened numpy array suitable for neural network input
    """
    # Components:
    # 1. Encoded masked word: max_word_length * 27
    # 2. Guessed letters: 26
    # 3. Remaining lives: 1
    # 4. HMM probabilities: 26
    
    encoded_word = state['encoded_masked_word'].flatten()  # max_word_length * 27
    encoded_guessed = state['encoded_guessed']  # 26
    lives = np.array([state['remaining_lives'] / 6.0])  # Normalize to [0, 1]
    hmm_probs = state['hmm_probs']  # 26
    
    # Concatenate all components
    state_vector = np.concatenate([encoded_word, encoded_guessed, lives, hmm_probs])
    
    return state_vector.astype(np.float32)