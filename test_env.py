"""
Unit tests for the Hangman environment.

Tests basic functionality including:
- Masking
- Wrong guess handling
- Repeated guess detection
- Game termination conditions
- Reward calculation
"""

import unittest
import numpy as np
from hangman_env import HangmanEnv
from utils import LETTER_TO_IDX


class TestHangmanEnv(unittest.TestCase):
    """Test cases for HangmanEnv."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_word = "apple"
        self.env = HangmanEnv(self.test_word, hmm_model=None, max_lives=6)
    
    def test_initialization(self):
        """Test environment initialization."""
        self.assertEqual(self.env.target_word, "apple")
        self.assertEqual(self.env.max_lives, 6)
        self.assertEqual(self.env.remaining_lives, 6)
        self.assertEqual(self.env.masked_word, "_____")
        self.assertFalse(self.env.done)
        self.assertFalse(self.env.won)
        self.assertEqual(len(self.env.guessed_letters), 0)
    
    def test_reset(self):
        """Test environment reset."""
        # Make some guesses
        self.env.step(LETTER_TO_IDX['a'])
        self.env.step(LETTER_TO_IDX['b'])
        
        # Reset
        state = self.env.reset()
        
        self.assertEqual(self.env.remaining_lives, 6)
        self.assertEqual(self.env.masked_word, "_____")
        self.assertEqual(len(self.env.guessed_letters), 0)
        self.assertFalse(self.env.done)
    
    def test_correct_guess(self):
        """Test correct letter guess."""
        # Guess 'a' (correct)
        state, reward, done, info = self.env.step(LETTER_TO_IDX['a'])
        
        self.assertEqual(reward, 10)  # Correct guess reward
        self.assertIn('a', self.env.guessed_letters)
        self.assertEqual(self.env.masked_word, "a____")
        self.assertEqual(self.env.remaining_lives, 6)  # No life lost
        self.assertFalse(done)
    
    def test_wrong_guess(self):
        """Test wrong letter guess."""
        # Guess 'z' (wrong)
        state, reward, done, info = self.env.step(LETTER_TO_IDX['z'])
        
        self.assertEqual(reward, -5)  # Wrong guess penalty
        self.assertIn('z', self.env.guessed_letters)
        self.assertEqual(self.env.masked_word, "_____")
        self.assertEqual(self.env.remaining_lives, 5)  # One life lost
        self.assertEqual(self.env.wrong_guesses, 1)
        self.assertFalse(done)
    
    def test_repeated_guess(self):
        """Test repeated letter guess."""
        # Guess 'a' first time
        self.env.step(LETTER_TO_IDX['a'])
        
        # Guess 'a' again
        state, reward, done, info = self.env.step(LETTER_TO_IDX['a'])
        
        self.assertEqual(reward, -2)  # Repeated guess penalty
        self.assertEqual(self.env.repeated_guesses, 1)
        self.assertEqual(self.env.remaining_lives, 6)  # No life lost
    
    def test_winning_game(self):
        """Test winning the game."""
        # Guess all correct letters: a, p, l, e
        self.env.step(LETTER_TO_IDX['a'])
        self.env.step(LETTER_TO_IDX['p'])
        self.env.step(LETTER_TO_IDX['l'])
        
        # Last correct letter should win
        state, reward, done, info = self.env.step(LETTER_TO_IDX['e'])
        
        self.assertEqual(self.env.masked_word, "apple")
        self.assertTrue(done)
        self.assertTrue(self.env.won)
        self.assertEqual(reward, 60)  # 10 (correct) + 50 (win bonus)
    
    def test_losing_game(self):
        """Test losing the game (running out of lives)."""
        # Make 6 wrong guesses
        wrong_letters = ['z', 'x', 'q', 'w', 'k', 'j']
        
        for i, letter in enumerate(wrong_letters[:-1]):
            state, reward, done, info = self.env.step(LETTER_TO_IDX[letter])
            self.assertFalse(done)
            self.assertEqual(self.env.remaining_lives, 6 - i - 1)
        
        # Last wrong guess should lose
        state, reward, done, info = self.env.step(LETTER_TO_IDX[wrong_letters[-1]])
        
        self.assertTrue(done)
        self.assertFalse(self.env.won)
        self.assertEqual(self.env.remaining_lives, 0)
        self.assertEqual(reward, -55)  # -5 (wrong) + -50 (loss penalty)
    
    def test_masked_word_updates(self):
        """Test that masked word updates correctly."""
        # Initially all masked
        self.assertEqual(self.env.masked_word, "_____")
        
        # Guess 'p' (appears twice)
        self.env.step(LETTER_TO_IDX['p'])
        self.assertEqual(self.env.masked_word, "_pp__")
        
        # Guess 'a'
        self.env.step(LETTER_TO_IDX['a'])
        self.assertEqual(self.env.masked_word, "app__")
        
        # Guess 'l'
        self.env.step(LETTER_TO_IDX['l'])
        self.assertEqual(self.env.masked_word, "appl_")
    
    def test_valid_actions(self):
        """Test valid actions mask."""
        # Initially all actions valid
        valid = self.env.get_valid_actions()
        self.assertEqual(valid.sum(), 26)
        
        # Guess 'a'
        self.env.step(LETTER_TO_IDX['a'])
        valid = self.env.get_valid_actions()
        self.assertEqual(valid.sum(), 25)  # One less valid action
        self.assertEqual(valid[LETTER_TO_IDX['a']], 0)  # 'a' not valid
        
        # Guess 'b'
        self.env.step(LETTER_TO_IDX['b'])
        valid = self.env.get_valid_actions()
        self.assertEqual(valid.sum(), 24)  # Two less valid actions
    
    def test_info_dict(self):
        """Test info dictionary contents."""
        self.env.step(LETTER_TO_IDX['a'])
        self.env.step(LETTER_TO_IDX['z'])
        
        state, reward, done, info = self.env.step(LETTER_TO_IDX['a'])  # Repeated
        
        self.assertIn('total_guesses', info)
        self.assertIn('wrong_guesses', info)
        self.assertIn('repeated_guesses', info)
        self.assertIn('target_word', info)
        self.assertIn('won', info)
        
        self.assertEqual(info['total_guesses'], 3)
        self.assertEqual(info['wrong_guesses'], 1)
        self.assertEqual(info['repeated_guesses'], 1)
    
    def test_state_dict(self):
        """Test state dictionary structure."""
        state = self.env._get_state()
        
        self.assertIn('masked_word', state)
        self.assertIn('guessed_letters', state)
        self.assertIn('remaining_lives', state)
        self.assertIn('hmm_probs', state)
        self.assertIn('encoded_masked_word', state)
        self.assertIn('encoded_guessed', state)
        
        # Check shapes
        self.assertEqual(len(state['hmm_probs']), 26)
        self.assertEqual(state['encoded_guessed'].shape, (26,))
        self.assertEqual(state['encoded_masked_word'].shape, (20, 27))


class TestEnvironmentEdgeCases(unittest.TestCase):
    """Test edge cases in the environment."""
    
    def test_single_letter_word(self):
        """Test with single letter word."""
        env = HangmanEnv("a", hmm_model=None)
        state, reward, done, info = env.step(LETTER_TO_IDX['a'])
        
        self.assertTrue(done)
        self.assertTrue(env.won)
        self.assertEqual(env.masked_word, "a")
    
    def test_no_repeated_letters(self):
        """Test word with no repeated letters."""
        env = HangmanEnv("abcd", hmm_model=None)
        
        # Guess all letters
        for letter in "abcd":
            state, reward, done, info = env.step(LETTER_TO_IDX[letter])
        
        self.assertTrue(done)
        self.assertTrue(env.won)
        self.assertEqual(env.masked_word, "abcd")
    
    def test_all_same_letter(self):
        """Test word with all same letters."""
        env = HangmanEnv("aaaa", hmm_model=None)
        state, reward, done, info = env.step(LETTER_TO_IDX['a'])
        
        self.assertTrue(done)
        self.assertTrue(env.won)
        self.assertEqual(env.masked_word, "aaaa")


def run_tests():
    """Run all tests."""
    print("Running Hangman Environment Tests...\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestHangmanEnv))
    suite.addTests(loader.loadTestsFromTestCase(TestEnvironmentEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)