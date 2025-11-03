"""
Evaluation script for the Hangman agent.

Runs 2000 games with the trained agent and computes:
- Success Rate
- Total Wrong Guesses
- Total Repeated Guesses
- Final Score = (Success Rate * 2000) - (Total Wrong Guesses * 5) - (Total Repeated Guesses * 2)
"""

import os
import sys
import json
import numpy as np
from tqdm import tqdm

from utils import load_corpus, split_corpus
from hmm_model import HangmanHMM
from hangman_env import HangmanEnv, create_state_vector
from rl_agent import DQNAgent


def evaluate_agent(agent, hmm_model, test_words, num_games=2000, max_lives=6, verbose=False):
    """
    Evaluate the agent on a set of test words.
    
    Args:
        agent: Trained DQN agent
        hmm_model: Trained HMM model
        test_words: List of test words
        num_games: Number of games to play
        max_lives: Maximum wrong guesses allowed
        verbose: Whether to print detailed game information
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'=' * 60}")
    print(f"Evaluating Agent on {num_games} Games")
    print(f"{'=' * 60}\n")
    
    # Sample games from test words
    if len(test_words) < num_games:
        # If not enough test words, sample with replacement
        game_words = np.random.choice(test_words, size=num_games, replace=True)
    else:
        game_words = np.random.choice(test_words, size=num_games, replace=False)
    
    # Statistics
    total_games = 0
    total_wins = 0
    total_wrong_guesses = 0
    total_repeated_guesses = 0
    total_guesses = 0
    
    game_results = []
    
    # Play games
    for i, word in enumerate(tqdm(game_words, desc="Playing games")):
        env = HangmanEnv(word, hmm_model, max_lives=max_lives)
        state = env.reset()
        
        done = False
        steps = 0
        max_steps = 50
        
        while not done and steps < max_steps:
            # Get state and select action
            state_vector = create_state_vector(state)
            valid_actions = env.get_valid_actions()
            
            # Use agent without exploration (greedy)
            action = agent.select_action(state_vector, valid_actions, training=False)
            
            # Take step
            state, reward, done, info = env.step(action)
            steps += 1
        
        # Record results
        total_games += 1
        if info['won']:
            total_wins += 1
        
        total_wrong_guesses += info['wrong_guesses']
        total_repeated_guesses += info['repeated_guesses']
        total_guesses += info['total_guesses']
        
        game_results.append({
            'word': word,
            'won': info['won'],
            'wrong_guesses': info['wrong_guesses'],
            'repeated_guesses': info['repeated_guesses'],
            'total_guesses': info['total_guesses']
        })
        
        if verbose and (i + 1) % 200 == 0:
            print(f"\nAfter {i + 1} games:")
            print(f"  Success Rate: {total_wins / total_games:.2%}")
            print(f"  Avg Wrong Guesses: {total_wrong_guesses / total_games:.2f}")
            print(f"  Avg Repeated Guesses: {total_repeated_guesses / total_games:.2f}")
    
    # Calculate final metrics
    success_rate = total_wins / total_games
    avg_wrong_guesses = total_wrong_guesses / total_games
    avg_repeated_guesses = total_repeated_guesses / total_games
    avg_total_guesses = total_guesses / total_games
    
    # Calculate final score
    final_score = (success_rate * num_games) - (total_wrong_guesses * 5) - (total_repeated_guesses * 2)
    
    results = {
        'num_games': num_games,
        'total_wins': total_wins,
        'success_rate': success_rate,
        'total_wrong_guesses': total_wrong_guesses,
        'avg_wrong_guesses': avg_wrong_guesses,
        'total_repeated_guesses': total_repeated_guesses,
        'avg_repeated_guesses': avg_repeated_guesses,
        'avg_total_guesses': avg_total_guesses,
        'final_score': final_score,
        'game_results': game_results
    }
    
    return results


def print_evaluation_summary(results):
    """
    Print evaluation summary.
    
    Args:
        results: Dictionary with evaluation results
    """
    print(f"\n{'=' * 60}")
    print("EVALUATION RESULTS")
    print(f"{'=' * 60}\n")
    
    print(f"Number of Games:        {results['num_games']}")
    print(f"Total Wins:             {results['total_wins']}")
    print(f"Success Rate:           {results['success_rate']:.2%}")
    print(f"\nTotal Wrong Guesses:    {results['total_wrong_guesses']}")
    print(f"Avg Wrong Guesses:      {results['avg_wrong_guesses']:.2f}")
    print(f"\nTotal Repeated Guesses: {results['total_repeated_guesses']}")
    print(f"Avg Repeated Guesses:   {results['avg_repeated_guesses']:.2f}")
    print(f"\nAvg Total Guesses:      {results['avg_total_guesses']:.2f}")
    
    print(f"\n{'=' * 60}")
    print(f"FINAL SCORE: {results['final_score']:.2f}")
    print(f"{'=' * 60}\n")
    
    print("Score Breakdown:")
    print(f"  Success Rate × 2000:         +{results['success_rate'] * results['num_games']:.2f}")
    print(f"  Total Wrong Guesses × 5:     -{results['total_wrong_guesses'] * 5:.2f}")
    print(f"  Total Repeated Guesses × 2:  -{results['total_repeated_guesses'] * 2:.2f}")
    print(f"  {'=' * 40}")
    print(f"  Final Score:                  {results['final_score']:.2f}")


def main():
    """Main evaluation function."""
    
    print("=" * 60)
    print("Hangman Agent Evaluation")
    print("=" * 60)
    
    # Load corpus
    try:
        words = load_corpus('data/corpus.txt')
    except (FileNotFoundError, ValueError) as e:
        print(f"\n{e}")
        sys.exit(1)
    
    # Split into train/test
    train_words, test_words = split_corpus(words, train_ratio=0.8, random_seed=42)
    print(f"\nTest set size: {len(test_words)} words")
    
    # Load HMM
    if not os.path.exists('results/hmm.pkl'):
        print("\nERROR: HMM model not found at results/hmm.pkl")
        print("Please run train_hmm.py first!")
        sys.exit(1)
    
    hmm_model = HangmanHMM.load('results/hmm.pkl')
    
    # Load DQN agent
    model_path = 'results/dqn_best.pth' if os.path.exists('results/dqn_best.pth') else 'results/dqn.pth'
    
    if not os.path.exists(model_path):
        print(f"\nERROR: DQN model not found at {model_path}")
        print("Please run train_rl.py first!")
        sys.exit(1)
    
    print(f"\nLoading model from: {model_path}")
    
    # Initialize agent
    sample_env = HangmanEnv(test_words[0], hmm_model)
    sample_state = sample_env.reset()
    state_size = len(create_state_vector(sample_state))
    
    agent = DQNAgent(state_size=state_size, action_size=26)
    agent.load(model_path)
    
    # Evaluate
    results = evaluate_agent(
        agent=agent,
        hmm_model=hmm_model,
        test_words=test_words,
        num_games=2000,
        max_lives=6,
        verbose=True
    )
    
    # Print summary
    print_evaluation_summary(results)
    
    # Save detailed results
    os.makedirs('results', exist_ok=True)
    
    # Save summary (without game-by-game details for size)
    summary = {k: v for k, v in results.items() if k != 'game_results'}
    
    with open('results/evaluation_results.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"Evaluation results saved to: results/evaluation_results.json")
    
    # Create a simple text report
    with open('results/evaluation_summary.txt', 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("HANGMAN AGENT EVALUATION RESULTS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Number of Games:        {results['num_games']}\n")
        f.write(f"Total Wins:             {results['total_wins']}\n")
        f.write(f"Success Rate:           {results['success_rate']:.2%}\n\n")
        
        f.write(f"Total Wrong Guesses:    {results['total_wrong_guesses']}\n")
        f.write(f"Avg Wrong Guesses:      {results['avg_wrong_guesses']:.2f}\n\n")
        
        f.write(f"Total Repeated Guesses: {results['total_repeated_guesses']}\n")
        f.write(f"Avg Repeated Guesses:   {results['avg_repeated_guesses']:.2f}\n\n")
        
        f.write(f"Avg Total Guesses:      {results['avg_total_guesses']:.2f}\n\n")
        
        f.write("=" * 60 + "\n")
        f.write(f"FINAL SCORE: {results['final_score']:.2f}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Score Calculation:\n")
        f.write(f"  Success Rate × 2000:         +{results['success_rate'] * results['num_games']:.2f}\n")
        f.write(f"  Total Wrong Guesses × 5:     -{results['total_wrong_guesses'] * 5:.2f}\n")
        f.write(f"  Total Repeated Guesses × 2:  -{results['total_repeated_guesses'] * 2:.2f}\n")
        f.write(f"  " + "=" * 40 + "\n")
        f.write(f"  Final Score:                  {results['final_score']:.2f}\n")
    
    print(f"Evaluation summary saved to: results/evaluation_summary.txt")
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()