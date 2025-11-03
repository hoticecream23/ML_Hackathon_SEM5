"""
Training script for the DQN Hangman agent.

This script:
1. Loads the trained HMM from results/hmm.pkl
2. Implements supervised pretraining using HMM outputs
3. Trains the DQN agent on corpus words
4. Saves checkpoints and training plots
5. Saves final model to results/dqn.pth
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import load_corpus, split_corpus
from hmm_model import HangmanHMM
from hangman_env import HangmanEnv, create_state_vector
from rl_agent import DQNAgent


def collect_pretraining_data(hmm_model, words, num_samples=5000):
    """
    Collect state-action pairs from HMM for supervised pretraining.
    
    Args:
        hmm_model: Trained HMM
        words: List of words
        num_samples: Number of samples to collect
        
    Returns:
        Tuple of (states, hmm_probs)
    """
    print(f"Collecting {num_samples} pretraining samples...")
    
    states = []
    hmm_probs_list = []
    
    sample_words = np.random.choice(words, size=min(num_samples, len(words)), replace=False)
    
    for word in tqdm(sample_words, desc="Collecting pretraining data"):
        env = HangmanEnv(word, hmm_model)
        state = env.reset()
        
        # Collect state and HMM prediction
        state_vector = create_state_vector(state)
        hmm_probs = state['hmm_probs']
        
        states.append(state_vector)
        hmm_probs_list.append(hmm_probs)
    
    return states, hmm_probs_list


def train_dqn(
    hmm_model,
    train_words,
    num_episodes=6000,
    max_steps_per_episode=50,
    pretrain_epochs=5,
    save_freq=200,
    device='cpu'
):
    """
    Train the DQN agent.
    
    Args:
        hmm_model: Trained HMM model
        train_words: List of training words
        num_episodes: Number of training episodes
        max_steps_per_episode: Maximum steps per episode
        pretrain_epochs: Number of supervised pretraining epochs
        save_freq: Frequency of saving checkpoints
        device: Device to use for training
        
    Returns:
        Trained agent
    """
    print("\n" + "=" * 60)
    print("Training DQN Agent")
    print("=" * 60)
    
    # Initialize environment and agent
    sample_env = HangmanEnv(train_words[0], hmm_model)
    sample_state = sample_env.reset()
    state_size = len(create_state_vector(sample_state))
    
    print(f"State size: {state_size}")
    print(f"Action size: 26")
    print(f"Training words: {len(train_words)}")
    
    agent = DQNAgent(
        state_size=state_size,
        action_size=26,
        learning_rate=0.0005,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        buffer_capacity=10000,
        batch_size=64,
        target_update_freq=10,
        device=device
    )
    
    # Supervised pretraining
    if pretrain_epochs > 0:
        print("\n" + "=" * 60)
        print("Supervised Pretraining")
        print("=" * 60)
        
        pretrain_states, pretrain_hmm_probs = collect_pretraining_data(
            hmm_model, train_words, num_samples=min(3000, len(train_words))
        )
        
        agent.supervised_pretrain(
            pretrain_states, pretrain_hmm_probs, 
            epochs=pretrain_epochs, lr=0.001
        )
    
    # Training loop
    print("\n" + "=" * 60)
    print("Reinforcement Learning Training")
    print("=" * 60)
    
    episode_rewards = []
    episode_lengths = []
    episode_success = []
    losses = []
    
    best_avg_reward = -float('inf')
    
    for episode in tqdm(range(num_episodes), desc="Training"):
        # Sample a random word
        word = np.random.choice(train_words)
        env = HangmanEnv(word, hmm_model)
        state = env.reset()
        
        episode_reward = 0
        steps = 0
        
        for step in range(max_steps_per_episode):
            # Get state vector
            state_vector = create_state_vector(state)
            valid_actions = env.get_valid_actions()
            
            # Select action
            action = agent.select_action(state_vector, valid_actions, training=True)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            # Store experience
            next_state_vector = create_state_vector(next_state)
            agent.store_experience(state_vector, action, reward, next_state_vector, done)
            
            # Train
            loss = agent.train_step()
            if loss > 0:
                losses.append(loss)
            
            state = next_state
            
            if done:
                break
        
        # Update epsilon
        agent.update_epsilon()
        
        # Record episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        episode_success.append(1 if info['won'] else 0)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            success_rate = np.mean(episode_success[-100:])
            
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Length: {avg_length:.2f}")
            print(f"  Success Rate: {success_rate:.2%}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            
            # Save best model
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save('results/dqn_best.pth')
                print(f"  New best model saved!")
        
        # Save checkpoint
        if (episode + 1) % save_freq == 0:
            agent.save(f'results/dqn_checkpoint_{episode + 1}.pth')
    
    # Save final model
    agent.save('results/dqn.pth')
    
    # Save training statistics
    stats = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_success': episode_success,
        'losses': losses
    }
    
    return agent, stats


def plot_training_results(stats, save_path='results/'):
    """
    Plot training results.
    
    Args:
        stats: Dictionary with training statistics
        save_path: Path to save plots
    """
    print("\nGenerating training plots...")
    
    episode_rewards = stats['episode_rewards']
    episode_success = stats['episode_success']
    losses = stats['losses']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Episode Rewards
    axes[0, 0].plot(episode_rewards, alpha=0.3, label='Episode Reward')
    # Moving average
    window = 50
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(episode_rewards)), moving_avg, 
                       label=f'{window}-Episode Moving Avg', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Episode Rewards over Training')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Success Rate
    window = 50
    if len(episode_success) >= window:
        success_rate = np.convolve(episode_success, np.ones(window)/window, mode='valid')
        axes[0, 1].plot(range(window-1, len(episode_success)), success_rate, linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Success Rate')
    axes[0, 1].set_title(f'Success Rate ({window}-Episode Moving Average)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    # Plot 3: Training Loss
    if len(losses) > 0:
        axes[1, 0].plot(losses, alpha=0.3)
        window = 100
        if len(losses) >= window:
            moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
            axes[1, 0].plot(range(window-1, len(losses)), moving_avg, 
                           linewidth=2, label=f'{window}-Step Moving Avg')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Reward Distribution
    axes[1, 1].hist(episode_rewards, bins=50, edgecolor='black')
    axes[1, 1].set_xlabel('Episode Reward')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Episode Rewards')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_results.png'), dpi=150)
    print(f"Training plots saved to {os.path.join(save_path, 'training_results.png')}")
    plt.close()


def main():
    """Main training function."""
    
    # Load corpus
    try:
        words = load_corpus('data/corpus.txt')
    except (FileNotFoundError, ValueError) as e:
        print(f"\n{e}")
        sys.exit(1)
    
    # Split into train/test
    train_words, test_words = split_corpus(words, train_ratio=0.8, random_seed=42)
    print(f"Train words: {len(train_words)}, Test words: {len(test_words)}")
    
    # Load HMM
    if not os.path.exists('results/hmm.pkl'):
        print("\nERROR: HMM model not found at results/hmm.pkl")
        print("Please run train_hmm.py first!")
        sys.exit(1)
    
    hmm_model = HangmanHMM.load('results/hmm.pkl')
    
    # Train DQN
    os.makedirs('results', exist_ok=True)
    
    agent, stats = train_dqn(
        hmm_model=hmm_model,
        train_words=train_words,
        num_episodes=6000,
        pretrain_epochs=5,
        save_freq=500
    )
    
    # Plot results
    plot_training_results(stats)
    
    # Save configuration
    config = {
        'num_episodes': 5000,
        'state_size': agent.state_size,
        'action_size': 26,
        'learning_rate': 0.0005,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.05,
        'epsilon_decay': 0.997,
        'batch_size': 64,
        'pretrain_epochs': 10,
        'train_words': len(train_words),
        'test_words': len(test_words)
    }
    
    with open('results/config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Model saved to: results/dqn.pth")
    print(f"Best model saved to: results/dqn_best.pth")
    print(f"Config saved to: results/config.json")
    print(f"Plots saved to: results/training_results.png")


if __name__ == "__main__":
    main()