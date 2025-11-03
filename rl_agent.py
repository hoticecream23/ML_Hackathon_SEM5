"""
Deep Q-Network (DQN) agent for Hangman.

This module implements a DQN agent with:
- Experience replay buffer
- Target network for stable training
- Epsilon-greedy exploration with decay
- Supervised pretraining using HMM outputs
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Dict, Tuple, List


class DQN(nn.Module):
    """
    Deep Q-Network for Hangman.
    
    Takes as input a state vector containing:
    - Encoded masked word
    - Guessed letters binary vector
    - Remaining lives
    - HMM probability distribution
    
    Outputs Q-values for each of the 26 possible actions (letters).
    """
    
    def __init__(self, state_size: int, action_size: int = 26, hidden_size: int = 256):
        """
        Initialize the DQN.
        
        Args:
            state_size: Size of the state vector
            action_size: Number of actions (26 letters)
            hidden_size: Size of hidden layers
        """
        super(DQN, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, action_size)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        """Forward pass through the network."""
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class ReplayBuffer:
    """
    Experience replay buffer for DQN.
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add an experience to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample a batch of experiences."""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)


class DQNAgent:
    """
    DQN Agent for playing Hangman.
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int = 26,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 10,
        device: str = None
    ):
        """
        Initialize the DQN agent.
        
        Args:
            state_size: Size of the state vector
            action_size: Number of actions (26)
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon
            buffer_capacity: Replay buffer capacity
            batch_size: Batch size for training
            target_update_freq: Frequency of target network updates
            device: Device to use (cuda/cpu)
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"DQN Agent using device: {self.device}")
        
        # Networks
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_capacity)
        
        # Training statistics
        self.training_step = 0
        self.episode_rewards = []
    
    def select_action(self, state_vector: np.ndarray, valid_actions: np.ndarray = None, training: bool = True) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state_vector: Current state as numpy array
            valid_actions: Binary mask of valid actions (1 = valid)
            training: Whether in training mode (affects exploration)
            
        Returns:
            Selected action index
        """
        if valid_actions is None:
            valid_actions = np.ones(self.action_size)
        
        # Exploration: random action
        if training and random.random() < self.epsilon:
            valid_indices = np.where(valid_actions > 0)[0]
            if len(valid_indices) > 0:
                return np.random.choice(valid_indices)
            else:
                return np.random.randint(0, self.action_size)
        
        # Exploitation: best action according to Q-network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor).cpu().numpy()[0]
            
            # Mask invalid actions
            q_values = q_values * valid_actions + (1 - valid_actions) * (-1e9)
            
            return np.argmax(q_values)
    
    def select_action_with_hmm(self, state_vector: np.ndarray, hmm_probs: np.ndarray, 
                                valid_actions: np.ndarray = None, alpha: float = 0.5) -> int:
        """
        Select action combining DQN Q-values with HMM probabilities.
        
        Args:
            state_vector: Current state
            hmm_probs: HMM probability distribution
            valid_actions: Binary mask of valid actions
            alpha: Weight for combining (0 = only DQN, 1 = only HMM)
            
        Returns:
            Selected action
        """
        if valid_actions is None:
            valid_actions = np.ones(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor).cpu().numpy()[0]
            
            # Normalize Q-values to probabilities
            q_values = q_values - q_values.min()
            if q_values.max() > 0:
                q_values = q_values / q_values.max()
            
            # Combine with HMM
            combined = (1 - alpha) * q_values + alpha * hmm_probs
            
            # Mask invalid actions
            combined = combined * valid_actions + (1 - valid_actions) * (-1e9)
            
            return np.argmax(combined)
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self) -> float:
        """
        Perform one training step (batch update).
        
        Returns:
            Loss value
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q-values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.training_step += 1
        
        # Update target network
        if self.training_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def update_epsilon(self):
        """Decay epsilon for exploration."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def supervised_pretrain(self, states: List[np.ndarray], hmm_probs_list: List[np.ndarray], 
                           epochs: int = 10, lr: float = 0.001):
        """
        Pretrain the network using HMM outputs (behavioral cloning).
        
        Args:
            states: List of state vectors
            hmm_probs_list: List of HMM probability distributions (targets)
            epochs: Number of training epochs
            lr: Learning rate for pretraining
        """
        print(f"Supervised pretraining for {epochs} epochs...")
        
        # Create optimizer for pretraining
        pretrain_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        targets_tensor = torch.FloatTensor(np.array(hmm_probs_list)).to(self.device)
        
        # Training loop
        for epoch in range(epochs):
            # Shuffle data
            indices = torch.randperm(len(states))
            states_shuffled = states_tensor[indices]
            targets_shuffled = targets_tensor[indices]
            
            total_loss = 0.0
            num_batches = 0
            
            # Batch training
            for i in range(0, len(states), self.batch_size):
                batch_states = states_shuffled[i:i+self.batch_size]
                batch_targets = targets_shuffled[i:i+self.batch_size]
                
                # Forward pass
                outputs = self.policy_net(batch_states)
                
                # Cross-entropy loss (treat as classification)
                loss = -torch.sum(batch_targets * torch.log_softmax(outputs, dim=1)) / len(batch_states)
                
                # Backward pass
                pretrain_optimizer.zero_grad()
                loss.backward()
                pretrain_optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            if (epoch + 1) % 2 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        print("Pretraining complete!")
        
        # Update target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, filepath: str):
        """Save the agent's networks."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
        }, filepath)
        print(f"Agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load the agent's networks."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        self.training_step = checkpoint.get('training_step', 0)
        print(f"Agent loaded from {filepath}")