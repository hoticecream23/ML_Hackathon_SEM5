# src/agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import string

ALPH = string.ascii_lowercase
A2I = {a: i for i, a in enumerate(ALPH)}

class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=26):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class HangmanAgent:
    def __init__(self, state_dim, lr=1e-3, gamma=0.95, device="cpu"):
        self.device = device
        self.model = DQN(state_dim).to(self.device)
        self.target = DQN(state_dim).to(self.device)
        self.target.load_state_dict(self.model.state_dict())
        self.optim = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.steps = 0

    def act(self, state, epsilon, guessed_set=None):
        """
        state: 1D numpy array (state vector)
        guessed_set: set of chars that must be masked out (optional)
        """
        if np.random.rand() < epsilon:
            # random among unguessed letters if possible
            if guessed_set:
                choices = [c for c in ALPH if c not in guessed_set]
                return np.random.choice(choices) if choices else np.random.choice(list(ALPH))
            else:
                return np.random.choice(list(ALPH))

        # model expects batch dimension
        st = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.model(st).squeeze(0).cpu().numpy()  # shape (26,)

        # mask out already guessed letters so they are not chosen
        if guessed_set:
            for g in guessed_set:
                if g in A2I:
                    q[A2I[g]] = -1e9

        idx = int(np.argmax(q))
        return ALPH[idx]

    def update(self, batch, device="cpu"):
        s, a, r, ns, d = batch
        # convert lists to tensors
        s = torch.tensor(np.stack(s), dtype=torch.float32, device=self.device)
        ns = torch.tensor(np.stack(ns), dtype=torch.float32, device=self.device)
        r = torch.tensor(r, dtype=torch.float32, device=self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device)
        a_idx = torch.tensor([A2I[x] for x in a], dtype=torch.long, device=self.device)

        q = self.model(s).gather(1, a_idx.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            max_q_next = self.target(ns).max(1)[0]
            target = r + self.gamma * max_q_next * (1 - d)
        loss = (q - target).pow(2).mean()
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.steps += 1
        if self.steps % 200 == 0:
            self.target.load_state_dict(self.model.state_dict())
        return loss.item()
