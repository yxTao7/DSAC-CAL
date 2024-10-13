import numpy as np
import random
from collections import namedtuple, deque
from model import Actor, Critic, QcNetwork
import torch
import torch.optim as optim
from torch.distributions import Normal
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

hidden_size = 128
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128  # minibatch size
WARM_SIZE = 200
GAMMA = 0.99  # discount factor
TAU = 0.005  # for soft update of target parameters
LR_ACTOR = 1e-4  # learning rate of the actor
LR_CRITIC = 1e-4  # learning rate of the critic
LR_QC = 1e-4  # learning rate of the safety critic
alpha_lr = 3e-4  # learning rate of alpha
lam_lr = 3e-4  # learning rate of lam
Delay_update = 2
EPS = 1e-6
k = 0.5
c = 10
qc_E = 4


def requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad_(value)


class Agent:

    def __init__(self, state_size, action_size, random_seed):

        self.state_size = state_size
        self.action_size = action_size
        self.iter = 1
        self.mean_std1 = -1.0
        self.mean_std2 = -1.0
        self.target_cost = 1

        # Actor Network
        self.actor = Actor(state_size, action_size, hidden_size, random_seed)
        self.target_actor = Actor(state_size, action_size, hidden_size, random_seed)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        # Critic Network
        self.critic_1 = Critic(state_size, action_size, hidden_size, random_seed)
        self.critic_2 = Critic(state_size, action_size, hidden_size, random_seed)
        self.target_critic_1 = Critic(state_size, action_size, hidden_size, random_seed)
        self.target_critic_2 = Critic(state_size, action_size, hidden_size, random_seed)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=LR_CRITIC)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=LR_CRITIC)

        # Safety Critic Networks
        self.safety_critics = QcNetwork(state_size, action_size, qc_E, hidden_size, random_seed)
        self.safety_critic_targets = QcNetwork(state_size, action_size, qc_E, hidden_size, random_seed)
        self.safety_critic_targets.load_state_dict(self.safety_critics.state_dict())
        self.safety_critic_optimizer = torch.optim.Adam(self.safety_critics.parameters(), lr=LR_QC)

        # 冻结目标网络梯度
        requires_grad(self.target_actor, False)
        requires_grad(self.target_critic_1, False)
        requires_grad(self.target_critic_2, False)
        requires_grad(self.safety_critic_targets, False)

        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.nn.Parameter(torch.tensor(0.8, dtype=torch.float32))
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = -action_size  # 目标熵的大小

        # lam Parameter
        self.log_lam = torch.tensor(np.log(np.clip(0.6931, 1e-8, 1e8)))
        self.log_lam.requires_grad = True
        self.log_lam_optimizer = torch.optim.Adam([self.log_lam], lr=lam_lr)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def step(self, state, action, reward, cost, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience
        self.memory.add(state, action, reward, cost, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > WARM_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences)

    def act(self, state):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float()
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state)[0].detach().numpy()
        self.actor.train()
        return action

    def act_d(self, state):
        """Returns actions with no sample for test."""
        state = torch.from_numpy(state).float()
        action = self.actor(state)[-1].detach().numpy()
        action = np.clip(action, -1, 1)
        return action

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples.
        """
        states, actions, rewards, costs, next_states, dones = experiences

        # update q network
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        next_act, log_prob_next_act = self.target_actor(next_states)[0], self.target_actor(next_states)[1]

        StochaQ1 = self.critic_1(states, actions)
        mean1, std1 = StochaQ1[0], StochaQ1[1]
        q1, q1_std = mean1, std1

        StochaQ2 = self.critic_2(states, actions)
        mean2, std2 = StochaQ2[0], StochaQ2[1]
        q2, q2_std = mean2, std2

        if self.mean_std1 == -1.0:
            self.mean_std1 = torch.mean(q1_std.detach())
        if self.mean_std2 == -1.0:
            self.mean_std2 = torch.mean(q2_std.detach())

        StochaQ1n = self.target_critic_1(next_states, next_act)
        mean1n, std1n = StochaQ1n[0], StochaQ1n[1]
        normal1 = Normal(torch.zeros_like(mean1n), torch.ones_like(std1n))
        z1 = normal1.sample()
        z1 = torch.clamp(z1, -3, 3)
        q_value1n = mean1n + torch.mul(z1, std1n)
        q1_next, _, q1_next_sample = mean1n, std1n, q_value1n

        StochaQ2n = self.target_critic_2(next_states, next_act)
        mean2n, std2n = StochaQ2n[0], StochaQ2n[1]
        normal2 = Normal(torch.zeros_like(mean2n), torch.ones_like(std2n))
        z2 = normal2.sample()
        z2 = torch.clamp(z2, -3, 3)
        q_value2n = mean2n + torch.mul(z2, std2n)
        q2_next, _, q2_next_sample = mean2n, std2n, q_value2n

        q_next = torch.min(q1_next, q2_next)
        q_next_sample = torch.where(torch.lt(q1_next, q2_next), q1_next_sample, q2_next_sample)

        target_q = rewards + (1 - dones) * GAMMA * (
                q_next.detach() - self.log_alpha.exp().item() * log_prob_next_act.detach())
        target_q_sample = rewards + (1 - dones) * GAMMA * (
                q_next_sample.detach() - self.log_alpha.exp().item() * log_prob_next_act.detach())
        td_bound = 3 * self.mean_std1.detach()
        target_z = torch.clamp(target_q_sample, q1.detach()-td_bound, q1.detach()+td_bound)
        target_q1, target_z1 = target_q.detach(), target_z.detach()

        td_bound = 3 * self.mean_std2.detach()
        target_z = torch.clamp(target_q_sample, q2.detach() - td_bound, q2.detach() + td_bound)
        target_q2, target_z2 = target_q.detach(), target_z.detach()

        # bound
        q1_std_detach = torch.clamp(q1_std, min=0.).detach()
        q2_std_detach = torch.clamp(q2_std, min=0.).detach()
        bias = 0.1

        q1_loss = (torch.pow(self.mean_std1, 2) + bias) * torch.mean(-(target_q1 - q1.detach()) / (torch.pow(q1_std_detach, 2) + bias) * q1
            - ((torch.pow(q1.detach() - target_z1, 2) - q1_std_detach.pow(2)) / (torch.pow(q1_std_detach, 3) + bias)) * q1_std)

        q2_loss = (torch.pow(self.mean_std2, 2) + bias) * torch.mean(-(target_q2 - q2.detach()) / (torch.pow(q2_std_detach, 2) + bias) * q2
            - ((torch.pow(q2.detach() - target_z2, 2) - q2_std_detach.pow(2)) / (torch.pow(q2_std_detach, 3) + bias)) * q2_std)
        loss_q = q1_loss + q2_loss
        loss_q.backward()

        # update safety network
        self.safety_critic_optimizer.zero_grad()
        QCs = self.safety_critics(states, actions)
        next_QCs = self.safety_critic_targets(next_states, next_act)
        target_QCs = costs[None, :, :].repeat(qc_E, 1, 1) + (1 - dones)[None, :, :].repeat(qc_E, 1, 1) * GAMMA * next_QCs
        safety_critic_loss = F.mse_loss(QCs, target_QCs.detach())
        safety_critic_loss.backward()

        # update policy network
        requires_grad(self.critic_1, False)
        requires_grad(self.critic_2, False)
        requires_grad(self.safety_critics, False)
        self.actor_optimizer.zero_grad()

        new_act, new_log_prob = self.actor(states)[0], self.actor(states)[1]
        StochaQ1new = self.critic_1(states, new_act)
        q1_new = StochaQ1new[0]
        StochaQ2new = self.critic_2(states, new_act)
        q2_new = StochaQ2new[0]
        q_new = torch.min(q1_new, q2_new)

        new_QCs = self.safety_critics(states, new_act)
        qc_std, qc_mean = torch.std_mean(QCs, dim=0)
        QC = qc_mean + k * qc_std
        new_qc_std, new_qc_mean = torch.std_mean(new_QCs, dim=0)
        Qc_ucb = new_qc_mean + k * new_qc_std

        rect = c * torch.mean(self.target_cost - QC)
        rect = torch.clamp(rect.detach(), max=self.log_lam.exp().item())

        loss_policy = torch.mean(self.log_alpha.exp().item() * new_log_prob - q_new + (self.log_lam.exp().item() - rect) * Qc_ucb)
        entropy = - new_log_prob.detach().mean()
        loss_policy.backward()
        requires_grad(self.critic_1, True)
        requires_grad(self.critic_2, True)
        requires_grad(self.safety_critics, True)

        # update alpha
        self.log_alpha_optimizer.zero_grad()
        loss_alpha = torch.mean(-self.log_alpha.exp() * (new_log_prob.detach() + self.target_entropy))
        loss_alpha.backward()

        # update lamda
        self.log_lam_optimizer.zero_grad()
        lam_loss = torch.mean(self.log_lam.exp() * (self.target_cost - QC).detach())
        lam_loss.backward()

        # update critic and delay update others
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        if self.iter % Delay_update == 0:
            self.safety_critic_optimizer.step()
            self.actor_optimizer.step()
            self.log_alpha_optimizer.step()
            self.log_lam_optimizer.step()
            self.iter = 0
        self.iter += 1

        # soft update target network
        with torch.no_grad():
            self.soft_update(self.critic_1, self.target_critic_1, TAU)
            self.soft_update(self.critic_2, self.target_critic_2, TAU)
            self.soft_update(self.safety_critics, self.safety_critic_targets, TAU)
            self.soft_update(self.actor, self.target_actor, TAU)


class ReplayBuffer:

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
             buffer_size (int): maximum size of buffer
             batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "cost", "next_state", "done"])
        np.random.seed(seed)

    def add(self, state, action, reward, cost, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, cost, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        costs = torch.from_numpy(np.vstack([e.cost for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()
        return states, actions, rewards, costs, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)