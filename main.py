import numpy
import pygame as game
from rimworldenv import SimpleRimWorldEnv

import gym
import collections
import random

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000
batch_size = 32


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self, sizeGrid, maxActors):
        super(Qnet, self).__init__()
        self.actionspace = sizeGrid*maxActors + sizeGrid*maxActors

        self.fc1 = nn.Linear(sizeGrid, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, sizeGrid*maxActors + sizeGrid*maxActors)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, self.actionspace - 1)
        else:
            return out.argmax().item()


def train(q, q_target, memory, optimizer):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1, a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def actionMove(targetId: int, pos: tuple):
    return targetId, (True, pos), 0, (False, (0, 0))


def actionAttack(targetId: int, pos: tuple):
    return targetId, (False, (0, 0)), 0, (True, pos)


SIZE_X = 10
SIZE_Y = 10
MAX_ACTORS = 2


def translateToAction(result: int):
    index = numpy.unravel_index(result, (SIZE_X, SIZE_Y, MAX_ACTORS, 2))
    if index[3] == 0:
        return actionMove(index[2], (index[0], index[1]))
    else:
        return actionAttack(index[2], (index[0], index[1]))


def main():
    plot = True
    xToPlot = []
    yToPlot = []

    render = False

    if render:
        screen = game.display.set_mode((640, 480))
    else:
        screen = None

    env = SimpleRimWorldEnv(SIZE_X, SIZE_Y, screen)
    q = Qnet(SIZE_X*SIZE_Y, MAX_ACTORS)
    q_target = Qnet(SIZE_X*SIZE_Y, MAX_ACTORS)
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    total = 10000
    print_interval = 20
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(total):
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))  # Linear annealing from 8% to 1%
        s = env.reset()
        done = False

        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, r, done, info = env.step(translateToAction(a))
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r / 100.0, s_prime, done_mask))
            s = s_prime

            score += r
            if render:
                env.render()
            if done:
                break

        if memory.size() > 2000:
            train(q, q_target, memory, optimizer)

        if plot:
            xToPlot.append(n_epi)
            yToPlot.append(score)
            score = 0.0

            if n_epi/total*100 - int(n_epi/total*100) < 0.01:
                print("{:.2f}%".format(n_epi/total*100))
        else:
            if n_epi % print_interval == 0 and n_epi != 0:
                q_target.load_state_dict(q.state_dict())
                print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                    n_epi, score / print_interval, memory.size(), epsilon * 100))
                score = 0.0
    env.close()

    plt.plot(xToPlot, yToPlot, 'ok')
    plt.xlabel("Episode number")
    plt.ylabel("Score")
    plt.show()

if __name__ == '__main__':
    main()
