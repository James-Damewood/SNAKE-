import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import random
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
import numpy as np
from RL_Arena import Game
import pygame
import time

pygame.init()
Window_size = 100
block_size = 10
gameDisplay = pygame.display.set_mode((Window_size,Window_size))
gameDisplay.fill((0, 255, 0))

env = Game()

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

######## From Pytorch Tutorial
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######## From Pytorch Tutorial
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self,h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(5, 10, kernel_size=3, stride=1,padding = 0).double()
        self.bn1 = nn.BatchNorm2d(10).double()
        self.conv2 = nn.Conv2d(10, 10, kernel_size=3, stride=1,padding = 0).double()
        self.bn2 = nn.BatchNorm2d(10).double()
        self.Lin = nn.Linear(1000, 5).double()

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.view(-1,10,10)
        board_view = torch.zeros(x.shape[0],5,10,10)
        board_view[:,0,:,:] = torch.eq(x,0)
        board_view[:,1,:,:] = torch.eq(x,1)
        board_view[:,2,:,:] = torch.eq(x,2)
        board_view[:,3,:,:] = torch.eq(x,3)
        board_view[:,4,:,:] = torch.eq(x,4)
        board_view = board_view.view(-1,5,10,10).double()
        p2d = (1, 1, 1, 1)
        board_view = F.pad(board_view,p2d,"constant",-1)
        board_view = F.relu(self.bn1(self.conv1(board_view)))
        board_view = F.pad(board_view,p2d,"constant",-1)
        board_view = F.relu(self.bn2(self.conv2(board_view)))
        return self.Lin(board_view.view(board_view.size(0), -1))

def get_board(env):
    return env.get_board()

BATCH_SIZE = 64
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 100
TARGET_UPDATE = 10
n_actions = 5
init_board = env.get_board()
policy_net = DQN(10, 10, n_actions)
target_net = DQN(10, 10, n_actions)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)
steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Length')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
       means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
       means = torch.cat((torch.zeros(99), means))
       plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device)

    #print(batch.next_state)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device).double()
    next_state_values[non_final_mask] = target_net(non_final_next_states.double()).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch.double()

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

episode_durations = []
num_episodes = 400

colors = {
            0: (0,255,0),
            1: (0,0,255),
            2: (0,0,255),
            3: (0,0,255),
            4: (255,0,0)
}

def draw_grid(env):
    grid_points = np.linspace(0,Window_size-block_size,int(Window_size/block_size))
    #print(grid_points)
    board = env.get_board()
    #print(board)
    for i in range(len(grid_points)):
        for j in range(len(grid_points)):
            #print(colors[board[i,j]])
            pygame.draw.rect(gameDisplay,colors[board[i,j]],(grid_points[i],grid_points[j],block_size,block_size),0)


for i_episode in range(num_episodes):
    # Initialize the environment and state
    iter = Game()
    state = torch.tensor(iter.get_board())
    for t in count():
        # Select and perform an action
        action = select_action(state)

        reward, done = iter.game_step(action.item())

        reward = torch.tensor([reward], device=device)

        # Observe new state
        if not done:
            next_state = torch.tensor(iter.get_board())
        else:
            next_state = None


        #print("here")
        #print(state)
        #print(next_state)
        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state
        keys=pygame.key.get_pressed()
        pygame.time.delay(50)
        draw_grid(iter)

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_on = False
        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            length = iter.get_total_performance()
            episode_durations.append(length)
            plot_durations()
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
#env.render()
#env.close()
plt.ioff()
plt.show()
pygame.quit()
