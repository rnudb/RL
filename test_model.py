import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, action_size, seed=42):
        super(QNetwork, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.seed = torch.manual_seed(seed)

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(64*7*7, 512)
        self.fc2 = nn.Linear(512, action_size)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        # x = F.dropout(x, p=0.2)
        x = F.relu(self.conv2(x))
        # x = F.dropout(x, p=0.2)
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def step(self, state):
        state = state.float().to(self.device)
        self.eval()
        # with torch.no_grad():
        action_values = self.forward(state)
        self.train()
        # print("action values: ", action_values.cpu().data.numpy())
        action = np.argmax(action_values.cpu().data.numpy()[-1])
        return action
    

import cv2

def preprocess_image_stack(img_stack):
    res = []
    for idx, image in enumerate(img_stack):
        # First, to encode a singleframe we take the maximum value for each pixel colour
        # value over the frame being encoded and the previous frame. 
        img = np.array(image)
        prev_img = img_stack[idx - 1]
        img = np.maximum(img, prev_img)

        # Second, we then extract the Y channel, also known as luminance, from the RGB frame
        luminance = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

        # Add luminance as 4th channel to img
        img = np.dstack((img, luminance))
        
        # Resize to 84 x 84
        # img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
        res.append(img.reshape(4, 84, 84))

    # Stack into a single tensor
    return torch.from_numpy(np.stack(res, axis=0)).float()


import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import time

model = QNetwork(4, 42)
model.load_state_dict(torch.load('model.pt'))

model.eval()

# make the model play the game live
env = gym.make("ALE/Breakout-v5", frameskip=1, render_mode="human")
env = gym.wrappers.AtariPreprocessing(env, grayscale_obs=False)
env = gym.wrappers.FrameStack(env, num_stack=4)
env.metadata["render_fps"] = 60
# num_stack_frames=4max_episode_reward = 0

# env = gym.wrappers.FrameStack(env, num_stack=num_stack_frames)


def run_env():
    state, _ = env.reset()
    state, _, _, _, _ = env.step(1) # One action before the game starts

    state = preprocess_image_stack(state)
    done = False
    i=0
    while not done:
        action = model.step(state)
        print(action, i)
        # for _ in range(4):
        state, r, done, _, info = env.step(action)
        state = preprocess_image_stack(state)
        env.render()

        env.step(1)
        i += 1

    env.close()

run_env()