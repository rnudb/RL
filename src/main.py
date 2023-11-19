from utils import fill_replay_buffer, preprocess_image_stack, update_epsilon
from qnetwork import declare_model_and_optimier

import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import torch

FRAMESKIP = 4
NUM_STACK = 4
# NUM_ACTION_IN_A_ROW = 4

EPSILON = 1
GAMMA = 0.99
LEARNING_RATE = 1e-4

NUM_EPISODES = 5
NUM_EPOCHS = 50000
NUM_EPOCHS_BEFORE_NEW_QLOSS = 10000
NUM_EPOCHS_BEFORE_LEARNING = 10000
BUFFER_SIZE = 5000
BATCH_SIZE = 128


def new_env():
    env = gym.make("ALE/Breakout-v5", autoreset=True, frameskip=1)
    env = gym.wrappers.AtariPreprocessing(env, scale_obs=False, grayscale_obs=False,frame_skip=FRAMESKIP) # We rescale in the forward loop and we need to keep to rgb values for luminance preprocessing
    env = gym.wrappers.FrameStack(env, num_stack=NUM_STACK)
    return env

def main():
    env = new_env()
    
    # Create a DQN Agent
    model, optimizer = declare_model_and_optimier(env=env, lr=LEARNING_RATE)
    model.to(model.device)
    losses_per_episode = []
    rewards_per_episode = []

    for episode in range(NUM_EPISODES):
        env = new_env()
        replay_buffer = []

        qnet_loss, _ = declare_model_and_optimier(env, model_state_dict=model.state_dict())
        qnet_loss.to(model.device)
        done = False

        # Pepare the environment
        s_t = preprocess_image_stack(env.reset()[0])

        losses = []
        rewards = []

        for epoch in range(NUM_EPOCHS):
            # Epsilon decay
            epsilon = update_epsilon(EPSILON, epoch, NUM_EPOCHS)

            # Get action from the model or random
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = model.step(s_t)

            s_t1, reward, done, _, info = env.step(action)
            
            # Preprocess the next frame
            s_t1 = preprocess_image_stack(s_t1)

            # Update replay buffer
            if len(replay_buffer) > BUFFER_SIZE:
                # Generate an array of random numbers in range BUFFER_SIZE
                idx = np.random.randint(0, BUFFER_SIZE, int(BUFFER_SIZE * 1/10))
                # Remove the elements from the replay buffer
                for i in sorted(idx, reverse=True):
                    replay_buffer.pop(i)
            replay_buffer.append((
                torch.tensor(np.array(s_t.to('cpu'))[-1, :, :]),
                action,
                reward,
                torch.tensor(np.array(s_t.to('cpu'))[-1, :, :]),
                done,
            ))

            # Current state is now the next state
            s_t = s_t1

            if epoch < NUM_EPOCHS_BEFORE_LEARNING:
                continue

            # Sample random minibatch of transitions from D
            minibatch = random.sample(replay_buffer, BATCH_SIZE)

            # Unpack the transitions in minibatch
            loss_s_t, loss_action, loss_reward, loss_s_t1, loss_done = zip(*minibatch)

            # Convert to tensors the minibatch
            loss_s_t = torch.stack(loss_s_t).float().to(model.device)
            loss_action = torch.tensor(loss_action).long().to(model.device)
            loss_reward = torch.tensor(loss_reward).float().to(model.device)
            loss_s_t1 = torch.stack(loss_s_t1).float().to(model.device)
            loss_done = torch.tensor(loss_done).float().to(model.device)

            # Compute each outputs for minibatch from curent model
            outputs = torch.max(model(loss_s_t), dim=1).values.to(model.device)

            # Compute labels for each outputs (qnet_loss)
            y_j = loss_reward + GAMMA * torch.max(qnet_loss(loss_s_t1), dim=1).values * loss_done
            y_j = y_j.float().to(model.device)

            # Backpropagate the loss
            loss = model.criterion(outputs, y_j)
            losses.append(loss.item())
            rewards.append(reward)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 500 == 0:
                print("TOTAL EPOCH: {}, Loss: {}, Mean Reward: {}, Current Reward: {}, Actions in batch: {}".format(epoch, loss.item(), loss_reward.mean().cpu().detach(), reward,{k:v for k,v in zip(*np.unique(loss_action.to('cpu'), return_counts=True))}))

            # Update the goal network
            if epoch % NUM_EPOCHS_BEFORE_NEW_QLOSS == 0:
                qnet_loss, _ = declare_model_and_optimier(env, model_state_dict=model.state_dict())
                qnet_loss.to(model.device)
                print("New QNet Loss")

        # Save losses and rewrads for this episode
        losses_per_episode.append(losses)
        rewards_per_episode.append(rewards)

        # Save the model
        torch.save(model.state_dict(), f"models/model_ep_{episode}.pt")

    # Print losses one the same plot with different colors
    for episode in range(NUM_EPISODES):
        plt.plot(losses_per_episode[episode], label=f"Episode {episode}")
    plt.legend()
    plt.savefig("graphs/losses.png")

    # Print rewards one the same plot with different colors
    for episode in range(NUM_EPISODES):
        plt.plot(rewards_per_episode[episode], label=f"Episode {episode}")
    plt.legend()
    plt.savefig("graphs/rewards.png")



if __name__ == "__main__":
    main()