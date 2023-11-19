import numpy as np
import torch

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
        res.append(img.reshape(4, 84, 84)) # Already in the right img format

    # Stack into a single tensor
    return torch.from_numpy(np.stack(res, axis=0)).float()

def fill_replay_buffer(env, replay_buffer, buffer_size=100000):
    """
    Fills the replay buffer with random actions
    """
    env.reset()
    state = env.step(0)[0]
    state = preprocess_image_stack(state)
    state = state
    for i in range(buffer_size):
        action = env.action_space.sample()
        next_state, reward, done, _, _ = env.step(action)
        next_state = preprocess_image_stack(next_state)
        next_state = next_state
        replay_buffer.append((
            torch.tensor(np.array(state)[-1, :, :]),
            action,
            reward,
            torch.tensor(np.array(next_state)[-1, :, :]),
            done
            ))
        state = next_state
        if done:
            env.reset()
            state = env.step(0)[0]
            state = preprocess_image_stack(state)
            state = state
    
    return replay_buffer

def update_epsilon(epsilon, current_epoch, max_epoch, epsilon_min=0.1):
    """
    Updates epsilon according to the current epoch and max epoch
    """
    return max(epsilon_min, epsilon - (epsilon / max_epoch) * current_epoch)