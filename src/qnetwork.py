import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from torch.optim import RMSprop, Adam

# Create a Q-Network
class QNetwork(nn.Module):
    def __init__(self, action_size, seed):
        super(QNetwork, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.seed = torch.manual_seed(seed)
        self.normalize = lambda x: x / 255.0
        self.criterion = nn.MSELoss()


        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(64*7*7, 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        x = self.normalize(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def step(self, state):
        state = state.float().to(self.device)
        action_values = self.forward(state)
        action = np.argmax(action_values.cpu().data.numpy()[-1])
        return action


def declare_model_and_optimier(env, model_state_dict=None, optimizer_state_dict=None, lr=1e-4, eps=1e-7):
    """
    Returns new model and load state dict if needed
    Declares a new optimizer with the model
    """
    model = QNetwork(action_size=env.action_space.n, seed=42)
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)


    # optimizer = RMSprop(model.parameters(), lr=lr, momentum=0.95, eps=eps)
    optimizer = Adam(params=model.parameters(), lr=lr, eps=eps)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    return model, optimizer