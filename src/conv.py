import torch.nn as nn
import torch.nn.functional as F
from rmdn import RecursiveMetadataNorm


class Conv(nn.Module):
    """
    A 2D convolutional neural network with two convolutional layers and two fully connected layers.
    Statistical regression using R-MDN layers is performed after every convolutional layer and pre-
    logits layer.
    
    Parameters
        debias (bool): true if R-MDN should be performed, else false
        forgetting_factor (float): the closer to 1, the less forgetting of past information
        reg (float): regularization parameter
    """
    def __init__(
        self,
        debias=False,
        forgetting_factor=0.999,
        reg=1e-4,
    ):
        super(Conv, self).__init__()

        self.debias = debias

        self.conv1 = nn.Conv2d(1, 16, 5)
        if self.debias:
            self.residualize1 = RecursiveMetadataNorm(num_features=16*28*28, forgetting_factor=forgetting_factor, reg=reg)
        self.conv2 = nn.Conv2d(16, 32, 5)
        if self.debias:
            self.residualize2 = RecursiveMetadataNorm(32*24*24, forgetting_factor=forgetting_factor, reg=reg)
        self.fc1 = nn.Linear(18432, 84)
        if self.debias:
            self.residualize3 = RecursiveMetadataNorm(84, forgetting_factor=forgetting_factor, reg=reg)
        self.fc2 = nn.Linear(84, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, cfs):
        x = self.conv1(x)
        if self.debias:
            x = self.residualize1(x, cfs)
        x = F.relu(x)
        x = self.conv2(x)
        if self.debias:
            x = self.residualize2(x, cfs)
        x = F.relu(x)
        x = x.view(-1, 18432)
        x = self.fc1(x)
        if self.debias:
            x = self.residualize3(x, cfs)
        fc = x.cpu().detach().numpy()
        x = F.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x, fc