import torch
import torch.nn as nn


class RecursiveMetadataNorm(nn.Module):
    def __init__(self, num_features, forgetting_factor=0.999, reg=1e-4): 
        """
        Recursive Metadata Normalization (R-MDN) module. R-MDN can be inserted at any stage within a DNN
        to regress out the influence of confounding variables from intermediate feature representations.
.
        Parameters
            num_features (int): number of features used to initialize beta
            forgetting_factor (float): forgetting factor
            reg (float): regularization parameter
        """
        super(RecursiveMetadataNorm, self).__init__()
        self.num_features = num_features
        delta = 0.01
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.P = torch.eye(3, device=self.device) / delta
        self.register_buffer('beta', torch.zeros(3, num_features))
        self.forgetting_factor = forgetting_factor
        self.reg = reg

    def forward(self, x, bias):
        Y = x
        N = x.shape[0]
        Y = Y.reshape(N, -1)
        X_batch = bias.to(self.device)
        
        if self.training:
            self.P = (1 / self.forgetting_factor) * self.P
            K = self.P @ X_batch.T @ torch.linalg.inv(torch.eye(N, device=self.device) + X_batch @ self.P @ X_batch.T)
            self.P = self.P - K @ X_batch @ self.P + self.reg * torch.eye(3, device=self.device)
            with torch.no_grad():
                self.beta += K @ (Y - X_batch @ self.beta)
    
        Y_r = torch.mm(X_batch[:, 1:], self.beta[1:]) 
        residual = Y - Y_r
        residual = residual.reshape(x.shape)
        return residual