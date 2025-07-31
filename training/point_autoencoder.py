import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_utils import persistence

'''
PointNet AutoEncoder
Learning Representations and Generative Models For 3D Point Clouds
https://arxiv.org/abs/1707.02392
'''

def jitter_point_cloud(pc_batch, sigma=0.01, clip=0.05):
    """
    Randomly jitters a batch of point clouds.

    Args:
        pc_batch (torch.Tensor): The batch of point clouds to jitter, 
                                 shape (B, 3, N).
        sigma (float): The standard deviation of the Gaussian noise to add.
        clip (float): The maximum absolute value for the noise. This prevents
                      adding outlier noise.

    Returns:
        torch.Tensor: The jittered point cloud batch.
    """
    B, C, N = pc_batch.shape
    assert(C == 3)

    noise = torch.randn(B, C, N, device=pc_batch.device) * sigma

    clipped_noise = torch.clamp(noise, -clip, clip)

    jittered_pc = pc_batch + clipped_noise
    
    return jittered_pc


@persistence.persistent_class
class PointAutoEncoder(nn.Module):
    def __init__(self, point_size, latent_size, sigma=0.01, clip=0.03):
        super().__init__()
        
        self.latent_size = latent_size
        self.point_size = point_size
        self.sigma = sigma
        self.clip = clip

        self.dropout = torch.nn.Dropout(0.8)
        
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, self.latent_size, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(self.latent_size)
        
        self.dec1 = nn.Linear(self.latent_size,256)
        self.dec2 = nn.Linear(256,256)
        self.dec3 = nn.Linear(256,self.point_size*3)

    def encoder(self, x):
        x = jitter_point_cloud(x, self.sigma, self.clip)
        x = self.dropout(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.latent_size)
        return x
    
    def decoder(self, x):
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = self.dec3(x)
        return x.view(-1, self.point_size, 3)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    