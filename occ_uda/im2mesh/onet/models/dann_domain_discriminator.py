import torch.nn as nn
import torch.nn.functional as F
from .gradient_reversal_module import GradientReversal

class DANN_Domain_Discriminator(nn.Module):

    def __init__(self, c_dim, grad_rev=True):
        super().__init__()
        self.domain_dis = nn.Sequential(
            GradientReversal(on=grad_rev),
            nn.Linear(c_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )


    def forward(self, c):

        return self.domain_dis(c)