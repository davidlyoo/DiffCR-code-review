import torch.nn as nn

class BaseNetwork(nn.Module):
    def __init__(self, init_type='kaiming', gain=0.02):
        super(BaseNetwork, self).__init__()
        self.init_type = init_type
        self.gain = gain

    def init_weights(self):
        