import torch.nn as nn
import torch
class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *config, max_norm=1, **kwconfig):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*config, **kwconfig)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)