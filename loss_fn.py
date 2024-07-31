import torch as tc
from torch.nn.functional import mse_loss, huber_loss

getattr(tc.nn.functional, "l1_loss")(tc.randn([3,2,5]), tc.randn([3,2,5]))


class TraininngLoss(tc.nn.Module):
    def __init__(self, reduction="mean", channel_weights = tc.ones(3)/3, name=None):
        super().__init__()
        self.reduction = reduction
        self.channel_weights = channel_weights if channel_weights.sum()== 1 else channel_weights/channel_weights.sum()
        self.name = name

    def forward(self, input:tc.Tensor, target:tc.Tensor)-> tc.Tensor:
        loss_intercept = mse_loss(input[:,0], target[:,0], reduction=self.reduction)
        loss_coef = mse_loss(input[:,1], target[:,1], reduction=self.reduction)
        loss_stepsize = huber_loss(input[:,2], target[:,2], reduction=self.reduction)
        return loss_intercept*self.channel_weights[0] + loss_coef*self.channel_weights[1] + loss_stepsize*self.channel_weights[2]
        # tc.tensor([loss_intercept, loss_coef, loss_stepsize], requires_grad=True).matmul(self.channel_weights)
