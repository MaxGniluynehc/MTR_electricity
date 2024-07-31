import torch as tc
# from torch.nn.functional import mse_loss, huber_loss


class TraininngLoss(tc.nn.Module):
    def __init__(self, reduction="mean", channel_weights = tc.ones(3)/3, name=None, loss_fn_names=["mse_loss", "mse_loss", "huber_loss"]):
        super().__init__()
        self.reduction = reduction
        self.channel_weights = channel_weights if channel_weights.sum()== 1 else channel_weights/channel_weights.sum()
        self.name = name
        self.loss_fn_names = loss_fn_names

    def forward(self, input:tc.Tensor, target:tc.Tensor)-> tc.Tensor:
        loss_intercept = getattr(tc.nn.functional, self.loss_fn_names[0])(input[:,0], target[:,0], reduction=self.reduction)
        loss_coef = getattr(tc.nn.functional, self.loss_fn_names[1])(input[:,1], target[:,1], reduction=self.reduction)
        loss_stepsize = getattr(tc.nn.functional, self.loss_fn_names[2])(input[:,2], target[:,2], reduction=self.reduction)
        return loss_intercept*self.channel_weights[0] + loss_coef*self.channel_weights[1] + loss_stepsize*self.channel_weights[2]
        # tc.tensor([loss_intercept, loss_coef, loss_stepsize], requires_grad=True).matmul(self.channel_weights)
