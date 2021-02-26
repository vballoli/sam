from sam import SAM

import torch
from torchvision.models import resnet18

def test_sam():
  model = resnet18()
  optim = torch.optim.SGD(model.parameters(), 1e-3)
  optim = SAM(model.parameters(), optim)
  def closure():
    optim.zero_grad()
    loss = model(torch.randn(1,3,64,64)).sum()
    loss.backward()
    return loss
  optim.step(closure)