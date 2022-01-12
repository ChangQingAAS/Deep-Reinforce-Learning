import torch 
x = torch.tensor([1,2,6456556,54655])
mu = torch.tensor([x.argmax().item()])
print(mu)