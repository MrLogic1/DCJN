import torch
from torch import nn
from torch.nn import functional as F

class Img_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(100,50)
        self.hidden2 = nn.Linear(50,10)
        self.out = nn.Linear(10,5)
    def forward(self,X):
        h1_out = self.hidden(X)
        h1_activate_out = F.relu(h1_out)
        h2_out = self.hidden2(h1_activate_out)
        h2_activate_out = F.relu(h2_out)
        out = self.out(h2_activate_out)
        return h2_out,out

class Model_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(100,50)
        self.hidden2 = nn.Linear(50, 10)
        self.out = nn.Linear(10,5)
    def forward(self,X):
        h1_out = self.hidden(X)
        h1_activate_out = F.relu(h1_out)
        h2_out = self.hidden2(h1_activate_out)
        h2_activate_out = F.relu(h2_out)
        out = self.out(h2_activate_out)
        return h2_out,out

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.ImgNet = Img_Net()
        self.ModelNet = Model_Net()
    def forward(self,Image,Model):
        r_i,c_i = self.ImgNet(Image)
        r_m,c_m = self.ModelNet(Model)
        return r_i,c_i,r_m,c_m

if __name__=='__main__':
    a = torch.randn((10,100))
    net = Img_Net()
    out,_ = net(a)
    print(out.size())