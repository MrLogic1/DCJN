import torch
import torchvision.models as models
from torch import nn
from torchvision.io import read_image
from dataset import Dataset

class backbone():
    def __init__(self):
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'

        self.model = models.vgg16(weights=None).to(device=device)
        pre = torch.load('./backbone/vgg16-397923af.pth')
        self.model.load_state_dict(pre)
        self.model.classifier = nn.Sequential(*[self.model.classifier[i] for i in range(4)])
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model = self.model.eval()
    def get(self,x):
        out = self.model(x)
        return out



class ImgNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = backbone()
        # 修改参数并添加层
        self.hidden = nn.Linear(4096, 4096)

        self.Relu = nn.ReLU(inplace=True)
        self.drop_out = nn.Dropout(p=0.5, inplace=False)

        self.hidden2 = nn.Linear(4096, 100)

        self.hidden3 = nn.Linear(100, 4)


    def forward(self, X):
        input =self.backbone.get(X)

        out1 = self.hidden(input)
        out1 =  self.Relu(out1)
        out1 = self.drop_out(out1)

        out2 = self.hidden2(out1)
        out2 = self.Relu(out2)
        out2 = self.drop_out(out2)

        out3 = self.hidden3(out2)
        return out2, out3

class ModelNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, batch_first):
        super().__init__()
        self.backbone = backbone()
        self.model = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first = batch_first)
        self.Relu = nn.ReLU(inplace=True)
        self.drop_out = nn.Dropout(p=0.5,inplace=False)
        self.hidden = nn.Linear(4096, 100)
        self.hidden2 = nn.Linear(100, 4)
    def forward(self,X):
        X = X.reshape(-1,3,224,224)
        input = self.backbone.get(X)
        input = input.reshape(-1, 12, 4096)
        _, (out1, _), = self.model(input)

        out1 = out1[0]

        out2 = self.hidden(out1)
        out2 = self.Relu(out2)
        out2 = self.drop_out(out2)

        out3 = self.hidden2(out2)

        return out2, out3


class Net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ImgNet = ImgNet()
        self.ModelNet = ModelNet(cfg.modelnet.input_dim, cfg.modelnet.hidden_dim, cfg.modelnet.num_layers, cfg.modelnet.bat_first)
    def forward(self,Image,Model):
        r_i,c_i = self.get_Img_ebd(Image)
        r_m,c_m = self.get_model_ebd(Model)
        return r_i,c_i,r_m,c_m

    def get_Img_ebd(self, Image):
        return self.ImgNet(Image)

    def get_model_ebd(self, Model):
        return self.ModelNet(Model)