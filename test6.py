import torch
import torchvision.models as models
from torch import nn

def backbone(x):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model = models.vgg16(weights=None).to(device=device)
    pre = torch.load('./backbone/vgg16-397923af.pth')
    model.load_state_dict(pre)
    model.classifier = nn.Sequential(*[model.classifier[i] for i in range(4)])
    model = model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model(x)

if __name__=='__main__':
    a = torch.randn(1,3,224,224).to(device='cuda')

    p = backbone(a)

    print(p.size())



    # p = model(a)
    # print(p)


    # print(my_model)

