import torch
import torchvision.models as models
from torch import nn
from torchvision.io import read_image

from dataset.dataset import QueryDataset

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


class model(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载模型预训练参数
        self.weights = None
        self.model = models.vgg16(weights=None)
        pre = torch.load('./backbone/vgg16-397923af.pth')
        self.model.load_state_dict(pre)

        # 冻结参数
        for param in self.model.parameters():
            param.requires_grad_(False)

        # 修改参数并添加层
        self.model.classifier[6] = nn.Linear(4096, 1000)

        self.leakyRelu = nn.LeakyReLU(inplace=True)
        self.drop_out = nn.Dropout(p=0.5, inplace=False)
        self.hidden = nn.Linear(1000, 4)

    def forward(self, X):
        out1 = self.model(X)

        out2 = self.leakyRelu(out1)
        out2 = self.drop_out(out2)

        out3 = self.hidden(out2)
        return out1, out3

def train():
    import yaml
    import argparse
    import tqdm

    with open('./configs/pix3d.yaml', 'r') as f:
        config = yaml.safe_load(f)


    def dict2namespace(config):
        namespace = argparse.Namespace()
        for key, value in config.items():
            if isinstance(value, dict):
                new_value = dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace


    config = dict2namespace(config)

    retrieval_loader = QueryDataset.load_query_dataset(cfg=config, batch_size=5, shuffle=True,
                                                       num_workers=1,
                                                       drop_last=True)

    lr = 0.00005
    beta1 = 0.5
    beta2 = 0.999
    net = model().to(device=device)
    loss = nn.CrossEntropyLoss()
    optimal = torch.optim.Adam(net.parameters(),lr=lr,betas=(beta1,beta2))

    epochs = config.trainer.epochs
    for epoch in range(epochs):

        pbar = tqdm.tqdm(retrieval_loader)
        for meta in pbar:
        ##### dataset debug ######
        # with torch.no_grad():
            mask_img = meta['mask_img'].to(device=device)
            embeddings = meta['rendering_img'].to(device=device)
            cats = meta['cat'].to(device=device)
            instances = meta['instance']
            query_img = meta['query_img'].to(device=device)


            # print(cats)
            optimal.zero_grad()
            x1, x2 = net(query_img)
            l = loss(x2, cats)
            l.backward()
            optimal.step()
            loss_item = l.item()
            # print(loss_item)

            info_dict = {'loss': '%.5f' % (loss_item)}
            pbar.set_postfix(info_dict)
            pbar.set_description('Epoch: %d' % (epoch))


if __name__ =='__main__':
    train()