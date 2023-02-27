import os

import torch
import tqdm
from Net import ModelNet
from Net import ImgNet
from dataset import Dataset
from dataset import ShapeDataset
from dataset import QueryDataset
from torch import nn
import yaml
import argparse

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

class DCJN():
    def __init__(self, config):
        self.config = config
        lr = 0.00005
        beta1 = 0.5
        beta2 = 0.999
        self.img_net = ImgNet().to(device=device)
        self.model_net = ModelNet(self.config.modelnet.input_dim, self.config.modelnet.hidden_dim, self.config.modelnet.num_layers, self.config.modelnet.bat_first).to(device=device)
        self.loss = nn.CrossEntropyLoss(reduction='none')
        # loss = nn.CrossEntropyLoss()
        self.optimal_img = torch.optim.Adam(self.img_net.parameters(), lr=lr, betas=(beta1, beta2))
        self.optimal_model = torch.optim.Adam(self.model_net.parameters(), lr=lr, betas=(beta1, beta2))
        self.best_acc = 0
        self.epoch = 0
        self.it = 0

    def train(self):

        retrieval_loader = Dataset.load_dataset(cfg=self.config, batch_size=self.config.data.batch_size, shuffle=True,
                                                           num_workers=self.config.data.num_workers,
                                                           drop_last=True)


        epochs = config.trainer.epochs
        for epoch in range(epochs):
            self.img_net.train()
            self.model_net.train()
            pbar = tqdm.tqdm(retrieval_loader)
            for meta in pbar:
            ##### dataset debug ######
            # with torch.no_grad():
                info_dict = {}
                query_img = meta['query_img'].to(device=device)
                cats_img = meta['cat_img'].to(device=device)
                model= meta['model'].to(device=device)
                model = model.reshape(len(query_img), -1, 3, self.config.data.pix_size, self.config.data.pix_size)
                cats_model = meta['cat_model'].to(device=device)

                self.optimal_img.zero_grad()
                self.optimal_model.zero_grad()

                ri, ci= self.img_net(query_img)
                rm, cm = self.model_net(model)

                l_di = self.loss(ci, cats_img)
                l_di = l_di.sum()
                l_dm = self.loss(cm, cats_model)
                l_dm = l_dm.sum()
                # a = torch.exp(torch.sum(ci * cats_img, dim=1)) / torch.sum(torch.exp(ci), dim=1)
                # b = torch.exp(torch.sum(cm * cats_model, dim=1)) / torch.sum(torch.exp(cm), dim=1)


                # l1 = torch.sum(-torch.log(a), dim = 0)
                # l2 = torch.sum(-torch.log(b), dim=0)

                indict = cats_img * cats_model
                indict = torch.sum(indict, dim = 1)
                indict = (-1)**indict
                l_cross =torch.sum(indict * torch.sum((ri - rm)**2, dim=1), dim = 0)


                if  epoch < 0:
                    l_di.backward()
                    l_dm = l_dm.sum()
                    l_dm.backward()

                    self.optimal_img.step()
                    self.optimal_model.step()
                else :
                    l_img = 0.8 * (l_di.sum()) + 0.2 * l_cross
                    l_img.backward(retain_graph=True)

                    l_model = 0.8 * (l_dm.sum()) + 0.2 * l_cross
                    l_model.backward()

                    self.optimal_img.step()
                    self.optimal_model.step()
                    loss_img = l_img.item()
                    loss_model = l_model.item()
                    info_dict['loss_img'] = '%.5f' % (loss_img)
                    info_dict['loss_model'] = '%.5f' % (loss_model)
                loss_di = l_di.item()
                loss_dm = l_dm.item()

                # print(loss_item)

                info_dict['loss_di'] = '%.5f'% (loss_di)
                info_dict['loss_dm'] = '%.5f'%(loss_dm)
                pbar.set_postfix(info_dict)
                pbar.set_description('Epoch: %d' % (epoch))

                self.it += 1
            if epoch >= 1 :
                self.test()
            self.epoch = epoch
    def test(self):
        cfg = self.config
        self.img_net.eval()
        self.model_net.eval()
        shape_loader = ShapeDataset.load_shape_dataset(cfg,batch_size=cfg.data.batch_size,shuffle=False, num_workers=cfg.data.num_workers, drop_last=False)

        shape_cats_list = []
        shape_ebd_list = []

        pbar = tqdm.tqdm(shape_loader)
        for meta in pbar:
            with torch.no_grad():
                rendering_img = meta['rendering_img'].to(device=device)
                cats = meta['labels']['cat']

                rendering = rendering_img.reshape(len(cats), -1, 3, self.config.data.pix_size, self.config.data.pix_size)
                rendering_ebds, _ = self.model_net(rendering)
                shape_cats_list += cats
                shape_ebd_list.append(rendering_ebds)
        shape_ebd = torch.concat(shape_ebd_list, dim=0)

        query_loader = QueryDataset.load_query_dataset(cfg,batch_size = 1,shuffle=False, num_workers=cfg.data.num_workers, drop_last=False)

        total_dict = {}
        acc_cats_dict = {}
        with torch.no_grad():
            pbar = tqdm.tqdm(query_loader)
            for meta in pbar:
                query_img = meta['query_img'].to(device=device)
                gt_cat = meta['cat'][0]

                query_ebd, _ = self.img_net(query_img)
                query_ebd = query_ebd.repeat(shape_ebd.shape[0], 1)
                same_score = 1 / (torch.sum((query_ebd - shape_ebd)**2, dim=1))

                max_id = same_score.argmax()
                pr_cat = shape_cats_list[max_id.item()]

                try:
                    total_dict[gt_cat] = total_dict[gt_cat] + 1
                except:
                    total_dict[gt_cat] = 1
                if gt_cat == pr_cat:
                    try:
                        acc_cats_dict[gt_cat] = acc_cats_dict[gt_cat] + 1
                    except:
                        acc_cats_dict[gt_cat] = 1

        total_num = 0
        acc_num = 0
        out_info = []
        for keys in total_dict.keys():
            num = total_dict[keys]
            try:
                cats_num = acc_cats_dict[keys]
            except:
                cats_num = 0
            total_num += num
            acc_num += cats_num
            out_info.append('%s: cats: %d, total: %d\n' %(keys, cats_num, num) )

        out_infos = ''.join(out_info)
        print(out_infos)

        final_acc = acc_num / total_num

        print(final_acc)


        if final_acc > self.best_acc:
            self.best_acc = final_acc
            paths_list=cfg.models.pre_trained_path.split('/')[:-1]
            paths_list.append(cfg.data.name+'.pt')
            paths = '/'.join(paths_list)
            self.saving(paths=paths)

        print('best acc: %.3f' %(self.best_acc, ))

    def saving(self, paths=None):
        cfg = self.config

        if paths == None:
            save_name = "epoch_{}_iter_{}.pt".format(self.epoch, self.it)
            save_path = os.path.join(cfg.save_dir, save_name)
            print('models %s saved!\n' % (save_name,))
        else:
            save_path = paths
            print('model paths %s saved!\n' % (paths,))

        torch.save(self.img_net.state_dict(), save_path)
        torch.save(self.model_net.state_dict(), save_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="./configs/pix3d.yaml", help="Path to (.yaml) config file."
    )

    configargs = parser.parse_args()
    with open(configargs.config, 'r') as f:
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
    d = DCJN(config)
    d.train()
