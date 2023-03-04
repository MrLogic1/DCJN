#
import os
import torch
from PIL import Image
import pickle
import json
import time
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils import data
import yaml
import argparse

# 1，先得到json文件路径 数据集根路径 + 数据集名字 + json名字
# 2，读取json
# 3，读取具体数据
# 3.1 读取数据集路径 根目录 + 数据集名字
# 3.2 根据json得到数据项名字
# 3。3 读取具体的数据 根目录 + 数据集名字 + 数据项名字
class Dataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.is_training = cfg.setting.is_training
        self.data_dir = os.path.join(cfg.data.root_dir, cfg.data.name)
        if cfg.setting.is_training:
            self.json_path = os.path.join(self.data_dir, cfg.data.training_json)
        else:
            self.json_path = os.path.join(self.data_dir, cfg.data.name, cfg.data.test_json)

        self.json_dict = self.read_json(self.json_path)

        crop_scale = (0.85, 0.95)
        self.aug = cfg.setting.is_aug
        self.query_transform = self.get_query_transform(cfg.data.pix_size, crop_scale, self.aug)
        self.rendering_transform = self.get_rendering_transform(cfg.data.pix_size, self.aug)


        render_path = os.path.join(self.data_dir, cfg.data.render_path)
        with open(render_path, 'rb') as f:
            self.dicts = pickle.load(f)

    def __getitem__(self, index):
        dic_one = {'bed': torch.tensor([1, 0, 0, 0], dtype=torch.float),
                   'chair': torch.tensor([0, 1, 0, 0], dtype=torch.float),
                   'sofa': torch.tensor([0, 0, 1, 0], dtype=torch.float),
                   'table': torch.tensor([0, 0, 0, 1], dtype=torch.float)}



        info = self.json_dict[index]
        cat = info['category']
        cat_ones = dic_one[cat]
        query_img = self.query_transform(Image.open(os.path.join(self.data_dir, info['img'])).convert("RGB"))


        info2 = self.json_dict[index]
        cat2 = info2['category']
        cats_ones2 = dic_one[cat2]
        instance = info2['model'].split('/')[-2]
        renderings = self.dicts[cat2][instance]

        render_img = torch.concat([self.rendering_transform(renderings[vi].convert("RGB")) for vi in range(self.cfg.data.view_num)], dim=0)
        render_img = render_img.reshape(self.cfg.data.view_num, 3, self.cfg.data.pix_size, self.cfg.data.pix_size)
        # render_img = torch.unsqueeze(render_img, dim=1)

        return {'query_img': query_img, 'cat_img': cat_ones, 'model': render_img, 'cat_model': cats_ones2}

    def __len__(self):
        return len(self.json_dict)


    def get_query_transform(self, rsize=(224, 224), crop_scale=(0.85, 0.95), is_aug=False):
        transform_list = []
        if is_aug:
            transform_list.append(transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=None, shear=None, fill=0))
            transform_list.append(transforms.RandomResizedCrop(rsize, scale=crop_scale))
            transform_list.append(transforms.RandomHorizontalFlip(p=0.1))

        transform_list += [transforms.ToTensor()]
        return transforms.Compose(transform_list)

    def get_rendering_transform(self, rsize=224, is_aug=False):
        transform_list = []
        if is_aug:
            transform_list.append(transforms.RandomResizedCrop(rsize, scale=(0.65, 0.9)))
            transform_list.append(transforms.RandomHorizontalFlip(p=0.1))

        transform_list += [transforms.ToTensor()]
        transform_list += [transforms.Normalize(0.5,0.5)]
        return transforms.Compose(transform_list)

    def read_json(self, mdir):
        with open(mdir, 'r') as f:
            tmp = json.loads(f.read())
        return tmp

def load_dataset(cfg,batch_size,shuffle,num_workers,drop_last):
    dataset = Dataset(cfg=cfg)
    return data.DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,drop_last=drop_last)


if __name__ == '__main__':



    with open('../configs/pix3d.yaml', 'r') as f:
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


    retrieval_loader = load_dataset(cfg=config,batch_size=5, shuffle=True, num_workers=1, drop_last=True)

    for meta in retrieval_loader:
        ##### dataset debug ######
        with torch.no_grad():
            cats = meta['cat_img']
            cats2 = meta['cat_model']
            query_img = meta['query_img']
            model = meta['model']
            print(model.size())
            # model = model.reshape(5,-1, 3 ,224, 224)
            print(cats)
            print(cats2)

            indict = cats * cats2
            indict = torch.sum(indict, dim=1)
            print(indict)
            print((-1)**indict)

            topil = transforms.ToPILImage()#将torch的张量形式改成image形式
            pic =  query_img[0]
            q_img = topil(pic)
            q_img.save('./q_img.png')

            for i in range(12):
                pic = model[0][i]
                m = topil(pic)
                m.save(f'./{i}.png')

            break



