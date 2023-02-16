import os
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils import data
import yaml
import argparse

class QueryDataset(Dataset):
    def __init__(self,cfg):
        self.data_path = os.path.join(cfg.data.root_dir, cfg.data.name)
        self.json_path = os.path.join(self.data_path, cfg.data.test_json)
        self.json = self.read_json(self.json_path)

        self.transform = self.get_quey_transform()
    def __getitem__(self, index):
        info = self.json[index]
        query_img = self.transform(Image.open(os.path.join(self.data_path, info['img'])).convert('RGB'))
        cat = info['category']

        return {'query_img':query_img, 'cat':cat}
    def __len__(self):
        return len(self.json)
    def read_json(self,path):
        with open(path, 'r') as f:
            tmp = json.loads(f.read())
        return tmp

    def get_quey_transform(self):
        transform_list = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        return transform_list


def load_query_dataset(cfg, batch_size, shuffle, num_workers, drop_last):
    dataset = QueryDataset(cfg=cfg)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                           drop_last=drop_last)
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

    retrieval_loader = load_query_dataset(cfg=config, batch_size=1, shuffle=False, num_workers=1,
                                                       drop_last=True)

    for meta in retrieval_loader:
        ##### dataset debug ######

            query_img = meta['query_img']
            cat = meta['cat']
            print(query_img)
            print(cat)

            trans = transforms.ToPILImage()
            p = trans(query_img[0])
            p.save('./a.png')


            break