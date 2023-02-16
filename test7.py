import torch
import yaml
import argparse
from dataset import QueryDataset
from torchvision import  transforms


if __name__ == '__main__':
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

    retrieval_loader = QueryDataset.load_query_dataset(cfg=config, batch_size=1, shuffle=False, num_workers=1,
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