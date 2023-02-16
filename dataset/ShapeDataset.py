import os
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils import data
from PIL import Image
import yaml
import argparse
import pickle
import torch


class ShapeDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        render_path = os.path.join(cfg.data.root_dir, cfg.data.name, cfg.data.render_path)
        with open(render_path, 'rb') as f:
            self.dicts = pickle.load(f)
        self.labels = self.make_dataset(self.dicts)

        self.transform = self.get_transform()
        self.view_num = cfg.data.view_num


    def __getitem__(self, index):
        labels = self.labels[index]
        cat = labels['cat']
        idx = labels['instance']
        renderings = self.dicts[cat][idx] # 12x224x224
        # debug = self.transform(Image.fromarray(renderings[0]))
        render_img = torch.concat([self.transform(renderings[vi].convert("RGB")) for vi in range(self.view_num)], dim=0)
        render_img = render_img.reshape(self.cfg.data.view_num, 3, self.cfg.data.pix_size, self.cfg.data.pix_size)
        return {'rendering_img': render_img, 'labels': labels}

    def __len__(self):
        return len(self.labels)

    def make_dataset(self,dicts):
        labels = []
        for cat in dicts.keys():
            for idx in dicts[cat].keys():
                labels.append({'cat': cat, 'instance': idx})
        return labels


    def get_transform(self):
        transform_list = []


        transform_list += [transforms.ToTensor()]
        transform_list += [transforms.Normalize(0.5,0.5)]
        return transforms.Compose(transform_list)


def load_shape_dataset(cfg, batch_size, shuffle, num_workers, drop_last):
    dataset = ShapeDataset(cfg=cfg)
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

    retrieval_loader = load_shape_dataset(cfg=config, batch_size=1, shuffle=False, num_workers=1,
                                                       drop_last=True)

    for meta in retrieval_loader:
        ##### dataset debug ######

            rendering_img = meta['rendering_img']
            cat = meta['labels']
            print(rendering_img.size())
            print(cat)

            trans = transforms.ToPILImage()
            for i in range(12):
                p = trans(rendering_img[0][i])
                p.save(f'./a{i}.png')


            break