from Net import Net
import torch
import os
import tqdm
import argparse
import yaml
from dataset import ShapeDataset
from dataset import QueryDataset

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
class DCJNTest:
    def __init__(self, config):
        self.cfg = config
        self.net = Net(cfg = config).to(device=device)
        self.loading(self.cfg.models.pre_trained_path)
    def loading(self, paths=None):
        if paths == None or not os.path.exists(paths):
            print('No ckpt!')
            exit(-1)
        else:
            # loading
            ckpt = torch.load(paths)
            self.net.load_state_dict(ckpt)
            print('loading %s successfully' %(paths))
    def test(self):
        cfg = self.cfg
        self.net.eval()
        shape_loader = ShapeDataset.load_shape_dataset(cfg,batch_size=cfg.data.batch_size,shuffle=False, num_workers=cfg.data.num_workers, drop_last=False)

        shape_cats_list = []
        shape_ebd_list = []

        pbar = tqdm.tqdm(shape_loader)
        for meta in pbar:
            with torch.no_grad():
                rendering_img = meta['rendering_img'].to(device=device)
                cats = meta['labels']['cat']
                instance = meta['labels']['instance']

                rendering = rendering_img.reshape(len(cats), -1, 3, self.cfg.data.pix_size, self.cfg.data.pix_size)
                rendering_ebds, _ = self.net.get_model_ebd(rendering)
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

                query_ebd, _ = self.net.get_Img_ebd(query_img)
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

        print(f'最终的正确率：{final_acc}')

if __name__=='__main__':
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

    D = DCJNTest(config)
    D.test()