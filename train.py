import os
import argparse
import numpy as np
# from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from DataLoad import DataLoad
from Model import MGAT


class Net:
    def __init__(self, args):
        ##########################################################################################################################################
        # seed = args.seed
        # np.random.seed(seed)
        # random.seed(seed)
        self.device = "cpu"#torch.device("cuda:1" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        ##########################################################################################################################################
        self.model_name = args.model_name
        self.data_path = args.data_path
        self.learning_rate = args.l_r
        self.weight_decay = args.weight_decay
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.num_epoch = args.num_epoch
        self.num_user = args.num_user
        self.num_item = args.num_item
        self.dim_latent = args.dim_latent
        self.dim_v = 2048
        self.dim_a = 128
        self.dim_t = 100
        print('args: ', args)

        ##########################################################################################################################################
        print('Loading data  ...')
        self.train_dataset = DataLoad(self.data_path, self.num_user, self.num_item)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                           num_workers=self.num_workers)

        self.edge_index = np.load(self.data_path + 'train.npy')
        self.val_dataset = np.load(self.data_path + 'val.npy')
        self.test_dataset = np.load(self.data_path + 'test.npy')

        self.v_feat = np.load(self.data_path + 'FeatureVideo_normal.npy')
        self.a_feat = np.load(self.data_path + 'FeatureAudio_avg_normal.npy')
        self.t_feat = np.load(self.data_path + 'FeatureText_stl_normal.npy')
        print('Data has been loaded.')
        ##########################################################################################################################################
        if self.model_name == 'MGAT':
            self.features = [self.v_feat, self.a_feat, self.t_feat]
            self.model = MGAT(self.features, self.edge_index, self.batch_size, self.num_user, self.num_item,
                               self.dim_latent).cuda()
        elif self.model_name == 'MMGCN':
            self.features = [self.v_feat, self.a_feat, self.t_feat]
            #self.model = MMGCN(self.features, self.edge_index, self.batch_size, self.num_user, self.num_item,
            #                   self.dim_latent).cuda()
        elif self.model_name == 'VBPR':
            self.dim_feat = self.dim_v + self.dim_a + self.dim_t
            self.features = torch.tensor(np.concatenate((self.v_feat, self.a_feat, self.t_feat), axis=1),
                                         dtype=torch.float)
            # self.model = VBPR_model(self.num_user, self.num_item, self.dim_latent, self.dim_feat, self.features).to(self.device)
        elif self.model_name == 'GraphSAGE':
            self.dim_feat = self.dim_v + self.dim_a + self.dim_t
            self.features = torch.tensor(np.concatenate((self.v_feat, self.a_feat, self.t_feat), axis=1),
                                         dtype=torch.float)
            # self.model = MMGraphSAGE(self.features, self.edge_index, self.batch_size, self.num_user, self.num_item, self.dim_latent, self.dim_feat).cuda()
        elif self.model_name == 'GCMC':
            self.dim_feat = self.dim_v + self.dim_a + self.dim_t
            self.features = torch.tensor(np.concatenate((self.v_feat, self.a_feat, self.t_feat), axis=1),
                                         dtype=torch.float)
            # self.model = GCMC(self.features, self.edge_index, self.batch_size, self.num_user, self.num_item, self.dim_latent, self.dim_feat).cuda()
        elif self.model_name == 'NGCF':
            self.dim_feat = self.dim_v + self.dim_a + self.dim_t
            self.features = torch.tensor(np.concatenate((self.v_feat, self.a_feat, self.t_feat), axis=1),
                                         dtype=torch.float)
            # self.model = NGCF(self.features, self.edge_index, self.batch_size, self.num_user, self.num_item, self.dim_latent, self.dim_feat).cuda()

        if args.PATH_weight_load and os.path.exists(args.PATH_weight_load):
            self.model.load_state_dict(torch.load(args.PATH_weight_load))
            print('module weights loaded....')
        ##########################################################################################################################################
        self.optimizer = torch.optim.Adam([{'params': self.model.parameters(), 'lr': self.learning_rate}],
                                          weight_decay=self.weight_decay)
        ##########################################################################################################################################

    def run(self):
        max_recall = 0.5330
        best_precision, best_recall, best_ndcg_score = 0., 0.0, 0.0
        for epoch in range(self.num_epoch):
            self.model.train()
            sum_loss = 0.0
            for data in self.train_dataloader:
                self.optimizer.zero_grad()
                self.loss = self.model.loss(data)

                self.loss.backward()
                self.optimizer.step()
                sum_loss += self.loss
            current_loss = sum_loss.item() / self.batch_size

            self.model.eval()
            with torch.no_grad():
                precision, recall, ndcg_score = self.model.accuracy(self.val_dataset)
            if recall > best_recall:
                best_precision, best_recall, best_ndcg_score = precision, recall, ndcg_score
            print(
                '{0}-th Loss:{1:.4f} Precision:{2:.4f} Recall:{3:.4f} NDCG:{4:.4f} / Best:Precision:{5:.4f} Recall:{6:.4f} NDCG:{7:.4f}'.format(
                    epoch, current_loss, precision, recall, ndcg_score, best_precision, best_recall, best_ndcg_score))

            if args.PATH_weight_save and recall > max_recall:
                max_recall = recall
                torch.save(self.model.state_dict(), args.PATH_weight_save)
                print('module weights saved....')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--model_name', default='MGAT', help='Model name.')
    parser.add_argument('--data_path', default='movielens/', help='Dataset path')
    parser.add_argument('--PATH_weight_load', default=None, help='Loading weight filename.')
    parser.add_argument('--PATH_weight_save', default=None, help='Writing weight filename.')
    parser.add_argument('--l_r', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay.')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size.')
    parser.add_argument('--dim_latent', type=int, default=64, help='Latent dimension.')
    parser.add_argument('--num_epoch', type=int, default=200, help='Epoch number')
    parser.add_argument('--num_workers', type=int, default=40, help='Workers number')
    parser.add_argument('--num_user', type=int, default=55485, help='User number')
    parser.add_argument('--num_item', type=int, default=5986, help='Item number')
    args = parser.parse_args()

    mgat = Net(args)
    mgat.run()
