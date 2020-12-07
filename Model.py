import math
# from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from GraphGAT import GraphGAT

class MGAT(torch.nn.Module):
    def __init__(self, features, edge_index, batch_size, num_user, num_item, dim_x=64):
        super(MGAT, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item

        self.edge_index = torch.tensor(edge_index).t().contiguous().cuda()
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1,0]]), dim=1)
        
        v_feat, a_feat, t_feat = features
        self.v_feat = torch.tensor(v_feat, dtype=torch.float).cuda()
        self.a_feat = torch.tensor(a_feat, dtype=torch.float).cuda()
        self.t_feat = torch.tensor(t_feat, dtype=torch.float).cuda()

        self.v_gnn = GNN(self.v_feat, self.edge_index, batch_size, num_user, num_item, dim_x, dim_latent=256)
        self.a_gnn = GNN(self.a_feat, self.edge_index, batch_size, num_user, num_item, dim_x, dim_latent=128)
        self.t_gnn = GNN(self.t_feat, self.edge_index, batch_size, num_user, num_item, dim_x, dim_latent=100)

        self.id_embedding = nn.Embedding(num_user+num_item, dim_x)
        nn.init.xavier_normal_(self.id_embedding.weight)

        #self.id_embedding = nn.init.xavier_normal_(torch.rand((num_user+num_item, dim_x), requires_grad=True)).cuda()
        self.result_embed = nn.init.xavier_normal_(torch.rand((num_user+num_item, dim_x))).cuda()

    def forward(self, user_nodes, pos_items, neg_items):
        v_rep = self.v_gnn(self.id_embedding)
        a_rep = self.a_gnn(self.id_embedding)
        t_rep = self.t_gnn(self.id_embedding)
        representation = (v_rep + a_rep + t_rep) / 3 #torch.max_pool2d((v_rep, a_rep, t_rep))#max()#torch.cat((v_rep, a_rep, t_rep), dim=1)
        self.result_embed = representation
        user_tensor = representation[user_nodes]
        pos_tensor = representation[pos_items]
        neg_tensor = representation[neg_items]
        pos_scores = torch.sum(user_tensor * pos_tensor, dim=1)
        neg_tensor = torch.sum(user_tensor * neg_tensor, dim=1)
        return pos_scores, neg_tensor


    def loss(self, data):
        user, pos_items, neg_items = data
        pos_scores, neg_scores = self.forward(user.cuda(), pos_items.cuda(), neg_items.cuda())
        loss_value = -torch.sum(torch.log2(torch.sigmoid(pos_scores - neg_scores)))
        return loss_value


    def accuracy(self, dataset, topk=10, neg_num=1000):
        all_set = set(list(np.arange(neg_num)))
        sum_pre = 0.0
        sum_recall = 0.0
        sum_ndcg = 0.0
        sum_item = 0
        # bar = tqdm(total=len(dataset))

        for data in dataset:
            # bar.update(1)
            if len(data) < 1002:
                continue
            sum_item += 1
            user = data[0]
            neg_items = data[1:1001]
            pos_items = data[1001:]

            batch_user_tensor = torch.tensor(user).cuda() 
            batch_pos_tensor = torch.tensor(pos_items).cuda()
            batch_neg_tensor = torch.tensor(neg_items).cuda()

            user_embed = self.result_embed[batch_user_tensor]
            pos_v_embed = self.result_embed[batch_pos_tensor]
            neg_v_embed = self.result_embed[batch_neg_tensor]

            num_pos = len(pos_items)
            pos_score = torch.sum(pos_v_embed*user_embed, dim=1)
            neg_score = torch.sum(neg_v_embed*user_embed, dim=1)

            _, index_of_rank_list = torch.topk(torch.cat((neg_score, pos_score)), topk)
            index_set = set([iofr.cpu().item() for iofr in index_of_rank_list])
            num_hit = len(index_set.difference(all_set))
            sum_pre += float(num_hit/topk)
            sum_recall += float(num_hit/num_pos)
            ndcg_score = 0.0
            for i in range(num_pos):
                label_pos = neg_num + i
                if label_pos in index_of_rank_list:
                    index = list(index_of_rank_list.cpu().numpy()).index(label_pos)
                    ndcg_score = ndcg_score + math.log(2) / math.log(index + 2)
            sum_ndcg += ndcg_score/num_pos
        # bar.close()

        return sum_pre/sum_item, sum_recall/sum_item, sum_ndcg/sum_item

class GNN(torch.nn.Module):
    def __init__(self, features, edge_index, batch_size, num_user, num_item, dim_id, dim_latent=None):
        super(GNN, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.dim_id = dim_id
        self.dim_feat = features.size(1)
        self.dim_latent = dim_latent
        self.edge_index = edge_index
        self.features = features

        self.preference = nn.Embedding(num_user, self.dim_latent)
        nn.init.xavier_normal_(self.preference.weight).cuda()
        if self.dim_latent:
            #self.preference = nn.init.xavier_normal_(torch.rand((num_user, self.dim_latent), requires_grad=True)).cuda()
            self.MLP = nn.Linear(self.dim_feat, self.dim_latent)

            self.conv_embed_1 = GraphGAT(self.dim_latent, self.dim_latent, aggr='add')
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_latent, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_latent, self.dim_id)    
            nn.init.xavier_normal_(self.g_layer1.weight) 
        else:
            #self.preference = nn.init.xavier_normal_(torch.rand((num_user, self.dim_feat), requires_grad=True)).cuda()
            self.conv_embed_1 = GraphGAT(self.dim_feat, self.dim_feat, aggr='add')
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_feat, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_feat, self.dim_id)    
            nn.init.xavier_normal_(self.g_layer1.weight)

        self.conv_embed_2 = GraphGAT(self.dim_id, self.dim_id, aggr='add')
        nn.init.xavier_normal_(self.conv_embed_2.weight)
        self.linear_layer2 = nn.Linear(self.dim_id, self.dim_id)
        nn.init.xavier_normal_(self.linear_layer2.weight)
        self.g_layer2 = nn.Linear(self.dim_id, self.dim_id)    
        nn.init.xavier_normal_(self.g_layer2.weight)

    def forward(self, id_embedding):
        temp_features = torch.tanh(self.MLP(self.features)) if self.dim_latent else self.features
        x = torch.cat((self.preference.weight, temp_features), dim=0)
        x = F.normalize(x).cuda()

        #layer-1
        h = F.leaky_relu(self.conv_embed_1(x, self.edge_index, None))
        x_hat = F.leaky_relu(self.linear_layer1(x)) + id_embedding.weight
        x_1 = F.leaky_relu(self.g_layer1(h)+x_hat)
        return x_1
        # layer-2
        h = F.leaky_relu(self.conv_embed_2(x_1, self.edge_index, None))
        x_hat = F.leaky_relu(self.linear_layer2(x_1)) + id_embedding.weight
        x_2 = F.leaky_relu(self.g_layer2(h)+x_hat)

        x = torch.cat((x_1, x_2), dim=1)

        return x