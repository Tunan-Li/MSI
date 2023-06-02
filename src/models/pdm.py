# coding: utf-8
# @email: enoche.chow@gmail.com
# version 3 conduct information bottleneck between three views
r"""

################################################
"""
import os
import copy
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
from torch.distributions import Normal, Independent
from torch.nn.functional import softplus
from common.abstract_recommender import GeneralRecommender
from common.loss import EmbLoss

class MIEstimator(nn.Module):
    def __init__(self, size1, size2):
        super(MIEstimator, self).__init__()

        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(size1 + size2, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1),
        )

    # Gradient for JSD mutual information estimation and EB-based estimation
    def forward(self, x1, x2):
        pos = self.net(torch.cat([x1, x2], 1))  # Positive Samples
        neg = self.net(torch.cat([torch.roll(x1, 1, 0), x2], 1))
        return -softplus(-pos).mean() - softplus(neg).mean(), pos.mean() - neg.exp().mean() + 1


def rand_projections(dim, num_projections=1000):
    projections = torch.randn((num_projections, dim))
    projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))
    return projections


class PDM(GeneralRecommender):
    def __init__(self, config, dataset):
        super(PDM, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['embedding_size']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.cl_weight = config['cl_weight']
        self.dropout = config['dropout']

        self.mib_weight = config['mib']
        self.beta = config['beta']
        
        print(self.beta)
        print(self.mib_weight)

        self.n_nodes = self.n_users + self.n_items
        self.mi_estimator = MIEstimator(64, 64)
        # load dataset info
        self.norm_adj = self.get_norm_adj_mat(dataset.inter_matrix(form='coo').astype(np.float32)).to(self.device)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim * 2)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.predictor = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.reg_loss = EmbLoss()

        nn.init.xavier_normal_(self.predictor.weight)

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim * 2)
            nn.init.xavier_normal_(self.image_trs.weight)
            
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim * 2)
            nn.init.xavier_normal_(self.text_trs.weight)
            

    def get_norm_adj_mat(self, interaction_matrix):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = interaction_matrix
        inter_M_t = interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))
    
    def forward(self):
        h = self.item_id_embedding.weight[:, :64]

        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight[:, :64]), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

        self.image_emb = self.image_trs(self.image_embedding.weight)[:, :64]
        self.text_emb = self.text_trs(self.text_embedding.weight)[:, :64]
        
        # mu_text, sigma_text = self.text_emb[:, :64], self.text_emb[:, 64:]
        # sigma_text = softplus(sigma_text) + 1e-7  # Make sigma always positive
        # p_z1_given_v1 = Independent(Normal(loc=mu_text, scale=sigma_text), 1)

        # self.text_emb_encoded = p_z1_given_v1.mean

        # mu_image, sigma_image = self.image_emb[:, :64], self.image_emb[:, 64:]
        # sigma_image = softplus(sigma_image) + 1e-7  # Make sigma always positive
        # p_z2_given_v2 = Independent(Normal(loc=mu_image, scale=sigma_image), 1)
        
        # self.image_emb_encoded = p_z2_given_v2.mean
        
        ego_embeddings_text = torch.cat((self.user_embedding.weight, self.text_emb), dim=0)
        all_embeddings_text = [ego_embeddings_text]
        for i in range(self.n_layers):
            ego_embeddings_text = torch.sparse.mm(self.norm_adj, ego_embeddings_text)
            all_embeddings_text += [ego_embeddings_text]
        all_embeddings_text = torch.stack(all_embeddings_text, dim=1)
        all_embeddings_text = all_embeddings_text.mean(dim=1, keepdim=False)
        u_g_embeddings_text, i_g_embeddings_text = torch.split(all_embeddings_text, [self.n_users, self.n_items], dim=0)
        
        ego_embeddings_image = torch.cat((self.user_embedding.weight, self.image_emb), dim=0)
        all_embeddings_image = [ego_embeddings_image]
        for i in range(self.n_layers):
            ego_embeddings_image = torch.sparse.mm(self.norm_adj, ego_embeddings_image)
            all_embeddings_image += [ego_embeddings_image]
        all_embeddings_image = torch.stack(all_embeddings_image, dim=1)
        all_embeddings_image = all_embeddings_image.mean(dim=1, keepdim=False)
        u_g_embeddings_image, i_g_embeddings_image = torch.split(all_embeddings_image, [self.n_users, self.n_items], dim=0)


        return u_g_embeddings, i_g_embeddings + h, u_g_embeddings_text, i_g_embeddings_text + self.text_emb, u_g_embeddings_image, i_g_embeddings_image + self.image_emb

    def calculate_loss(self, interactions):
        # online network
        u_online_ori, i_online_ori, _, t_feat_online, _, v_feat_online = self.forward()
        reg_multimodal_loss = self.reg_loss(t_feat_online, v_feat_online)

        with torch.no_grad():
            u_target, i_target = u_online_ori.clone(), i_online_ori.clone()
            u_target.detach()
            i_target.detach()
            u_target = F.dropout(u_target, self.dropout)
            i_target = F.dropout(i_target, self.dropout)

            if self.t_feat is not None:
                t_feat_target = t_feat_online.clone()
                t_feat_target = F.dropout(t_feat_target, self.dropout)

            if self.v_feat is not None:
                v_feat_target = v_feat_online.clone()
                v_feat_target = F.dropout(v_feat_target, self.dropout)

        u_online, i_online = self.predictor(u_online_ori), self.predictor(i_online_ori)

        users, items = interactions[0], interactions[1]
        
        #############
        v1 = self.text_embedding.weight[items, :]  # 2048 384
        v2 = self.image_embedding.weight[items, :]  # 2048 4096
        v3 = self.item_id_embedding.weight[items, :]
        # params_text = self.text_trs(v1)  # 2048 * 128
        # params_image = self.image_trs(v2)  # 2048 * 128

        mu_v1, sigma_v1 = self.text_trs(v1)[:, :64], self.text_trs(v1)[:, 64:]
        mu_v2, sigma_v2 = self.image_trs(v2)[:, :64], self.image_trs(v2)[:, 64:]
        mu_v3, sigma_v3 = v3[:, :64], v3[:, 64:]

        sigma_v1 = softplus(sigma_v1) + 1e-7  # Make sigma always positive
        sigma_v2 = softplus(sigma_v2) + 1e-7
        sigma_v3 = softplus(sigma_v3) + 1e-7
        
        p_z1_given_v1 = Independent(Normal(loc=mu_v1, scale=sigma_v1), 1)
        p_z2_given_v2 = Independent(Normal(loc=mu_v2, scale=sigma_v2), 1)
        p_z3_given_v3 = Independent(Normal(loc=mu_v3, scale=sigma_v3), 1)

        z_v1 = p_z1_given_v1.rsample()
        z_v2 = p_z2_given_v2.rsample()
        z_v3 = p_z3_given_v3.rsample()

        # view 1, 2
        mi_gradient_12, _ = self.mi_estimator(z_v1, z_v2)
        mi_gradient_12 = mi_gradient_12.mean()
        # mi_estimation = mi_estimation.mean()

        kl_1_2 = p_z1_given_v1.log_prob(z_v1) - p_z2_given_v2.log_prob(z_v1)
        kl_2_1 = p_z2_given_v2.log_prob(z_v2) - p_z1_given_v1.log_prob(z_v2)
        skl_12 = (kl_1_2 + kl_2_1).mean() / 2.

        loss_ib_12 = - mi_gradient_12 + self.beta * skl_12

        # view 1, 3
        mi_gradient_13, _ = self.mi_estimator(z_v1, z_v3)
        mi_gradient_13 = mi_gradient_13.mean()
        # mi_estimation = mi_estimation.mean()

        kl_1_3 = p_z1_given_v1.log_prob(z_v1) - p_z3_given_v3.log_prob(z_v1)
        kl_3_1 = p_z3_given_v3.log_prob(z_v3) - p_z1_given_v1.log_prob(z_v3)
        skl_13 = (kl_1_3 + kl_3_1).mean() / 2.

        loss_ib_13 = - mi_gradient_13 + self.beta * skl_13

        # view 2, 3
        mi_gradient_23, _ = self.mi_estimator(z_v2, z_v3)
        mi_gradient_23 = mi_gradient_23.mean()
        # mi_estimation = mi_estimation.mean()

        kl_2_3 = p_z2_given_v2.log_prob(z_v2) - p_z3_given_v3.log_prob(z_v2)
        kl_3_2 = p_z3_given_v3.log_prob(z_v3) - p_z2_given_v2.log_prob(z_v3)
        skl_23 = (kl_2_3 + kl_3_2).mean() / 2.

        loss_ib_23 = - mi_gradient_23 + self.beta * skl_23



        # loss_ib = (loss_ib_12 + loss_ib_13 + loss_ib_23) / 3
        loss_ib = loss_ib_12
        #############
        
        # swd_loss = self.sliced_wasserstein_distance(self.image_emb[items, :], self.text_emb[items, :])
        
        u_online = u_online[users, :]
        i_online = i_online[items, :]
        u_target = u_target[users, :]
        i_target = i_target[items, :]

        loss_t, loss_v, loss_tv, loss_vt = 0.0, 0.0, 0.0, 0.0
        if self.t_feat is not None:
            t_feat_online = self.predictor(t_feat_online)
            t_feat_online = t_feat_online[items, :]
            t_feat_target = t_feat_target[items, :]
            loss_t = 1 - cosine_similarity(t_feat_online, i_target.detach(), dim=-1).mean()
            loss_tv = 1 - cosine_similarity(t_feat_online, t_feat_target.detach(), dim=-1).mean()
        if self.v_feat is not None:
            v_feat_online = self.predictor(v_feat_online)
            v_feat_online = v_feat_online[items, :]
            v_feat_target = v_feat_target[items, :]
            loss_v = 1 - cosine_similarity(v_feat_online, i_target.detach(), dim=-1).mean()
            loss_vt = 1 - cosine_similarity(v_feat_online, v_feat_target.detach(), dim=-1).mean()

        loss_ui = 1 - cosine_similarity(u_online, i_target.detach(), dim=-1).mean()
        loss_iu = 1 - cosine_similarity(i_online, u_target.detach(), dim=-1).mean()

        # return (loss_ui + loss_iu).mean() + self.reg_weight * (self.reg_loss(u_online_ori, i_online_ori) + reg_multimodal_loss) + \
        #        self.cl_weight * (loss_t + loss_v + loss_tv + loss_vt).mean() + self.mib_weight * loss_ib
        # return (loss_ui + loss_iu).mean() + self.reg_weight * (self.reg_loss(u_online_ori, i_online_ori) + reg_multimodal_loss) + \
        #        self.cl_weight * (loss_t + loss_v).mean() + self.mib_weight * loss_ib
        return (loss_ui + loss_iu).mean() + self.reg_weight * (self.reg_loss(u_online_ori, i_online_ori) + reg_multimodal_loss) + \
               self.cl_weight * (loss_tv + loss_vt).mean() + self.mib_weight * loss_ib

    def full_sort_predict(self, interaction):
        user = interaction[0]
        u_online, i_online, _, _, _, _ = self.forward()
        u_online, i_online = self.predictor(u_online), self.predictor(i_online)
        score_mat_ui = torch.matmul(u_online[user], i_online.transpose(0, 1))
        return score_mat_ui