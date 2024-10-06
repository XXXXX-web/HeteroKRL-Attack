'''
    Adversarial Attacks on Neural Networks for Graph Data. ICML 2018.
        https://arxiv.org/abs/1806.02371
    Author's Implementation
       https://github.com/Hanjun-Dai/graph_adversarial_attack
    This part of code is adopted from the author's implementation (Copyright (c) 2018 Dai, Hanjun and Li, Hui and Tian, Tian and Huang, Xin and Wang, Lin and Zhu, Jun and Song, Le) but modified
    to be integrated into the repository.
'''
import os
import sys
import numpy as np
import torch
import networkx as nx
import random
from torch.nn.parameter import Parameter
import torch.nn as nn
from HAN.model import HAN
from env import GraphNormTool,StaticGraph
import torch.nn.functional as F

class TypeNet(nn.Module):

    def __init__(self, node_features, classification_features, list_action_space, embed_dim=64, mlp_hidden=64, max_lv=1, num_heads = 1, dropout=0.1, gm='mean', device='cpu'):
        '''
        bilin_q: bilinear q or not
        mlp_hidden: mlp hidden layer size
        mav_lv: max rounds of message passing
        '''
        super(TypeNet, self).__init__()
        self.node_features = node_features
        self.classification_features = classification_features
        self.list_action_space = list_action_space
        self.total_nodes = len(list_action_space)

        self.embed_dim = embed_dim
        self.mlp_hidden = mlp_hidden
        self.max_lv = max_lv
        self.gm = gm
        self.region_length = len(self.list_action_space[0][0])


        last_wout = embed_dim
        self.bias_target = Parameter(torch.Tensor(1, embed_dim))

        if mlp_hidden:
            self.linear_1 = nn.Linear(embed_dim * 2, mlp_hidden)
            self.linear_out = nn.Linear(mlp_hidden, last_wout)
        else:
            self.linear_out = nn.Linear(embed_dim * 2, last_wout)

        self.w_n2l = Parameter(torch.Tensor(node_features.size()[1], embed_dim))
        self.bias_n2l = Parameter(torch.Tensor(embed_dim))
        self.bias_picked = Parameter(torch.Tensor(1, embed_dim))
        self.conv_params = nn.Linear(embed_dim, embed_dim)
        self.norm_tool = GraphNormTool(normalize=True, gm=self.gm, device=device)
        self.HANModel = HAN(meta_paths=StaticGraph.metapaths,
                            in_size=self.classification_features.shape[1],
                            hidden_size=mlp_hidden,
                            out_size=self.embed_dim,
                            num_heads=[num_heads],
                            dropout=dropout).to(device)
        self.query_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_layer = nn.Linear(embed_dim, self.region_length)
        weights_init(self)

    def update_list_action_space(self, list_action_space):
        self.list_action_space = list_action_space

    def update_node_features_with_han(self, graph, classification_features, node_features):
        classification_node_embedding = self.HANModel(graph, classification_features)
        # node_features[:classification_node_embedding.shape[0]] = classification_node_embedding
        updated_node_features = node_features.clone()
        updated_node_features[:classification_node_embedding.shape[0]] = classification_node_embedding
        return updated_node_features


    def forward(self, state, is_inference=False):
        target_node, graph_info = state
        with torch.set_grad_enabled(mode=not is_inference):
            graph = graph_info.get_modified_heterograph(self.node_features.device)
            input_node_linear = self.update_node_features_with_han(graph, self.classification_features, self.node_features)
            if input_node_linear.data.is_sparse:
                input_node_linear = torch.spmm(input_node_linear, self.w_n2l) # 2708*64
            else:
                input_node_linear = torch.mm(input_node_linear, self.w_n2l)
            input_node_linear += self.bias_n2l
            region = self.list_action_space[target_node]
            node_embed = F.relu(input_node_linear)
            target_embed = node_embed[target_node,:].unsqueeze(0)
            graph_embed = torch.mean(node_embed, dim=0, keepdim=True)

            query = self.query_proj(torch.cat((target_embed, graph_embed),dim=1))
            key = self.key_proj(node_embed)
            value = self.value_proj(node_embed)

            attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (query.shape[-1] ** 0.5)
            attention_weights = F.softmax(attention_scores, dim=-1)

            context_vector = torch.matmul(attention_weights, value)
            action_probabilities = F.softmax(self.output_layer(context_vector), dim=-1)

            return action_probabilities.squeeze()


class ActionNet(nn.Module):

    def __init__(self, node_features, classification_features, list_action_space, embed_dim=64, mlp_hidden=64, max_lv=1, num_heads = 1, dropout=0.1, gm='mean', device='cpu'):
        '''
        bilin_q: bilinear q or not
        mlp_hidden: mlp hidden layer size
        mav_lv: max rounds of message passing
        '''
        super(ActionNet, self).__init__()
        self.node_features = node_features
        self.classification_features = classification_features
        self.list_action_space = list_action_space
        self.total_nodes = len(list_action_space)

        self.embed_dim = embed_dim
        self.mlp_hidden = mlp_hidden
        self.max_lv = max_lv
        self.gm = gm

        last_wout = embed_dim
        self.bias_target = Parameter(torch.Tensor(1, embed_dim))

        if mlp_hidden:
            self.linear_1 = nn.Linear(embed_dim * 2, mlp_hidden)
            self.linear_out = nn.Linear(mlp_hidden, last_wout)
        else:
            self.linear_out = nn.Linear(embed_dim * 2, last_wout)

        self.w_n2l = Parameter(torch.Tensor(node_features.size()[1], embed_dim))
        self.bias_n2l = Parameter(torch.Tensor(embed_dim))
        self.bias_picked = Parameter(torch.Tensor(1, embed_dim))
        self.conv_params = nn.Linear(embed_dim, embed_dim)
        self.norm_tool = GraphNormTool(normalize=True, gm=self.gm, device=device)
        self.HANModel = HAN(meta_paths=StaticGraph.metapaths,
                            in_size=self.classification_features.shape[1],
                            hidden_size=mlp_hidden,
                            out_size=self.embed_dim,
                            num_heads=[num_heads],
                            dropout=dropout).to(device)
        # self.query_proj = nn.Linear(embed_dim * 2, embed_dim)
        # self.key_proj = nn.Linear(embed_dim, embed_dim)
        # self.value_proj = nn.Linear(embed_dim, embed_dim)
        # self.output_layer = nn.Linear(embed_dim, self.region_length)
        weights_init(self)

    def update_list_action_space(self, list_action_space):
        self.list_action_space = list_action_space

    def update_node_features_with_han(self, graph, classification_features, node_features):
        classification_node_embedding = self.HANModel(graph, classification_features)
        # node_features[:classification_node_embedding.shape[0]] = classification_node_embedding
        updated_node_features = node_features.clone()
        updated_node_features[:classification_node_embedding.shape[0]] = classification_node_embedding
        return updated_node_features


    def forward(self, state, selected_type, is_inference=False):
        target_node, graph_info = state
        with torch.set_grad_enabled(mode=not is_inference):
            graph = graph_info.get_modified_heterograph(self.node_features.device)
            input_node_linear = self.update_node_features_with_han(graph, self.classification_features, self.node_features)
            if input_node_linear.data.is_sparse:
                input_node_linear = torch.spmm(input_node_linear, self.w_n2l) # 2708*64
            else:
                input_node_linear = torch.mm(input_node_linear, self.w_n2l)
            input_node_linear += self.bias_n2l

            start_point = StaticGraph.node_mapping[(selected_type,self.list_action_space[target_node][1][selected_type][0])]
            end_point = StaticGraph.node_mapping[(selected_type,self.list_action_space[target_node][1][selected_type][-1])]
            region = list(range(start_point, end_point+1))
            node_embed = input_node_linear.clone()
            node_embed = F.relu(node_embed)
            target_embed = node_embed[target_node, :].view(-1, 1) # 64*1
            node_embed = node_embed[region]

            graph_embed = torch.mean(input_node_linear, dim=0, keepdim=True)
            graph_embed = graph_embed.repeat(node_embed.size()[0], 1)
            embed_s_a = torch.cat((node_embed, graph_embed), dim=1)

            if self.mlp_hidden:
                embed_s_a = F.relu(self.linear_1(embed_s_a))  
            raw_pred = self.linear_out(embed_s_a)
            raw_pred = torch.mm(raw_pred, target_embed)
            action_probability = F.softmax(raw_pred.view(-1), dim=0)

        return action_probability





class NStepQNetNode(nn.Module):

    def __init__(self, num_steps, node_features, classification_features, node_labels, list_action_space, embed_dim=64, mlp_hidden=64, max_lv=1, num_heads = 1, dropout=0.1, gm='mean_field', device='cpu'):

        super(NStepQNetNode, self).__init__()
        self.node_features = node_features
        self.classification_features = classification_features
        self.node_labels = node_labels
        self.list_action_space = list_action_space
        self.total_nodes = len(list_action_space)

        list_mod = []
        for i in range(0, num_steps):
            # list_mod.append(QNetNode(node_features, node_labels, list_action_space))
            list_mod.append(TypeNet(classification_features, node_features, list_action_space, embed_dim, mlp_hidden, max_lv, num_heads, dropout, gm, device=device))
            list_mod.append(ActionNet(classification_features, node_features, list_action_space, embed_dim, mlp_hidden, max_lv, num_heads, dropout, gm, device=device))

        self.list_mod = nn.ModuleList(list_mod)
        self.num_steps = num_steps * 2

    def update_list_action_space(self, list_action_space):
        self.list_action_space = list_action_space
        for i in range(0, len(self.list_mod)):
            self.list_mod[i].update_list_action_space(list_action_space)


    def forward(self, time_t, state, selected_type = None, is_inference=False):
        assert time_t >= 0 and time_t < self.num_steps
        if selected_type == None:
            return self.list_mod[time_t](state, is_inference)
        else:
            return self.list_mod[time_t](state, selected_type, is_inference)



def glorot_uniform(t):
    if len(t.size()) == 2:
        fan_in, fan_out = t.size()
    elif len(t.size()) == 3:
        # out_ch, in_ch, kernel for Conv 1
        fan_in = t.size()[1] * t.size()[2]
        fan_out = t.size()[0] * t.size()[2]
    else:
        fan_in = np.prod(t.size())
        fan_out = np.prod(t.size())

    limit = np.sqrt(6.0 / (fan_in + fan_out))
    t.uniform_(-limit, limit)


def _param_init(m):
    if isinstance(m, Parameter):
        glorot_uniform(m.data)
    elif isinstance(m, nn.Linear):
        if m.bias is not None:
            m.bias.data.zero_()
        glorot_uniform(m.weight.data)

def weights_init(m):
    for p in m.modules():
        if isinstance(p, nn.ParameterList):
            for pp in p:
                _param_init(pp)
        else:
            _param_init(p)

    for name, p in m.named_parameters():
        if not '.' in name: # top-level parameters
            _param_init(p)
