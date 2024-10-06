import numpy as np
import torch
import random
from copy import deepcopy
from little_function import *
from deeprobust.graph import utils
import dgl
class GraphNormTool(object):

    def __init__(self, normalize, gm, device):
        self.adj_norm = normalize
        self.gm = gm
        g = StaticGraph.homoGraph
        edges = np.array(g.edges(), dtype=np.int64)
        rev_edges = np.array([edges[:, 1], edges[:, 0]], dtype=np.int64)

        edges = np.hstack((edges.T, rev_edges))
        idxes = torch.LongTensor(edges)
        values = torch.ones(idxes.size()[1])

        self.raw_adj = torch.sparse.FloatTensor(idxes, values, StaticGraph.get_gsize())
        self.raw_adj = self.raw_adj.to(device)

        self.normed_adj = self.raw_adj.clone()
        if self.adj_norm:
            if self.gm == 'gcn':
                self.normed_adj = utils.normalize_adj_tensor(self.normed_adj, sparse=True)
            else:
                self.normed_adj = utils.degree_normalize_adj_tensor(self.normed_adj, sparse=True)

    def norm_extra(self, added_adj = None):
        if added_adj is None:
            return self.normed_adj

        new_adj = self.raw_adj + added_adj
        if self.adj_norm:
            if self.gm == 'gcn':
                new_adj = utils.normalize_adj_tensor(new_adj, sparse=True)
            else:
                new_adj = utils.degree_normalize_adj_tensor(new_adj, sparse=True)

        return new_adj


class StaticGraph(object):
    homoGraph = None
    heteroGraph = None
    node_mapping = None
    reversed_node_mapping = None
    embeddings = None
    target_type = None
    metapaths = None
    node_node_to_types = None

    @staticmethod
    def get_gsize():
        return torch.Size((len(StaticGraph.homoGraph), len(StaticGraph.homoGraph)))

    @staticmethod
    def node_node_etype_mapping():
        all_edge_types = StaticGraph.heteroGraph.canonical_etypes
        StaticGraph.node_node_to_types = {}
        for t in all_edge_types:
            StaticGraph.node_node_to_types[(t[0], t[-1])] = t[1]

class ModifiedGraph(object):
    def __init__(self, directed_edges=None, weights=None):
        self.edge_set = []
        self.node_set = []
        self.directed_edges = []
        self.weights = []

    def add_edge(self, attack_node_type, target_node, action):
        assert attack_node_type is not None and action is not None
        etype = StaticGraph.node_node_to_types[(StaticGraph.target_type, attack_node_type)]
        reversed_etype = StaticGraph.node_node_to_types[(attack_node_type, StaticGraph.target_type)]
        self.edge_set.append((etype, (target_node, action)))
        self.edge_set.append((reversed_etype, (action, target_node)))
        if StaticGraph.heteroGraph.has_edges_between(target_node, action, etype=etype):
            self.weights.append(-1)
            self.weights.append(-1)
        else:
            self.weights.append(1)
            self.weights.append(1)

    def get_modified_heterograph(self, device):
        graph = deepcopy(StaticGraph.heteroGraph)
        index = 0
        for i in range(len(self.edge_set)):
            edge_info = self.edge_set[i]
            add_or_delete = self.weights[i]
            metapath, (src,dst) = edge_info
            src_tensor = torch.tensor([src],dtype=torch.int64)
            dst_tensor = torch.tensor([dst],dtype=torch.int64)
            if add_or_delete == 1:
                graph = dgl.add_edges(graph, src_tensor, dst_tensor, etype = metapath)
            else:
                graph = dgl.remove_edges(graph, torch.tensor([0, 1]), etype = metapath)
        return graph.to(device)


    def get_extra_adj(self, device):
        if len(self.directed_edges):
            edges = np.array(self.directed_edges, dtype=np.int64)
            rev_edges = np.array([edges[:, 1], edges[:, 0]], dtype=np.int64)
            edges = np.hstack((edges.T, rev_edges))

            idxes = torch.LongTensor(edges)
            values = torch.Tensor(self.weights)

            added_adj = torch.sparse.FloatTensor(idxes, values, StaticGraph.get_gsize())

            added_adj = added_adj.to(device)
            return added_adj
        else:
            return None



class NodeAttackEnv(object):
    def __init__(self, features, labels, all_targets, list_action_space, node_mapping, classifier, k_value, args, num_mod = 1):
        self.features = features
        self.labels = labels
        self.classifier = classifier
        self.all_targets = all_targets
        self.list_action_space_original = list_action_space
        self.node_mapping = node_mapping
        self.reversed_node_mapping = {v: k for k, v in node_mapping.items()}
        self.num_mod = num_mod
        self.k_value = k_value
        self.evaluation_logits = []
        self.evaluation_labels = []
        self.args = args


    def setup(self, target_node):
        self.target_node = target_node
        self.n_steps = 0
        self.chosen_metapath = None
        self.rewards = None
        self.binary_rewards = None
        self.modified_graph_info = ModifiedGraph()
        self.alternative_modified_list = []
        self.list_action_space = {target_node: deepcopy(self.list_action_space_original[self.target_node])} # 重置action space
        self.list_acc_of_all = [] # 用于保存所有节点的准确率 我记得这变量也没什么用
        self.negative_rewards = -1/StaticGraph.homoGraph.degree(self.target_node) # 计算负的rewards是多少 -1/degree

    def update_action_space(self, target_node,action):
        self.list_action_space[target_node].remove(action)

    def update_actions_with_knn(self,action):
        new_actions = []
        loss_list = []
        acc_list = []

        k_actions = find_k_nearest_nodes_pyflann(StaticGraph.embeddings, action,
                                                 self.list_action_space[self.target_node], self.chosen_metapath,
                                                 StaticGraph.node_mapping, StaticGraph.reversed_node_mapping,
                                                 self.k_value)
        self.alternative_modified_list = []
        for i in range(len(k_actions)):
            self.alternative_modified_list.append(deepcopy(self.modified_graph_info))

        for j in range(len(k_actions)):
            self.alternative_modified_list[j].add_edge(k_actions[j][0], self.target_node, k_actions[j][1])
            modified_graph = self.alternative_modified_list[j].get_modified_heterograph(device=self.features.device)
            output = self.classifier(modified_graph, self.features)
            loss, acc = loss_acc(output, self.labels, self.all_targets, avg_loss=False)
            loss_list.append(loss.data.cpu().view(-1).numpy()[self.all_targets.index(self.target_node)])
            acc_list.append(acc.double().cpu().view(-1).numpy()[self.all_targets.index(self.target_node)])
        index = find_max_value_and_index(loss_list, acc_list)
        new_action = k_actions[index][1]
        return new_action

    def step(self, action_index, evaluation = False):
        assert self.n_steps < self.num_mod * 2, "The current number of steps exceeds the budget"
        self.rewards = 0
        self.binary_rewards = []
        self.list_acc_of_all = []
        self.dones = False
        acc_list = []
        loss_list = []
        type_or_nodes = self.n_steps % 2
        self.n_steps += 1
        device  = self.features.device
        if type_or_nodes == 0:
            action = self.list_action_space[self.target_node][type_or_nodes][action_index]
            self.chosen_metapath = action
            self.rewards = 0
            self.dones = False
        else:
            action = self.list_action_space[self.target_node][type_or_nodes][self.chosen_metapath][action_index]
            if evaluation and self.args.use_knn == True:
                action = self.update_actions_with_knn(action)
            # pass
            self.modified_graph_info.add_edge(self.chosen_metapath, self.target_node, action)
            modified_graph = self.modified_graph_info.get_modified_heterograph(device = device)
            output = self.classifier(modified_graph, self.features)
            if evaluation == False:
                loss, acc = loss_acc(output, self.labels, self.all_targets, avg_loss=False)
            else:
                loss, acc, logits, labels = loss_acc(output, self.labels, self.all_targets, evaluation=True,
                                                     avg_loss=False)
            cur_idx = self.all_targets.index(self.target_node)
            acc = np.copy(acc.double().cpu().view(-1).numpy())
            loss = loss.data.cpu().view(-1).numpy()
            self.list_acc_of_all.append(acc)
            acc_list.append(acc[cur_idx])
            loss_list.append(loss[cur_idx])
            self.chosen_metapath = None
            if evaluation == True:
                self.evaluation_logits = logits[cur_idx].detach().to('cpu').numpy()
                self.evaluation_labels = labels[cur_idx].detach().to('cpu').numpy()

            if acc[cur_idx] == 1:
                reward = loss[cur_idx]
                if reward > 0:
                    reward = -reward
                self.rewards = reward  # reward 的方法可能要更改
                if self.n_steps == self.num_mod * 2:
                    self.dones = True
                else:
                    self.dones = False
            else:
                self.rewards = 10
                self.dones = True
        return action

    def getStateRef(self):
        return [list((self.target_node, self.modified_graph_info)),self.n_steps, self.chosen_metapath]