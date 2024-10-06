import torch
import numpy as np
import torch.nn.functional as F
import networkx as nx
import copy
from node2vec import Node2Vec
import pickle
from sklearn.metrics import f1_score
from tqdm import tqdm
import json
import time
from pyflann import *
import math


def loss_acc(output, labels, targets, evaluation = False, avg_loss=True):
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()[targets]
    loss = F.nll_loss(output[targets], labels[targets], reduction='mean' if avg_loss else 'none')

    if evaluation == True:
        return loss, correct, output[targets],labels[targets]

    if avg_loss:
        return loss, correct.sum() / len(targets)
    return loss, correct


def get_neighbors(adj_matrix, node_id):
    indices = adj_matrix.coalesce().indices().cpu().numpy()
    mask = indices[0] == node_id
    neighbors = indices[1][mask]
    neighbors_filtered = neighbors[neighbors != node_id]

    return neighbors_filtered.tolist()



def heterograph_to_networkx(g,target_type):
    nx_graph = nx.Graph()
    node_mapping = {}
    types = copy.deepcopy(g.ntypes)
    types.remove(target_type)
    types.insert(0, target_type)

    node_id = 0
    for ntype in types:
        num_nodes = g.number_of_nodes(ntype)
        for nid in range(num_nodes):
            node_mapping[(ntype, nid)] = node_id
            nx_graph.add_node(node_id)
            node_id += 1

    for etype in g.canonical_etypes:
        src, dst = g.edges(etype=etype)
        src = src.numpy()
        dst = dst.numpy()

        for s, d in zip(src, dst):
            nx_src = node_mapping[(etype[0], s)]
            nx_dst = node_mapping[(etype[2], d)]
            nx_graph.add_edge(nx_src, nx_dst)

    return nx_graph, node_mapping


def get_whole_node_mapping(nx_graph, dimension, dataset, read = 1):
    file_path = 'homo_embeddings/homo_' + dataset + '_embedding' + '.pkl'
    if os.path.exists(file_path):
        read = 1
    else:
        read = 0
    if read == 0:
        node2vec = Node2Vec(nx_graph, dimensions=dimension, walk_length=10, num_walks=10, workers=4)
        model = node2vec.fit(window=5, min_count=1, batch_words=4)
        node_id = 0 
        embeddings = {int(node): model.wv[node] for node in model.wv.index_to_key}
        save_embeddings_to_file('homo_embeddings/homo_' + dataset + '_embedding_' + str(dimension) + '.pkl', embeddings)
    else:
        embeddings = load_embeddings_from_file(file_path)
    return embeddings


def load_embeddings_from_file(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def save_embeddings_to_file(file_name, embeddings):
    with open(file_name, 'wb') as f:
        pickle.dump(embeddings, f)


def embeddings_to_tensor(embeddings):
    return torch.tensor([embeddings[i] for i in range(len(embeddings))])

def find_two_hop_neighbors(G,reversed_node_mapping,target_type):
    two_hop_neighbors = {}
    for node in G.nodes():
        one_hop = set(G.neighbors(node)) 
        two_hop = set()
        for neighbor in one_hop:
            two_hop |= set(G.neighbors(neighbor)) 
        two_hop.discard(node)
        two_hop_copy = copy.deepcopy(two_hop)
        for n in two_hop:
            if reversed_node_mapping[n][0] == target_type:
                two_hop_copy.discard(n)
        two_hop_neighbors[node] = list(two_hop_copy)
    return two_hop_neighbors

def find_all_papers_three_hop_neighbors(graph,reversed_node_mapping, target_type):
    combined_neighbors = {}
    for node in tqdm(graph.nodes()):
        one_hop = set(graph.neighbors(node))
        two_hop = set()
        for neighbor in one_hop:
            two_hop |= set(graph.neighbors(neighbor))

        two_hop.discard(node)
        three_hop = set()
        for neighbor in two_hop:
            three_hop |= set(graph.neighbors(neighbor))

        three_hop.discard(node)
        all_hops = one_hop | two_hop | three_hop
        all_hops_copy = copy.deepcopy(all_hops)
        for n in all_hops:
            if reversed_node_mapping[n][0] == target_type:
                all_hops_copy.discard(n)

        combined_neighbors[node] = list(all_hops_copy)

    return combined_neighbors

def get_all_nodes(graph,reversed_node_mapping,target_type):
    all_nodes = graph.nodes()
    all_nodes = [n for n in all_nodes if reversed_node_mapping[n][0] != target_type]
    neighbors = {}
    for n in graph.nodes():
        neighbors_set = all_nodes.copy()
        neighbors[n] = list(neighbors_set)
    return neighbors

def get_action_space_on_hete(hg, reversed_node_mapping, target_type):
    action_space_on_hete = {}
    all_node_types = copy.deepcopy(hg.ntypes)
    all_node_types.remove(target_type)
    for node in hg.nodes(target_type):
        node = node.item()
        action_space_on_hete[node] = []
        action_space_on_hete[node].append(all_node_types)
        action_space_on_hete[node].append(dict())
        for t in all_node_types:
            action_space_on_hete[node][1][t] = hg.nodes(t).tolist()
    return action_space_on_hete


def preprocess(g, hg, features, labels, all_features, device, reversed_node_mapping, target_type):
    # dict = find_two_hop_neighbors(g) 
    #dict = get_all_nodes(g,reversed_node_mapping,target_type)
    dict = get_action_space_on_hete(hg, reversed_node_mapping,target_type)
    return hg.to(device),features.to(device),labels.to(device),all_features.to(device),dict


def score(logits, labels,num_wrong = 0):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()
    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')
    return accuracy, micro_f1, macro_f1


def deep_copy_by_pickle(obj):
    copy_result = json.loads(json.dumps(obj))
    return copy_result



def find_k_nearest_nodes_pyflann(embeddings, action, action_space,chosen_metapath, node_mapping,reversed_node_mapping, k=5):
    action_space_nodes = [node_mapping[(chosen_metapath,node)] for node in action_space[1][chosen_metapath] ]
    action_space_embedding_matrix = np.array([embeddings[node] for node in action_space_nodes])

    k = math.ceil(k * len(action_space_nodes))
    if k > len(action_space_nodes):
        k = len(action_space_nodes)
    action = node_mapping[(chosen_metapath,action)]
    flann = FLANN()
    params = flann.build_index(action_space_embedding_matrix, algorithm="kdtree", trees=4)
    query_embedding = np.array(embeddings[action]).reshape(1, -1)
    indices, _ = flann.nn_index(query_embedding, num_neighbors=k)

    if indices.ndim == 2:
        nearest_nodes = [action_space_nodes[i] for i in indices[0]]
    else:
        nearest_nodes = [action_space_nodes[indices[0]]] if k == 1 else [action_space_nodes[i] for i in indices]

    nearest_nodes = [reversed_node_mapping[n] for n in nearest_nodes]

    return nearest_nodes

def find_max_value_and_index(loss_list, acc_list):
    acc_0_indices = [i for i, x in enumerate(acc_list) if x == 0]

    if acc_0_indices:
        return acc_0_indices[0]
    else:
        max_value = max(loss_list)
        max_index = loss_list.index(max_value)
        return max_index


import numpy as np
import scipy.sparse as sp
import torch
from sklearn.model_selection import train_test_split
import torch.sparse as ts
import torch.nn.functional as F
import warnings

def encode_onehot(labels):
    """Convert label to onehot format.

    Parameters
    ----------
    labels : numpy.array
        node labels

    Returns
    -------
    numpy.array
        onehot labels
    """
    eye = np.eye(labels.max() + 1)
    onehot_mx = eye[labels]
    return onehot_mx

def tensor2onehot(labels):
    """Convert label tensor to label onehot tensor.

    Parameters
    ----------
    labels : torch.LongTensor
        node labels

    Returns
    -------
    torch.LongTensor
        onehot labels tensor

    """

    eye = torch.eye(labels.max() + 1)
    onehot_mx = eye[labels]
    return onehot_mx.to(labels.device)

def to_tensor(adj, features, labels=None, device='cpu'):
    """Convert adj, features, labels from array or sparse matrix to
    torch Tensor.

    Parameters
    ----------
    adj : scipy.sparse.csr_matrix
        the adjacency matrix.
    features : scipy.sparse.csr_matrix
        node features
    labels : numpy.array
        node labels
    device : str
        'cpu' or 'cuda'
    """
    if sp.issparse(adj):
        adj = sparse_mx_to_torch_sparse_tensor(adj)
    else:
        adj = torch.FloatTensor(adj)
    if sp.issparse(features):
        features = sparse_mx_to_torch_sparse_tensor(features)
    else:
        features = torch.FloatTensor(np.array(features))

    if labels is None:
        return adj.to(device), features.to(device)
    else:
        labels = torch.LongTensor(labels)
        return adj.to(device), features.to(device), labels.to(device)

def normalize_feature(mx):
    """Row-normalize sparse matrix or dense matrix

    Parameters
    ----------
    mx : scipy.sparse.csr_matrix or numpy.array
        matrix to be normalized

    Returns
    -------
    scipy.sprase.lil_matrix
        normalized matrix
    """
    if type(mx) is not sp.lil.lil_matrix:
        try:
            mx = mx.tolil()
        except AttributeError:
            pass
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(mx):
    """Normalize sparse adjacency matrix,
    A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
    Row-normalize sparse matrix

    Parameters
    ----------
    mx : scipy.sparse.csr_matrix
        matrix to be normalized

    Returns
    -------
    scipy.sprase.lil_matrix
        normalized matrix
    """

    # TODO: maybe using coo format would be better?
    if type(mx) is not sp.lil.lil_matrix:
        mx = mx.tolil()
    if mx[0, 0] == 0 :
        mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = mx.dot(r_mat_inv)
    return mx

def normalize_sparse_tensor(adj, fill_value=1):
    """Normalize sparse tensor. Need to import torch_scatter
    """
    edge_index = adj._indices()
    edge_weight = adj._values()
    num_nodes= adj.size(0)
    edge_index, edge_weight = add_self_loops(
	edge_index, edge_weight, fill_value, num_nodes)

    row, col = edge_index
    from torch_scatter import scatter_add
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    values = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    shape = adj.shape
    return torch.sparse.FloatTensor(edge_index, values, shape)

def add_self_loops(edge_index, edge_weight=None, fill_value=1, num_nodes=None):
    # num_nodes = maybe_num_nodes(edge_index, num_nodes)

    loop_index = torch.arange(0, num_nodes, dtype=torch.long,
                              device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)

    if edge_weight is not None:
        assert edge_weight.numel() == edge_index.size(1)
        loop_weight = edge_weight.new_full((num_nodes, ), fill_value)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

    edge_index = torch.cat([edge_index, loop_index], dim=1)

    return edge_index, edge_weight

def normalize_adj_tensor(adj, sparse=False):
    """Normalize adjacency tensor matrix.
    """
    device = adj.device
    if sparse:
        # warnings.warn('If you find the training process is too slow, you can uncomment line 207 in deeprobust/graph/utils.py. Note that you need to install torch_sparse')
        # TODO if this is too slow, uncomment the following code,
        # but you need to install torch_scatter
        # return normalize_sparse_tensor(adj)
        adj = to_scipy(adj)
        mx = normalize_adj(adj)
        return sparse_mx_to_torch_sparse_tensor(mx).to(device)
    else:
        mx = adj + torch.eye(adj.shape[0]).to(device)
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
    return mx

def degree_normalize_adj(mx):
    """Row-normalize sparse matrix"""
    mx = mx.tolil()
    if mx[0, 0] == 0 :
        mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    # mx = mx.dot(r_mat_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def degree_normalize_sparse_tensor(adj, fill_value=1):
    """degree_normalize_sparse_tensor.
    """
    edge_index = adj._indices()
    edge_weight = adj._values()
    num_nodes= adj.size(0)

    edge_index, edge_weight = add_self_loops(
	edge_index, edge_weight, fill_value, num_nodes)

    row, col = edge_index
    from torch_scatter import scatter_add
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-1)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    values = deg_inv_sqrt[row] * edge_weight
    shape = adj.shape
    return torch.sparse.FloatTensor(edge_index, values, shape)

def degree_normalize_adj_tensor(adj, sparse=True):
    """degree_normalize_adj_tensor.
    """

    device = adj.device
    if sparse:
        # return  degree_normalize_sparse_tensor(adj)
        adj = to_scipy(adj)
        mx = degree_normalize_adj(adj)
        return sparse_mx_to_torch_sparse_tensor(mx).to(device)
    else:
        mx = adj + torch.eye(adj.shape[0]).to(device)
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
    return mx

def accuracy(output, labels):
    """Return accuracy of output compared to labels.

    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        node labels

    Returns
    -------
    float
        accuracy
    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def get_perf(output, labels, mask, verbose=True):
    """evalute performance for test masked data"""
    loss = F.nll_loss(output[mask], labels[mask])
    acc = accuracy(output[mask], labels[mask])
    if verbose:
        print("loss= {:.4f}".format(loss.item()),
              "accuracy= {:.4f}".format(acc.item()))
    return loss.item(), acc.item()


def classification_margin(output, true_label):
    """Calculate classification margin for outputs.
    `probs_true_label - probs_best_second_class`

    Parameters
    ----------
    output: torch.Tensor
        output vector (1 dimension)
    true_label: int
        true label for this node

    Returns
    -------
    list
        classification margin for this node
    """

    probs = torch.exp(output)
    probs_true_label = probs[true_label].clone()
    probs[true_label] = 0
    probs_best_second_class = probs[probs.argmax()]
    return (probs_true_label - probs_best_second_class).item()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    sparserow=torch.LongTensor(sparse_mx.row).unsqueeze(1)
    sparsecol=torch.LongTensor(sparse_mx.col).unsqueeze(1)
    sparseconcat=torch.cat((sparserow, sparsecol),1)
    sparsedata=torch.FloatTensor(sparse_mx.data)
    return torch.sparse.FloatTensor(sparseconcat.t(),sparsedata,torch.Size(sparse_mx.shape))

	# slower version....
    # sparse_mx = sparse_mx.tocoo().astype(np.float32)
    # indices = torch.from_numpy(
    #     np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    # values = torch.from_numpy(sparse_mx.data)
    # shape = torch.Size(sparse_mx.shape)
    # return torch.sparse.FloatTensor(indices, values, shape)



def to_scipy(tensor):
    """Convert a dense/sparse tensor to scipy matrix"""
    if is_sparse_tensor(tensor):
        values = tensor._values()
        indices = tensor._indices()
        return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)
    else:
        indices = tensor.nonzero().t()
        values = tensor[indices[0], indices[1]]
        return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)

def is_sparse_tensor(tensor):
    """Check if a tensor is sparse tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        given tensor

    Returns
    -------
    bool
        whether a tensor is sparse tensor
    """
    # if hasattr(tensor, 'nnz'):
    if tensor.layout == torch.sparse_coo:
        return True
    else:
        return False

def get_train_val_test(nnodes, val_size=0.1, test_size=0.8, stratify=None, seed=None):
    """This setting follows nettack/mettack, where we split the nodes
    into 10% training, 10% validation and 80% testing data

    Parameters
    ----------
    nnodes : int
        number of nodes in total
    val_size : float
        size of validation set
    test_size : float
        size of test set
    stratify :
        data is expected to split in a stratified fashion. So stratify should be labels.
    seed : int or None
        random seed

    Returns
    -------
    idx_train :
        node training indices
    idx_val :
        node validation indices
    idx_test :
        node test indices
    """

    assert stratify is not None, 'stratify cannot be None!'

    if seed is not None:
        np.random.seed(seed)

    idx = np.arange(nnodes)
    train_size = 1 - val_size - test_size
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=None,
                                                   train_size=train_size + val_size,
                                                   test_size=test_size,
                                                   stratify=stratify)

    if stratify is not None:
        stratify = stratify[idx_train_and_val]

    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=None,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)

    return idx_train, idx_val, idx_test

def get_train_test(nnodes, test_size=0.8, stratify=None, seed=None):
    """This function returns training and test set without validation.
    It can be used for settings of different label rates.

    Parameters
    ----------
    nnodes : int
        number of nodes in total
    test_size : float
        size of test set
    stratify :
        data is expected to split in a stratified fashion. So stratify should be labels.
    seed : int or None
        random seed

    Returns
    -------
    idx_train :
        node training indices
    idx_test :
        node test indices
    """
    assert stratify is not None, 'stratify cannot be None!'

    if seed is not None:
        np.random.seed(seed)

    idx = np.arange(nnodes)
    train_size = 1 - test_size
    idx_train, idx_test = train_test_split(idx, random_state=None,
                                                train_size=train_size,
                                                test_size=test_size,
                                                stratify=stratify)

    return idx_train, idx_test

def get_train_val_test_gcn(labels, seed=None):
    """This setting follows gcn, where we randomly sample 20 instances for each class
    as training data, 500 instances as validation data, 1000 instances as test data.
    Note here we are not using fixed splits. When random seed changes, the splits
    will also change.

    Parameters
    ----------
    labels : numpy.array
        node labels
    seed : int or None
        random seed

    Returns
    -------
    idx_train :
        node training indices
    idx_val :
        node validation indices
    idx_test :
        node test indices
    """
    if seed is not None:
        np.random.seed(seed)

    idx = np.arange(len(labels))
    nclass = labels.max() + 1
    idx_train = []
    idx_unlabeled = []
    for i in range(nclass):
        labels_i = idx[labels==i]
        labels_i = np.random.permutation(labels_i)
        idx_train = np.hstack((idx_train, labels_i[: 20])).astype(np.int)
        idx_unlabeled = np.hstack((idx_unlabeled, labels_i[20: ])).astype(np.int)

    idx_unlabeled = np.random.permutation(idx_unlabeled)
    idx_val = idx_unlabeled[: 500]
    idx_test = idx_unlabeled[500: 1500]
    return idx_train, idx_val, idx_test
