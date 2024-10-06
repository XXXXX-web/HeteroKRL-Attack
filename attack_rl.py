'''
The code base framework is derived from deeprobust https://github.com/DSE-MSU/DeepRobust
'''
import torch
import argparse
import random
import numpy as np
import pickle as pkl
from HAN.utils import *
from HAN.model import *
from little_function import *
from env import StaticGraph,NodeAttackEnv
from Agent import Agent
from HAN.model import *



parser = argparse.ArgumentParser(description="Parameters")
parser.add_argument("--dataset", help="dataset choice", default="imdb")
parser.add_argument("--random_seed", help = "random seed", default = 123)
parser.add_argument("--target_model_location", help = "", default= "save_model/mid_dglHan_imdb.pth")
parser.add_argument("--budget", help = "num_mods of budget", type = int, default = 3)
parser.add_argument("--phase", help = "train or test", default = "test")
parser.add_argument("--num_steps", help = "rl training steps", default = 500000)
parser.add_argument("--save_dir", help = "save folder", default = "E:\\Study\\Complex System\\HANAttack\\save_model")
parser.add_argument("--bilin_q", help = "bilinear q or not", default = 1)
parser.add_argument("--latent_dim", help = "dimension of latent layers", default = 64,type = int)
parser.add_argument("--num_heads", help = "number of head of HAN", default = 1,type = int)
parser.add_argument("--dropout", help = "dropout of HAN", default = 0.0, type = float)
parser.add_argument("--mlp_hidden", help = "mlp hidden layer size", default = 64,type = int)
parser.add_argument("--max_lv", help = "max rounds of message passing", default = 1, type = int)
parser.add_argument("--gm", help = "mean_field/loopy_bp/gcn", default = "mean_field")
parser.add_argument("--learning_rate", help = "init learning_rate", default = 0.0001, type = float)
parser.add_argument("--k_value", help = "K nearest actions", default = 0.005, type = float)
parser.add_argument("--use_knn", help = "Weather use knn in test", default = False, type = bool)

args = parser.parse_args()

dataset = args.dataset 
seed = args.random_seed 
target_model = args.target_model_location
budget = args.budget
k_value = args.k_value
print("budget is ",budget)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed) 

def load_dataset(dataset_name):
    if dataset_name == "acm":
        hg, hete_adjs, classification_features, labels, num_classes, train_idx, val_idx, test_idx, \
            train_mask, val_mask, test_mask = load_acm_raw(False)
        target_type = load_acm_raw.return_target_type()
        metapaths = load_acm_raw.return_metapaths()
    elif dataset_name == "imdb":
        hg, hete_adjs, classification_features, labels, num_classes, train_idx, val_idx, test_idx, \
            train_mask, val_mask, test_mask = load_imdb_raw()
        target_type = load_imdb_raw.return_target_type()
        metapaths = load_imdb_raw.return_metapaths()
    elif dataset_name == "dblp":
        hg, hete_adjs, classification_features, labels, num_classes, train_idx, val_idx, test_idx, \
            train_mask, val_mask, test_mask = load_dblp_raw(False)
        target_type = load_dblp_raw.return_target_type()
        metapaths = load_dblp_raw.return_metapaths()
    else:
        raise TypeError("No such dataset")
    return hg, hete_adjs, classification_features, labels, num_classes, train_idx, val_idx, test_idx, \
            train_mask, val_mask, test_mask,target_type, metapaths

def init_setup():
    hg, hete_adjs, classification_features, labels, num_classes, train_idx, val_idx, test_idx, \
            train_mask, val_mask, test_mask, target_type, metapaths = load_dataset(dataset)
    device = torch.device('cuda') if torch.cuda.is_available() == True else 'cpu'
    nx_graph, node_mapping = heterograph_to_networkx(hg,target_type)
    embeddings = get_whole_node_mapping(nx_graph, 64, dataset, read=1)
    all_features = embeddings_to_tensor(embeddings)

    StaticGraph.homoGraph = nx_graph
    StaticGraph.heteroGraph = hg 
    StaticGraph.metapaths = metapaths
    StaticGraph.node_mapping = node_mapping
    StaticGraph.reversed_node_mapping = {value: key for key, value in node_mapping.items()}
    StaticGraph.embeddings = embeddings 
    StaticGraph.target_type = target_type
    StaticGraph.node_node_etype_mapping()

    hg,classification_features,labels,all_features, dict_of_lists = preprocess(nx_graph,hg,classification_features,labels, all_features,device,StaticGraph.reversed_node_mapping, target_type)
    victim_model = torch.load(target_model).to(device)
    if type(test_idx) != list:
        tar_idx = test_idx.tolist()
    if args.phase == 'test':
        tar_idx = test_idx
    elif args.phase == 'train':
        tar_idx = val_idx
    with torch.no_grad():
        logits = victim_model(hg, classification_features)
    tar_accuracy, tar_micro_f1_clean, tar_macro_f1_clean = score(logits[tar_idx], labels[tar_idx])
    print("Clean data:  Micro-F1:", tar_micro_f1_clean, " Macro-F1:",tar_macro_f1_clean,"Accuracy:",tar_accuracy)

    return hg, classification_features, labels, all_features, train_idx, val_idx, test_idx, dict_of_lists, victim_model, node_mapping, device

def generate_attack_list(victim_model, hg, features,  labels, train_idx, idx_val, idx_test,dict_of_lists): 
    with torch.no_grad():
        output = victim_model(hg, features) 
    preds = output.max(1)[1].type_as(labels) 
    acc = preds.eq(labels).double() 
    acc_test = acc[train_idx] 

    attack_list = []
    for i in range(len(train_idx)):
        # only attack those misclassifed and degree>0 nodes
        if acc_test[i] > 0 and len(dict_of_lists[train_idx[i]]): 
            attack_list.append(train_idx[i])

    total = range(features.shape[0])


    tar_idx = idx_val
    #tar_idx = random.sample(list(train_idx)+  list(idx_test),400)
    meta_list = []
    wrong_lables = []
    wrong_logits = []
    num_wrong = 0
    for i in tar_idx: 
        if acc[i] > 0:
            if len(dict_of_lists[i]):
                meta_list.append(i)
        else:
            num_wrong += 1
            wrong_lables.append(labels[i])
            wrong_logits.append(output[i])
    print( 'meta list ratio:', len(meta_list) / float(len(tar_idx))) 

    preprocessed_eval_idx = idx_test
    wrong_eval_labels = []
    wrong_eval_logits = []
    num_wrong_eval = 0
    processed_eval_idx = []
    for i in preprocessed_eval_idx:
        if acc[i] > 0:
            if len(dict_of_lists[i]):
                processed_eval_idx.append(i)
        else:
            num_wrong_eval += 1
            wrong_eval_labels.append(labels[i])
            wrong_eval_logits.append(output[i])
    print( 'processed_eval_idx ratio:', len(processed_eval_idx) / float(len(idx_test)))
    return total, acc, meta_list, attack_list,num_wrong, wrong_logits, wrong_lables,\
            processed_eval_idx, num_wrong_eval, wrong_eval_labels, wrong_eval_logits



def main():
     hg, classification_features, labels, all_features, train_idx, val_idx, test_idx, dict_of_lists, victim_model, node_mapping, device = init_setup()
     total, acc, meta_list, attack_list, num_wrong, wrong_logits, wrong_lables, processed_eval_idx, num_wrong_eval, wrong_eval_labels, wrong_eval_logits= generate_attack_list(
         victim_model, hg, classification_features, labels, train_idx, val_idx, test_idx, dict_of_lists)
     env = NodeAttackEnv(classification_features, labels, total, dict_of_lists,node_mapping, victim_model, k_value, args, num_mod =budget)

     agent = Agent(args.dataset, env, classification_features, all_features, labels, meta_list, attack_list,
                   processed_eval_idx, dict_of_lists, num_wrong_eval = num_wrong_eval, wrong_eval_labels = wrong_eval_labels,
                   wrong_eval_logits = wrong_eval_logits, wrong_logits=wrong_logits, wrong_labels=wrong_lables, num_wrong=num_wrong,
                   num_mod=int(args.budget), save_dir=args.save_dir,
                   bilin_q=args.bilin_q, embed_dim=args.latent_dim,
                   mlp_hidden=args.mlp_hidden, max_lv=args.max_lv,
                   num_heads=args.num_heads, dropout=args.dropout,
                   gm=args.gm, learning_rate=args.learning_rate, device=device)

     if args.phase == "train":
        agent.train_on_policy_agent()
     elif args.phase == "test":
         agent.test_on_policy_agent()

if __name__ == "__main__":
    main()
