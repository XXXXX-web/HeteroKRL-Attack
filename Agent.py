import os
import sys
import os.path as osp
import numpy as np
import torch
import networkx as nx
import random
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
from tqdm import tqdm
from copy import deepcopy
from q_net_node import NStepQNetNode
from little_function import score
import os
import pickle
from env import StaticGraph
from collections import OrderedDict

class Agent(object):
    def __init__(self, dataset_name, env, classification_features, features, labels, idx_meta, idx_test, idx_eval,
                 list_action_space, num_mod, num_wrong_eval, wrong_eval_labels,
                   wrong_eval_logits, wrong_logits, wrong_labels, batch_size=1,
                 num_wrong=0, bilin_q=1, embed_dim=64, gm='mean_field',
                 mlp_hidden=64, max_lv=1, num_heads = 1, dropout = 0.1, save_dir='checkpoint_dqn', learning_rate = 0.01, device=None):
        self.dataset = dataset_name
        self.features = features
        self.classification_features = classification_features
        self.labels = labels
        self.idx_meta = idx_meta
        self.idx_test = idx_test
        self.idx_eval = idx_eval
        self.num_wrong = num_wrong
        self.num_wrong_eval = num_wrong_eval
        if num_wrong_eval != 0:
            self.wrong_eval_logits = torch.stack(wrong_eval_logits).cpu()
            self.wrong_eval_labels = torch.from_numpy(np.array([t.cpu().numpy() for t in wrong_eval_labels]))

        self.list_action_space = list_action_space
        self.num_mod = num_mod
        self.batch_size = batch_size
        self.save_dir = save_dir
        if num_wrong != 0:
            self.wrong_logits = torch.stack(wrong_logits).cpu()
            self.wrong_labels = torch.from_numpy(np.array([t.cpu().numpy() for t in wrong_labels]))
        if not osp.exists(save_dir):
            os.system('mkdir -p {}'.format(save_dir))

        self.gm = gm
        self.device = device

        self.env = env
        self.net = NStepQNetNode(num_mod, classification_features, features, labels, list_action_space,
                                 embed_dim=embed_dim, mlp_hidden=mlp_hidden,max_lv=max_lv,
                                 num_heads=num_heads, dropout=dropout, gm=gm, device=device)
        self.net = self.net.to(device)
        self.embed_dim = embed_dim
        self.mlp_hidden = mlp_hidden
        self.max_lv = max_lv
        self.num_heads = num_heads
        self.eps_start = 1.0
        self.eps_end = 0.05
        self.eps_step = 100000
        self.burn_in = 10
        self.step = 0
        self.pos = 0
        self.gamma = 0.99
        self.best_success = -1
        self.lr = learning_rate
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50000], gamma=0.1, verbose=True)

    def make_actions(self, cur_state, is_evaluation=False):
        # 选择动作
        state = cur_state[0]
        t = cur_state[1]
        chosen_type = cur_state[2]
        if chosen_type is None:
            probs = self.net(t, state, is_inference=is_evaluation)
        else:
            probs = self.net(t, state, chosen_type, is_inference=is_evaluation)
        if is_evaluation == False:
            action_dist = Categorical(probs)
            action_index = action_dist.sample()
            log_prob = action_dist.log_prob(action_index)
        else:
            action_index = probs.argmax().item()
            log_prob = None
        return action_index, log_prob, probs

    def evaluate(self,meta_mode=True):
        success_rate = 0
        logits_history = []
        labels_history = []
        action_list = {}
        print("Evaluating....")
        if meta_mode:
            size = len(self.idx_meta)
            wrong_logits = self.wrong_logits
            wrong_labels = self.wrong_labels
        else:
            size = len(self.idx_eval)
            wrong_logits = self.wrong_eval_logits
            wrong_labels = self.wrong_eval_labels
        for i in tqdm(range(size)):
            if meta_mode:
                selected_idx = self.idx_meta[i]
            else:
                selected_idx = self.idx_eval[i]
            self.env.setup(selected_idx)
            self.net.update_list_action_space(self.env.list_action_space)  # 初始化动作空间
            action_choice = []
            for t in range(self.num_mod * 2):
                cur_state = self.env.getStateRef()
                actions, _, _ = self.make_actions(cur_state, is_evaluation=True)
                actions = self.env.step(actions, evaluation = True)
                action_choice.append(actions)
                if t % 2 == 1:
                    reward = self.env.rewards
                    if reward > 0:
                        success_rate += 1
                    if self.env.dones:
                        break
                else:
                    continue
            logits_history.append(self.env.evaluation_logits) # 这里的logist 是victim model的结果
            labels_history.append(self.env.evaluation_labels)
            action_list[selected_idx] = action_choice

        logits_history = torch.from_numpy(np.stack(logits_history))
        if self.num_wrong != 0:
            logits_history = torch.cat((logits_history, wrong_logits), dim=0)
        labels_history = torch.from_numpy(np.array([element.item() for element in labels_history]))
        if self.num_wrong != 0:
            labels_history = torch.cat((labels_history, wrong_labels), dim=0)
        tar_acc, tar_micro_f1_atk, tar_macro_f1_atk = score(logits_history, labels_history, self.num_wrong)
        success_rate = success_rate / size
        print("success_rate is ", success_rate)
        print("accuracy is ", tar_acc)

        with open("average_history.txt", "w") as file:
            data_line = f"Budget: {self.num_mod}, Accuracy: {tar_acc}, Micro F1 Score: {tar_micro_f1_atk}, Macro F1 Score: {tar_macro_f1_atk}\n"
            print(data_line)
            file.write(data_line)
        return success_rate

    def update(self, transition_dict, current_step, total_steps):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_index_list = transition_dict['action_indexs']
        log_probs = torch.stack(transition_dict['log_probs'])
        all_probs = transition_dict['all_probs']
        R = 0
        rewards = []
        for r in reward_list[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)

        rewards = torch.FloatTensor(rewards).to(self.device)

        loss = torch.sum(torch.mul(log_probs.view(-1), Variable(rewards)).mul(-1), -1).to(
            self.device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()  #

    def test_on_policy_agent(self):
        model_name = 'AgentModel/model_'+ self.dataset +'_budget_'+ str(self.num_mod) +'.pth'
        pretrained_state_dict = torch.load(model_name)

        self.net.load_state_dict(pretrained_state_dict)
        success_rate = self.evaluate(meta_mode=False)
        k_value = self.env.k_value
        with open('k_valueAndSuccessRate.txt', 'w') as file:
            file.write(f"{self.dataset}\t{self.num_mod}\t{k_value}\t{success_rate}\n")

    def train_on_policy_agent(self, num_steps=10000):
        torch.autograd.set_detect_anomaly(True)
        return_list = []
        current_step = 0
        epoch_num = 100
        for i in range(epoch_num):
            self.evaluate()
            with tqdm(total = int(num_steps / epoch_num), desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(num_steps/epoch_num)):
                    episode_return = 0
                    transistion_dict = {'states': [], "action_indexs": [], "log_probs": [], "next_states": [],
                                        "rewards": [], "dones": [], "all_probs": []}
                    if (self.pos + 1) * self.batch_size > len(self.idx_test):
                        self.pos = 0

                    random.shuffle(self.idx_test)
                    selected_idx = self.idx_test[self.pos * self.batch_size]
                    self.pos += 1
                    self.env.setup(selected_idx)
                    self.net.update_list_action_space(self.env.list_action_space)
                    state = self.env.getStateRef()
                    done = False
                    while not done:
                        action_index, log_prob, prob = self.make_actions(state)
                        current_state = deepcopy(state)
                        self.env.step(action_index)
                        next_state = self.env.getStateRef()
                        done = self.env.dones
                        reward = self.env.rewards
                        transistion_dict['states'].append(deepcopy(current_state))
                        transistion_dict['action_indexs'].append(action_index)
                        transistion_dict['next_states'].append(deepcopy(next_state))
                        transistion_dict['rewards'].append(reward)
                        transistion_dict['dones'].append(done)
                        transistion_dict['log_probs'].append(log_prob)
                        transistion_dict['all_probs'].append(prob)
                        state = next_state
                        episode_return += reward
                    return_list.append(episode_return)
                    self.update(transistion_dict, current_step, num_steps)
                    current_step += 1
                    if (i_episode + 1) % 100 == 0:
                        pbar.set_postfix(
                            {'episode': '%d' % (num_steps / 10 * i + i_episode + 1),
                             'return': '%.3f' % np.mean(return_list[-10:])})
                    pbar.update(1)



