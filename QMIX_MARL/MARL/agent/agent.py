import numpy as np
import torch
from MARL.policy.vdn import VDN
from MARL.policy.qmix import QMIX
from MARL.policy.coma import COMA
from MARL.policy.reinforce import Reinforce
from MARL.policy.central_v import CentralV
from MARL.policy.qtran_alt import QtranAlt
from MARL.policy.qtran_base import QtranBase
from MARL.policy.maven import MAVEN
from torch.distributions import Categorical


# Agent no communication
class Agents:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        if args.alg == 'vdn':
            self.policy = VDN(args)
        elif args.alg == 'qmix':
            self.policy = QMIX(args)
        elif args.alg == 'coma':
            self.policy = COMA(args)
        elif args.alg == 'qtran_alt':
            self.policy = QtranAlt(args)
        elif args.alg == 'qtran_base':
            self.policy = QtranBase(args)
        elif args.alg == 'maven':
            self.policy = MAVEN(args)
        elif args.alg == 'central_v':
            self.policy = CentralV(args)
        elif args.alg == 'reinforce':
            self.policy = Reinforce(args)
        else:
            raise Exception("No such algorithm")
        self.args = args
        # print('Init Agents')

    def agents_copy(self,policy):
        self.policy.deep_copy(policy)

    # 有可能分布选出来不符合要求的动作,目前貌似用不到这个东西
    def random_choice_with_mask(self, avail_actions):
        # temp = []
        temp = [i for i,j in  enumerate(avail_actions) if j==1]
        np.random.seed(None)
        return np.random.choice(temp, 1, False)[0]

    def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon, maven_z=None, evaluate=False):
        inputs = obs.copy()
        # avail_actions_ind = np.nonzero(avail_actions)[0]  # index of actions which can be choose

        # transform agent_num to onehot vector
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1.

        if self.args.last_action:
            inputs = np.hstack((inputs, last_action))
        if self.args.reuse_network:
            inputs = np.hstack((inputs, agent_id))
        hidden_state = self.policy.eval_hidden[:, agent_num, :]

        # transform the shape of inputs from (42,) to (1,42)
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)  # [[]]  -> []
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
            hidden_state = hidden_state.cuda()

        # get q value
        if self.args.alg == 'maven':
            maven_z = torch.tensor(maven_z, dtype=torch.float32).unsqueeze(0)
            if self.args.cuda:
                maven_z = maven_z.cuda()
            q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state, maven_z)
        else:
            q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state)

        # choose action from q value
        # print("")
        if self.args.alg == 'coma' or self.args.alg == 'central_v' or self.args.alg == 'reinforce':
            action = self._choose_action_from_softmax(q_value.cpu(), avail_actions, epsilon, evaluate)
        else:
            # print(avail_actions.shape, q_value.shape)
            q_value[avail_actions == 0.0] = - float("inf")
            # print("epsilon："+str(epsilon))
            if np.random.uniform() < epsilon:
                action = self.random_choice_with_mask(avail_actions[0])
            else:
                action = torch.argmax(q_value).cpu()  # 此处应该判断一下是不是都是-inf
                # print(66666,action,q_value)
        return action

    def _choose_action_from_softmax(self, inputs, avail_actions, epsilon, evaluate=False):
        """
        :param inputs: # q_value of all actions
        """
        action_num = avail_actions.sum(dim=1, keepdim=True).float().repeat(1, avail_actions.shape[-1])  # num of avail_actions
        # 先将Actor网络的输出通过softmax转换成概率分布
        prob = torch.nn.functional.softmax(inputs, dim=-1)
        # add noise of epsilon
        prob = ((1 - epsilon) * prob + torch.ones_like(prob) * epsilon / action_num)
        prob[avail_actions == 0] = 0.0  # 不能执行的动作概率为0

        """
        不能执行的动作概率为0之后，prob中的概率和不为1，这里不需要进行正则化，因为torch.distributions.Categorical
        会将其进行正则化。要注意在训练的过程中没有用到Categorical，所以训练时取执行的动作对应的概率需要再正则化。
        """

        if epsilon == 0 and evaluate:
            action = torch.argmax(prob)
        else:
            action = Categorical(prob).sample().long()
        return action

    # 这个函数有问题
    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[0]
        max_episode_len = 0
        # 由于episodelimit的长度内没有terminal==1，所以导致max_episode_len == 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        if max_episode_len == 0:  # 防止所有的episode都没有结束，导致terminated中没有1
            max_episode_len = self.args.episode_limit
        return max_episode_len

    def train(self, batch, train_step, epsilon=None):  # coma needs epsilon for training

        # different episode has different length, so we need to get max length of the batch
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            if key != 'z':
                batch[key] = batch[key][:, :max_episode_len]

        self.policy.learn(batch, max_episode_len, train_step, epsilon)
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            print("\n开始保存模型", train_step, self.args.save_cycle)
            self.policy.save_model(train_step)