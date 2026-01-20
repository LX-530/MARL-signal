import copy
import csv
import re

import numpy as np
import os
from MARL.common.rollout import RolloutWorker, CommRolloutWorker
from MARL.agent.agent import Agents #, CommAgents
from MARL.common.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
import sys
from utils.CTM.utils import load_all_firebytime, multiTask
# import torch
# from memory_profiler import profile

class Runner:
    def __init__(self, env, args):
        self.env = env
        self.fire = load_all_firebytime()

        if args.alg.find('commnet') > -1 or args.alg.find('g2anet') > -1:  # communication agent
            # self.agents = CommAgents(args)
            self.rolloutWorker = CommRolloutWorker(env, self.agents, args)
        else:  # no communication agent
            self.agents = Agents(args)
            self.rolloutWorker = RolloutWorker(env, self.agents, args)
        if args.learn and args.alg.find('coma') == -1 and args.alg.find('central_v') == -1 and args.alg.find('reinforce') == -1:  # these 3 algorithms are on-poliy
            self.buffer = ReplayBuffer(args)
        self.args = args
        self.win_rates = []
        self.episode_rewards = []

    # @profile
    def run(self, num):
        train_steps = 0
        for_gantt_data =[]
        # print('Run {} start'.format(num))
        r_s = [0]
        for epoch in range(self.args.n_epoch):
            # 显示输出
            text = '\rRun {}, train epoch {}, ave_rewards {}'
            sys.stdout.write(text.format(num, epoch, sum(r_s)/len(r_s)))
            sys.stdout.flush()

            # print('Run {}, train epoch {}'.format(num, epoch), flush=False)
            if epoch % self.args.evaluate_cycle == 0 and epoch != 0:
                step_len, episode_reward = self.evaluate(epoch)
                # print('win_rate is ', win_rate)
                self.win_rates.append(step_len)
                print("\nepisode_reward:", episode_reward, "epoch:", epoch)
                self.episode_rewards.append(episode_reward)
                # self.plt(num)
                with open(self.args.save_path + "/historydata/evaluate_reward.txt", "a") as f:
                    print(episode_reward, file=f)

            # 收集self.args.n_episodes个episodes
            MT = multiTask(self.args.n_episodes,self.env,self.args, self.rolloutWorker.agents.policy,self.fire) #启用多线程
            episodes, r_s = MT.run_simulation(epoch)
            
            # episode的每一项都是一个(1, episode_len, n_agents, 具体维度)四维数组，下面要把所有episode的obs拼在一起
            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
            if self.args.alg.find('coma') > -1 or self.args.alg.find('central_v') > -1 or self.args.alg.find('reinforce') > -1:
                self.agents.train(episode_batch, train_steps, self.rolloutWorker.epsilon)
                train_steps += 1
            else:
                # 这几个类型的算法需要进行buffer的存储
                self.buffer.store_episode(episode_batch)

                for train_step in range(self.args.train_steps):
                    mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                    self.agents.train(mini_batch, train_steps)
                    train_steps += 1
            if self.args.epsilon_anneal_scale == 'epoch':
                # if episode_num == 0:
                self.args.epsilon = self.args.epsilon - self.args.anneal_epsilon if self.args.epsilon > self.args.min_epsilon else  self.args.epsilon
        self.plt(num)

    def evaluate(self,epoch):
        win_number = 0
        episode_rewards = 0
        for e_epoch in range(self.args.evaluate_epoch):
            _, episode_reward, win_tag, for_gant = self.rolloutWorker.generate_episode(epoch=epoch,fire_info=self.fire,evaluate=True)
            episode_rewards += episode_reward
            if win_tag:
                win_number += 1
        # 返回的是平均获胜次数和平均奖励
        # print(for_gant)
        return win_tag, episode_rewards / self.args.evaluate_epoch

    def plt(self, num):
        plt.figure()
        plt.axis([0, self.args.n_epoch, 0, 100])
        plt.cla()
        plt.subplot(2, 1, 1)
        plt.plot(range(len(self.win_rates)), self.win_rates)
        plt.xlabel('epoch*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('finished_step')

        plt.subplot(2, 1, 2)
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        plt.xlabel('epoch*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('episode_rewards')

        plt.savefig(self.args.save_path + '/plt_{}.png'.format(num), format='png')
        with open(self.args.save_path + '/win_rates_{}.csv'.format(num), 'w', newline='') as file:
            writer = csv.writer(file)
            for j in self.win_rates:
                writer.writerow([j])
        # print(self.episode_rewards)
        with open(self.args.save_path + '/episode_rewards_{}.csv'.format(num), 'w', newline='') as file:
            writer = csv.writer(file)
            for i in self.episode_rewards:
                writer.writerow([i])