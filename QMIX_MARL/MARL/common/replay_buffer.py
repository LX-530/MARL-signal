import numpy as np
import threading


class ReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.n_actions = self.args.n_actions
        self.n_agents = self.args.n_agents
        self.state_shape = self.args.state_shape
        self.obs_shape = self.args.obs_shape
        self.size = self.args.buffer_size
        self.episode_limit = self.args.episode_limit
        # memory management
        self.current_idx = 0
        self.current_size = 0
        # create the buffer to store info
        self.buffers = {'o': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape],dtype='float16'),
                        'u': np.empty([self.size, self.episode_limit, self.n_agents, 1],dtype='float16'),
                        's': np.empty([self.size, self.episode_limit, self.state_shape],dtype='float16'),
                        'r': np.empty([self.size, self.episode_limit, 1],dtype='float16'),
                        'o_next': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape],dtype='float16'),
                        's_next': np.empty([self.size, self.episode_limit, self.state_shape],dtype='float16'),
                        'avail_u': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions],dtype='float16'),
                        'avail_u_next': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions],dtype='float16'),
                        'u_onehot': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions],dtype='float16'),
                        'padded': np.empty([self.size, self.episode_limit, 1],dtype='float16'),
                        'terminated': np.empty([self.size, self.episode_limit, 1],dtype='float16')
                        }
        if self.args.alg == 'maven':
            self.buffers['z'] = np.empty([self.size, self.args.noise_dim])
        # thread lock
        self.lock = threading.Lock()

        #priority of sample
        self.priority = np.ones(self.size,dtype="int16")
        # store the episode
    def store_episode(self, episode_batch):
        batch_size = episode_batch['o'].shape[0]  #
        list1 = np.array([0 if array.shape[0] >= 145 else 1 for array in episode_batch["terminated"]])
        list2 = [index for index, value in enumerate(list1) if value == 1]
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            for ii in list2:
                self.priority[idxs[ii]] = 500  # 增加500%被抽中的概率
            # store the informations
            self.buffers['o'][idxs] = episode_batch['o']
            self.buffers['u'][idxs] = episode_batch['u']
            self.buffers['s'][idxs] = episode_batch['s']
            self.buffers['r'][idxs] = episode_batch['r']
            self.buffers['o_next'][idxs] = episode_batch['o_next']
            self.buffers['s_next'][idxs] = episode_batch['s_next']
            self.buffers['avail_u'][idxs] = episode_batch['avail_u']
            self.buffers['avail_u_next'][idxs] = episode_batch['avail_u_next']
            self.buffers['u_onehot'][idxs] = episode_batch['u_onehot']
            self.buffers['padded'][idxs] = episode_batch['padded']
            self.buffers['terminated'][idxs] = episode_batch['terminated']
            if self.args.alg == 'maven':
                self.buffers['z'][idxs] = episode_batch['z']

    def sample(self, batch_size):
        """
        增加了优先经验回放，提高训练效率
        只有当Buffer满了的时候才使用优先经验回放
        """
        temp_buffer = {}
        if self.current_size  == self.size:
            probabilities = np.array(self.priority) / np.sum(self.priority)
            idx = np.random.choice(len(self.priority), size=batch_size, replace=False, p=probabilities)
        else:
            idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffers.keys():
            temp_buffer[key] = self.buffers[key][idx]
        return temp_buffer

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_idx + inc <= self.size:
            idx = np.arange(self.current_idx, self.current_idx + inc)
            self.current_idx += inc
        elif self.current_idx < self.size:
            overflow = inc - (self.size - self.current_idx)
            idx_a = np.arange(self.current_idx, self.size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])
            self.current_idx = overflow
        else:
            idx = np.arange(0, inc)
            self.current_idx = inc
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx
