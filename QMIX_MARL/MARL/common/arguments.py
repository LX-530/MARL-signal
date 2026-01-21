import argparse

"""
Here are the param for the training
"""


def get_common_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--difficulty', type=str, default='7', help='the difficulty of the game')
    parser.add_argument('--game_version', type=str, default='latest', help='the version of the game')
    parser.add_argument('--map', type=str, default='dynamicsignal', help='the map of the game')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--step_mul', type=int, default=8, help='how many steps to make an action')
    parser.add_argument('--replay_dir', type=str, default=r'', help='absolute path to save the replay')
    # 一共可以测试13种算法，但是智能体实际上只有8种
    # The alternative algorithms are vdn, coma, central_v, qmix, qtran_base,
    # qtran_alt, reinforce, coma+commnet, central_v+commnet, reinforce+commnet，
    # coma+g2anet, central_v+g2anet, reinforce+g2anet, maven
    parser.add_argument('--alg', type=str, default='qmix', help='the algorithm to train the agent')
    parser.add_argument('--last_action', type=bool, default=True, help='whether to use the last action to choose action')
    parser.add_argument('--reuse_network', type=bool, default=True, help='whether to use one network for all agents')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--optimizer', type=str, default="RMS", help='optimizer')
    parser.add_argument('--evaluate_epoch', type=int, default=20, help='number of the epoch to evaluate the agent')

    # 以下几个参数比较常用,注意只有指定变量名的时候，就会变成true，没法b变false，最好直接在这里改参数吧
    # parser.add_argument('--model_dir', type=str, default='./result/', help='model directory of the policy')
    #/home/dell/桌面/signal_RL/signal_fire5.7/MARL
    parser.add_argument('--result_dir', type=str, default='./result', help='model directory of the policy')
    parser.add_argument('--load_model_dir', type=str, default='./MARL/model', help='result directory of the policy')

    parser.add_argument('--load_model', type=bool, default=True, help='whether to load the pretrained model')
    parser.add_argument('--learn', type=bool, default=True, help='whether to train the model')
    parser.add_argument('--cuda', type=bool, default=False, help='whether to use the GPU')
    parser.add_argument('--havelook', type=bool, default=False, help='whether to have a look of the saruo')

    args = parser.parse_args()
    return args

def get_mixer_args(args):
    # network
    args.rnn_hidden_dim =  82  #82 #128 #64
    args.qmix_hidden_dim = 64 #32
    args.two_hyper_layers =  True
    args.hyper_hidden_dim = 64
    args.qtran_hidden_dim = 64
    args.lr = 1e-6

    # epsilon greedy
    args.epsilon = 0.01
    args.min_epsilon = 0
    anneal_steps = 1200 #3000
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
    args.epsilon_anneal_scale = 'epoch'

    # the number of the epoch to train the agent
    args.n_epoch =  1501#2300 #5000  #15000

    # the number of the episodes in one epoch
    args.n_episodes = 28#28 #120   #必须为使用进程数的倍数

    # the number of the train steps in one epoch
    args.train_steps = 1

    # # how often to evaluate
    args.evaluate_cycle = 1 #10

    args.evaluate_epoch = 1

    # experience replay
    args.batch_size =  320 #320
    args.buffer_size = int(5e3)

    # how often to save the model
    args.save_cycle = 100 #300

    # how often to update the target_net
    args.target_update_cycle =500 # 100

    # QTRAN lambda
    args.lambda_opt = 1
    args.lambda_nopt = 1

    # prevent gradient explosion
    args.grad_norm_clip = 1

    # MAVEN
    args.noise_dim = 16
    args.lambda_mi = 0.001
    args.lambda_ql = 1
    args.entropy_coefficient = 0.001
    return args

# arguments of coma
def get_coma_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'epoch'

    # lambda of td-lambda return
    args.td_lambda = 0.8

    # the number of the epoch to train the agent
    args.n_epoch = 20000

    # the number of the episodes in one epoch
    args.n_episodes = 1

    # how often to evaluate
    args.evaluate_cycle = 100

    # how often to save the model
    args.save_cycle = 5000

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args



# arguments of central_v
def get_centralv_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.05
    args.epsilon_anneal_scale = 'epoch'

    # the number of the epoch to train the agent
    args.n_epoch = 20000

    # the number of the episodes in one epoch
    args.n_episodes = 1

    # how often to evaluate
    args.evaluate_cycle = 100

    # lambda of td-lambda return
    args.td_lambda = 0.8

    # how often to save the model
    args.save_cycle = 5000

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


# arguments of central_v
def get_reinforce_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'epoch'

    # the number of the epoch to train the agent
    args.n_epoch = 20000

    # the number of the episodes in one epoch
    args.n_episodes = 1

    # how often to evaluate
    args.evaluate_cycle = 100

    # how often to save the model
    args.save_cycle = 5000

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


# arguments of coma+commnet
def get_commnet_args(args):
    if args.map == '3m':
        args.k = 2
    else:
        args.k = 3
    return args


def get_g2anet_args(args):
    args.attention_dim = 32
    args.hard = True
    return args
