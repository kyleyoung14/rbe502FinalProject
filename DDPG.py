import tensorflow as tf
import numpy as np
import os
import shutil
from environment import Environment
import matplotlib.pyplot as plt


np.random.seed(1)
tf.set_random_seed(1)

MAX_EPISODES = 1000
MAX_EP_STEPS = 400
LR_A = 2e-4  # learning rate for actor
LR_C = 2e-4  # learning rate for critic
GAMMA = 0.9  # reward discount
REPLACE_ITER_A = 1100
REPLACE_ITER_C = 1000
MEMORY_CAPACITY = 5000
BATCH_SIZE = 16
VAR_MIN = 0.1
LOAD = False

env = Environment()
STATE_DIM = env.state_dime
ACTION_DIM = env.act_dime
ACTION_BOUND = env.act_limits

# all placeholder for tf
with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s')
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, [None, 1], name='r')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s_')


class Actor(object):
    def __init__(self, sess, action_dim, action_bound, learning_rate, t_replace_iter):
        self.sess = sess
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        with tf.variable_scope('Actor'):
            # input s, output a
            self.a = self._build_net(S, scope='eval_net', trainable=True)

            # input s_, output a, get a_ for critic
            self.a_ = self._build_net(S_, scope='target_net', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.001)
            net = tf.layers.dense(s, 200, activation=tf.nn.relu6,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                  trainable=trainable)
            net = tf.layers.dense(net, 200, activation=tf.nn.relu6,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                  trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l3',
                                  trainable=trainable)
            with tf.variable_scope('a'):
                actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                          name='a', trainable=trainable)
                scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')  # Scale output to -action_bound to action_bound
        return scaled_a

    def learn(self, s):   # batch update
        self.sess.run(self.train_op, feed_dict={S: s})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1

    def choose_action(self, s):
        s = s[np.newaxis, :]    # single state
        return self.sess.run(self.a, feed_dict={S: s})[0]  # single action

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope('A_train'):
            opt = tf.train.RMSPropOptimizer(-self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))


class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, t_replace_iter, a, a_):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        with tf.variable_scope('Critic'):
            # Input (s, a), output q
            self.a = a
            self.q = self._build_net(S, self.a, 'eval_net', trainable=True)

            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(S_, a_, 'target_net', trainable=False)    # target_q is based on a_ from Actor's target_net

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            self.target_q = R + self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):
            self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, a)[0]   # tensor of gradients of each sample (None, a_dim)

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.01)

            with tf.variable_scope('l1'):
                n_l1 = 200
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net = tf.layers.dense(net, 200, activation=tf.nn.relu6,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                  trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l3',
                                  trainable=trainable)
            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)
        return q

    def learn(self, s, a, r, s_):
        self.sess.run(self.train_op, feed_dict={S: s, self.a: a, R: r, S_: s_})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1


class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]


sess = tf.Session()

# Create actor and critic.
actor = Actor(sess, ACTION_DIM, ACTION_BOUND[1], LR_A, REPLACE_ITER_A)
critic = Critic(sess, STATE_DIM, ACTION_DIM, LR_C, GAMMA, REPLACE_ITER_C, actor.a, actor.a_)
actor.add_grad_to_graph(critic.a_grads)

M = Memory(MEMORY_CAPACITY, dims=2 * STATE_DIM + ACTION_DIM + 1)

saver = tf.train.Saver()

path = './'

if LOAD:
    saver.restore(sess, tf.train.latest_checkpoint(path))
else:
    sess.run(tf.global_variables_initializer())


def train():
    var = 5.  # control exploration
    doneCnt = 0
    allRewards = []
    allVar = []
    doneCntTotal = []
    rollAvg25 = [0]*25
    rollAvg100 = [0]*100
    rollAvgRun = [0]*MAX_EPISODES
    allRollAvg25 = []
    allRollAvg100 = []
    allRollAvgRun = []
    rollReward = [0]*25
    allRollReward = []

    for ep in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0

        for t in range(MAX_EP_STEPS):
        # while True:
            # if RENDER:
            #     env.render()

            # Added exploration noise
            a = actor.choose_action(s)
            a = np.clip(np.random.normal(a, var), *ACTION_BOUND)    # add randomness to action selection for exploration
            s_, r, done = env.step(a)
            M.store_transition(s, a, r, s_)

            if M.pointer > MEMORY_CAPACITY:
                var = max([var*.9999, VAR_MIN])    # decay the action randomness
                b_M = M.sample(BATCH_SIZE)
                b_s = b_M[:, :STATE_DIM]
                b_a = b_M[:, STATE_DIM: STATE_DIM + ACTION_DIM]
                b_r = b_M[:, -STATE_DIM - 1: -STATE_DIM]
                b_s_ = b_M[:, -STATE_DIM:]

                critic.learn(b_s, b_a, b_r, b_s_)
                actor.learn(b_s)

            s = s_
            ep_reward += r

            if t == MAX_EP_STEPS-1 or done:
                if done:
                    doneCnt += 1
                    rollAvg25[ep%25] = 1
                    rollAvg100[ep%100] = 1
                    rollAvgRun[ep] = 1
                    result = '| done:%3i' % doneCnt  
                else:
                    rollAvg25[ep%25] = 0
                    rollAvg100[ep%100] = 0
                    rollAvgRun[ep] = 0
                    result = '| --------'

                doneCntTotal.append(doneCnt)

                if ep < 170:
                    allVar.append(var)

                roll25 = sum(rollAvg25)/min(ep+1,25)
                roll100 = sum(rollAvg100)/min(ep+1,100)
                rollRun = sum(rollAvgRun)/(ep+1)
                allRollAvg25.append(roll25)
                allRollAvg100.append(roll100)
                allRollAvgRun.append(rollRun)

                rollReward[ep%25] = int(ep_reward)
                rollRew = sum(rollReward)/min(ep+1,25)
                allRollReward.append(rollRew)

                print('Ep:', ep,
                      result,
                      '| R: %i' % int(ep_reward),
                      '| AvgR: %i ' % rollRew,
                      '| Explore: %.2f' % var,
                      '| Avg25: %.2f' % roll25,
                      '| Avg100: %.2f' % roll100,
                      '| AvgRun: %.2f' % rollRun,
                      '| Goal: ' + str(env.goal)
                      )
                

                break

        if not ep % 100:
            env.printToCSV(ep)

        # Add this episode's reward for later plotting
        allRewards.append(ep_reward)

    env.printToCSV(MAX_EPISODES)
    
    # Plot the rewards
    fig = plt.figure(1)
    plt.plot(allRewards[:200])
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('DDPG Rewards Over Time')
    fig.savefig('rewards.png')
    plt.close(fig)

    # Plot the rewards
    fig = plt.figure(1)
    plt.plot(allRewards[:500])
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('DDPG Rewards Over Time')
    fig.savefig('rewards.png')
    plt.close(fig)

    # Plot the rewards
    fig = plt.figure(1)
    plt.plot(allRewards)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('DDPG Rewards Over Time')
    fig.savefig('rewards.png')
    plt.close(fig)

    # Plot the average reward
    fig = plt.figure(1)
    plt.plot(allRollReward)
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average DDPG Rewards Over Time')
    fig.savefig('avgRewards.png')
    plt.close(fig)

    # Plot the exploration rate
    fig = plt.figure(2)
    plt.plot(allVar)
    plt.xlabel('Episodes')
    plt.ylabel('Exploration Rate')
    plt.title('Exploration Rate Over Time')
    fig.savefig('explore.png')
    plt.close(fig)

    # Plot the doneCnt over time
    fig = plt.figure(3)
    plt.plot(doneCntTotal)
    plt.xlabel('Episodes')
    plt.ylabel('Successes')
    plt.title('Successful Episodes Over Time')
    fig.savefig('done.png')
    plt.close(fig)

    # Plot the rollAvg25 over time
    fig = plt.figure(4)
    plt.plot(allRollAvg25)
    plt.xlabel('Episodes')
    plt.ylabel('Success Rate')
    plt.title('Success Rate Over Last 25 Episodes')
    fig.savefig('rollAvg25.png')
    plt.close(fig)

    # Plot the rollAvg100 over time
    fig = plt.figure(5)
    plt.plot(allRollAvg100)
    plt.xlabel('Episodes')
    plt.ylabel('Success Rate')
    plt.title('Success Rate Over Last 100 Episodes')
    fig.savefig('rollAvg100.png')
    plt.close(fig)

    # Plot the rollAvg25 over time
    fig = plt.figure(6)
    plt.plot(allRollAvgRun)
    plt.xlabel('Episodes')
    plt.ylabel('Success Rate')
    plt.title('Success Rate Over Time')
    fig.savefig('rollAvgRun.png')
    plt.close(fig)

    if os.path.isdir(path): shutil.rmtree(path)
    os.mkdir(path)
    ckpt_path = os.path.join('./'+MODE[n_model], 'DDPG.ckpt')
    save_path = saver.save(sess, ckpt_path, write_meta_graph=False)
    print("\nSave Model %s\n" % save_path)


def eval():
    # env.set_fps(30)
    s = env.reset()
    while True:
        # if RENDER:
        #     env.render()
        a = actor.choose_action(s)
        s_, r, done = env.step(a)
        s = s_

if __name__ == '__main__':
    if LOAD:
        eval()
    else:
        train()
