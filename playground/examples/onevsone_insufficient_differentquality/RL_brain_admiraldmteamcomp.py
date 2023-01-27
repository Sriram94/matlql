import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

np.random.seed(1)
tf.set_random_seed(1)


class AdmiralDMNetwork:
    def __init__(
            self,
            sess, 
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=10,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.learn_step_counter = 0

        self.memory = np.zeros((self.memory_size, n_features * 2 + 9))

        self._build_net()
        t_params = tf.get_collection('Advisorq_target_net_params')
        e_params = tf.get_collection('Advisorq_eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = sess



        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.cost_his = []

    def copy_network(self, s):
        
        saver = tf.train.Saver()
        saver.restore(self.sess, s)
        print("New model copied")



    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features+3], name='Advisorq_s')  
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Advisorq_Q_target')  
        with tf.variable_scope('ADMIRALValue', reuse = tf.AUTO_REUSE):
            self.name_scope = tf.get_variable_scope().name
            with tf.variable_scope('Advisorq_eval_net', reuse = tf.AUTO_REUSE):
                c_names, n_l1, w_initializer, b_initializer = \
                    ['Advisorq_eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 50, \
                    tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  

                with tf.variable_scope('Advisorq_l1', reuse = tf.AUTO_REUSE):
                    w1 = tf.get_variable('Advisorq_w1', [self.n_features+3, n_l1], initializer=w_initializer, collections=c_names)
                    b1 = tf.get_variable('Advisorq_b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                    l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

                with tf.variable_scope('Advisorq_hl1', reuse = tf.AUTO_REUSE):
                    wh1 = tf.get_variable('Advisorq_hw1', [n_l1, n_l1], initializer=w_initializer, collections=c_names)
                    bh1 = tf.get_variable('Advisorq_hb1', [1, n_l1], initializer=b_initializer, collections=c_names)
                    lh1 = tf.nn.relu(tf.matmul(l1, wh1) + bh1)
                
                with tf.variable_scope('Advisorq_l2', reuse = tf.AUTO_REUSE):
                    w2 = tf.get_variable('Advisorq_w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('Advisorq_b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    self.q_eval = tf.matmul(lh1, w2) + b2

            with tf.variable_scope('Advisorq_loss', reuse = tf.AUTO_REUSE):
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
            with tf.variable_scope('Advisorq_train', reuse = tf.AUTO_REUSE):
                self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

            # ------------------ build target_net ------------------
            self.s_ = tf.placeholder(tf.float32, [None, self.n_features+3], name='Advisorq_s_')    
            with tf.variable_scope('Advisorq_target_net', reuse = tf.AUTO_REUSE):
                c_names = ['Advisorq_target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

                with tf.variable_scope('Advisorq_l1', reuse = tf.AUTO_REUSE):
                    w1 = tf.get_variable('Advisorq_w1', [self.n_features+3, n_l1], initializer=w_initializer, collections=c_names)
                    b1 = tf.get_variable('Advisorq_b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                    l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

                with tf.variable_scope('Advisorq_hl1', reuse = tf.AUTO_REUSE):
                    wh1 = tf.get_variable('Advisorq_hw1', [n_l1, n_l1], initializer=w_initializer, collections=c_names)
                    bh1 = tf.get_variable('Advisorq_hb1', [1, n_l1], initializer=b_initializer, collections=c_names)
                    lh1 = tf.nn.relu(tf.matmul(l1, wh1) + bh1)
                
                with tf.variable_scope('Advisorq_l2', reuse = tf.AUTO_REUSE):
                    w2 = tf.get_variable('Advisorq_w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('Advisorq_b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    self.q_next = tf.matmul(lh1, w2) + b2



    def store_transition(self, s, a, a1, a1_, r, s_, a_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        s = list(s)
        for i in range(len(a1)):
            new_a = a1[i]
            new_a = float(new_a)
            s.append(new_a)
        s=np.array(s)
        s_ = list(s_)
        for i in range(len(a1_)):
            new_a = float(a1_[i])
            new_a = float(new_a)
            s_.append(new_a)
        s_=np.array(s_)

        transition = np.hstack((s, [a, r, a_], s_))

        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation, a2, execution=False):
        if execution==True:
            self.epsilon = 1        
        observation = list(observation)
        for i in range(len(a2)):
            new_a = a2[i]
            new_a = float(new_a)
            observation.append(new_a)
        observation=np.array(observation)
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -(self.n_features+3):],  
                self.s: batch_memory[:, :(self.n_features+3)],  
            })

        q_target = q_eval.copy()
        a_ = batch_memory[:, self.n_features + 5].astype(int)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features+3].astype(int)
        reward = batch_memory[:, self.n_features + 4]
        
        q_target[batch_index, eval_act_index] = reward + self.gamma * q_next[0][a_]


        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :(self.n_features+3)],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1


    def save_model(self, s):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)
        save_path = saver.save(self.sess, s)
        print("Model saved in path: %s" % save_path)



    def restore_model(self, s):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)
        saver.restore(self.sess, s)
        print("Model restored")



    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()



