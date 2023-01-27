import numpy as np
import pandas as pd
import random
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()



class matlqlNetwork:
    def __init__(
            self,
            sess, 
            n_actions,
            n_features,
            n_advisors,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            model_path=False,
    ):
        self.n_actions = n_actions
        self.n_advisors = n_advisors 
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.value = 0
        self.learn_step_counter = 0

        self.memory = np.zeros((self.memory_size, n_features * 2 + 4))
        self.memory2 = np.zeros((self.memory_size, n_features * 2 + 4))

        self._build_net()
        self._build_net2()
        self._build_net3()
        t_params = tf.get_collection('matlqltarget_net_params')
        e_params = tf.get_collection('matlqleval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        t2_params = tf.get_collection('matlql_advisor_net_params')
        e2_params = tf.get_collection('matlql_advisor_target_net_params')
        self.replace_target_op2 = [tf.assign(t, e) for t, e in zip(t2_params, e2_params)]
        
        self.sess = sess

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.cost_his = []



    def copy_network(self, s):
        
        saver = tf.train.Saver()
        saver.restore(self.sess, s)
        print("New model copied")

    def _build_net(self):
        self.s = tf.placeholder(tf.float32, [None, self.n_features+1], name='matlqls1')  
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='matlqlQ_target_1')  
        
        with tf.variable_scope('matlqlValue'):
            self.name_scope = tf.get_variable_scope().name
            with tf.variable_scope('matlqleval_net_1'):
                c_names, n_l1, w_initializer, b_initializer = \
                    ['matlqleval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 50, \
                    tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  

                with tf.variable_scope('matlqll_1'):
                    w1 = tf.get_variable('matlqlw_1', [self.n_features+1, n_l1], initializer=w_initializer, collections=c_names)
                    b1 = tf.get_variable('matlqlb_1', [1, n_l1], initializer=b_initializer, collections=c_names)
                    l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

                with tf.variable_scope('matlql_lh1'):
                    wh1 = tf.get_variable('wh1', [n_l1, n_l1], initializer=w_initializer, collections=c_names)
                    bh1 = tf.get_variable('bh1', [1, n_l1], initializer=b_initializer, collections=c_names)
                    lh1 = tf.nn.relu(tf.matmul(l1, wh1) + bh1)
                
                with tf.variable_scope('matlqll_2'):
                    w2 = tf.get_variable('matlqlw_2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('matlqlb_2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    self.q_eval = tf.matmul(lh1, w2) + b2

            with tf.variable_scope('matlqlloss'):
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
            with tf.variable_scope('matlqltrain'):
                self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

            self.s_ = tf.placeholder(tf.float32, [None, self.n_features+1], name='matlqls1_')    
            
            
            with tf.variable_scope('matlqltarget_net_1'):
                c_names = ['matlqltarget_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

                with tf.variable_scope('matlqll_1'):
                    w1 = tf.get_variable('matlqlw_1', [self.n_features+1, n_l1], initializer=w_initializer, collections=c_names)
                    b1 = tf.get_variable('matlqlb_1', [1, n_l1], initializer=b_initializer, collections=c_names)
                    l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)
                
                with tf.variable_scope('matlql_lh1'):
                    wh1 = tf.get_variable('wh1', [n_l1, n_l1], initializer=w_initializer, collections=c_names)
                    bh1 = tf.get_variable('bh1', [1, n_l1], initializer=b_initializer, collections=c_names)
                    lh1 = tf.nn.relu(tf.matmul(l1, wh1) + bh1)

                with tf.variable_scope('matlqll_2'):
                    w2 = tf.get_variable('matlqlw_2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('matlqlb_2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    self.q_next = tf.matmul(lh1, w2) + b2

    
    def _build_net2(self):
    
        self.q_target2 = tf.placeholder(tf.float32, [None, self.n_advisors], name='matlqlQ_target_2')  
        
        with tf.variable_scope('matlqlAdvisor'):
            self.name_scope2 = tf.get_variable_scope().name
            with tf.variable_scope('matlqladvisor_net_1'):
                c_names, n_l1, w_initializer, b_initializer = \
                    ['matlql_advisor_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 50, \
                    tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  

                with tf.variable_scope('matlqll'):
                    w1 = tf.get_variable('matlqlw', [self.n_features+1, n_l1], initializer=w_initializer, collections=c_names)
                    b1 = tf.get_variable('matlqlb', [1, n_l1], initializer=b_initializer, collections=c_names)
                    l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

                with tf.variable_scope('matlqllh1'):
                    wh1 = tf.get_variable('wh1', [n_l1, n_l1], initializer=w_initializer, collections=c_names)
                    bh1 = tf.get_variable('bh1', [1, n_l1], initializer=b_initializer, collections=c_names)
                    lh1 = tf.nn.relu(tf.matmul(l1, wh1) + bh1)
                
                with tf.variable_scope('matlqll2'):
                    w2 = tf.get_variable('matlqlw2', [n_l1, self.n_advisors], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('matlqlb2', [1, self.n_advisors], initializer=b_initializer, collections=c_names)
                    self.q_advisor = tf.matmul(lh1, w2) + b2

            with tf.variable_scope('matlqlloss2'):
                self.loss2 = tf.reduce_mean(tf.squared_difference(self.q_target2, self.q_advisor))
            with tf.variable_scope('matlqltrain2'):
                self._train_op2 = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss2)
        
            with tf.variable_scope('matlql_advisor_target_net_1'):
                c_names = ['matlql_advisor_target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

                with tf.variable_scope('matlqll_1'):
                    w1 = tf.get_variable('matlqlw_1', [self.n_features+1, n_l1], initializer=w_initializer, collections=c_names)
                    b1 = tf.get_variable('matlqlb_1', [1, n_l1], initializer=b_initializer, collections=c_names)
                    l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)
                
                with tf.variable_scope('matlql_lh1'):
                    wh1 = tf.get_variable('wh1', [n_l1, n_l1], initializer=w_initializer, collections=c_names)
                    bh1 = tf.get_variable('bh1', [1, n_l1], initializer=b_initializer, collections=c_names)
                    lh1 = tf.nn.relu(tf.matmul(l1, wh1) + bh1)

                with tf.variable_scope('matlqll_2'):
                    w2 = tf.get_variable('matlqlw_2', [n_l1, self.n_advisors], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('matlqlb_2', [1, self.n_advisors], initializer=b_initializer, collections=c_names)
                    self.q_advisor_next = tf.matmul(lh1, w2) + b2
            
    
    def _build_net3(self):

        self.previous_action = tf.placeholder(tf.float32, [None, self.n_actions], name='previousaction')
        self.target_action = tf.placeholder(tf.float32, [None, self.n_actions], name='target_action')
        self.s_new = tf.placeholder(tf.float32, [None, self.n_features], name='s_new')

        with tf.variable_scope('opponentnetwork'):
            self.name_scope3 = tf.get_variable_scope().name

            with tf.variable_scope('opponentaction_net_1'):
                c_names, n_l1, w_initializer, b_initializer = \
                    ['opponent_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 50, \
                    tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

                with tf.variable_scope('oppactionl'):
                    w1 = tf.get_variable('oppactionw', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                    b1 = tf.get_variable('oppactionb', [1, n_l1], initializer=b_initializer, collections=c_names)
                    l1 = tf.nn.relu(tf.matmul(self.s_new, w1) + b1)


                with tf.variable_scope('previousoppactionl_3'):
                    w3 = tf.get_variable('previousoppactionw_3', [self.n_actions, n_l1], initializer=w_initializer, collections=c_names)
                    b3 = tf.get_variable('previousoppactionb_3', [1, n_l1], initializer=b_initializer, collections=c_names)
                    l3 = tf.nn.relu(tf.matmul(self.previous_action, w3) + b3)

                concat_layer = tf.concat([l1, l3], axis=1)


                with tf.variable_scope('oppactionl_h1'):
                    wh1 = tf.get_variable('oppactionw_h1', [n_l1*2, n_l1], initializer=w_initializer, collections=c_names)
                    bh1 = tf.get_variable('oppactionb_h1', [1, n_l1], initializer=b_initializer, collections=c_names)
                    lh1 = tf.nn.relu(tf.matmul(concat_layer, wh1) + bh1)

                with tf.variable_scope('oppactionl_h2'):
                    wh2 = tf.get_variable('oppactionw_h2', [n_l1, n_l1], initializer=w_initializer, collections=c_names)
                    bh2 = tf.get_variable('oppactionb_h2', [1, n_l1], initializer=b_initializer, collections=c_names)
                    lh2 = tf.nn.relu(tf.matmul(lh1, wh2) + bh2)



                with tf.variable_scope('oppactionl2'):
                    w2 = tf.get_variable('oppactionw2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('oppactionb2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    self.new_action = tf.matmul(lh2, w2) + b2

            with tf.variable_scope('oppactionloss'):
                self.loss3 = tf.losses.softmax_cross_entropy(self.target_action, self.new_action)
            with tf.variable_scope('oppactiontrain'):
                self._train_op3 = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss3)

 
    
    def store_transition(self, s, a, a1, a2, r, s_, b):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
            self.memory_counter2 = 0
        
        s = list(s)
        a1 = float(a1)
        s.append(a1)
        s=np.array(s)
        s_ = list(s_)
        a2 = float(a2)
        s_.append(a2)
        s_=np.array(s_)


        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1
        
        if b < self.n_advisors: 
            transition2 = np.hstack((s, [r, b], s_))
            index = self.memory_counter2 % self.memory_size
            self.memory2[index, :] = transition2

            self.memory_counter2 += 1
        

    def get_opp_action(self, observation, a1):


        a1 = np.array(a1)
        a1 = np.eye(self.n_actions)[a1]
        observation = np.expand_dims(observation, axis=0)
        a1 = np.expand_dims(a1, axis=0)
        opp_action = self.sess.run(self.new_action, feed_dict={self.s_new: observation, self.previous_action: a1})
        opp_action = opp_action[0]
        opp_action = np.argmax(opp_action)

        return opp_action


    def learn_opp_action(self, observation, a1, previous_action):

        a1 = np.array(a1)
        a1 = np.eye(self.n_actions)[a1]
        a1 = np.expand_dims(a1, axis=0)

        previous_action = np.array(previous_action)
        previous_action = np.eye(self.n_actions)[previous_action]
        previous_action = np.expand_dims(previous_action, axis=0)

        observation = np.expand_dims(observation, axis=0)
        _, self.cost3 = self.sess.run([self._train_op3, self.loss3], feed_dict={self.s_new: observation, self.previous_action: previous_action, self.target_action:a1})
        return self.cost3








    def choose_advisor(self, observation, a2, list_actions, list_frequency, execution=False):
        
        if execution==True:
            self.epsilon = 1
        
        observation = list(observation)
        a2 = float(a2)
        observation.append(a2)
        observation=np.array(observation)
        observation = observation[np.newaxis, :]
        
        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_advisor, feed_dict={self.s: observation})
            actions_value = actions_value[0]
            value_dict = {}  
            seen = set() 
            dupes = [] 
            i = 0
            for x in list_actions: 
                value_dict[x] = actions_value[i]
                i = i + 1
                if x in seen: 
                    dupes.append(x) 
                else: 
                    seen.add(x) 
            seen_list = list(seen)

            
            for i in range(len(dupes)): 
                new_list = [] 
                for j in range(len(list_actions)): 
                    if list_actions[j] == dupes[i]: 
                        new_list.append(actions_value[j])
                
                max_value = max(new_list)
                max_index = new_list.index(max_value)
                sum_value = max_value
                for k in range(len(new_list)): 
                    if k!=max_index: 
                        sum_value = sum_value + list_frequency * new_list[k]     
                
                value_dict[dupes[i]] = sum_value

            prob_list = []
            key_list = []
            for key in value_dict: 
                key_list.append(key)
                prob_list.append(value_dict[key])
            
            prob_list = [float(i)/sum(prob_list) for i in prob_list]
            prob_max = max(prob_list)
            prob_index = prob_list.index(prob_max)
            advisor_action = key_list[prob_index]
            indices = [i for i, x in enumerate(list_actions) if x == advisor_action]
            new_action_value = [actions_value[i] for i in indices]
            max_value = max(new_action_value)
            new_index, = np.where(actions_value == max_value) 
            advisor = new_index[0]



        else:
            advisor = np.random.randint(0, self.n_advisors)
        
        
        return advisor
    
    
    def choose_action(self, observation, a2, execution=False):


        if execution==True:
            self.epsilon = 1
        
        observation = list(observation)
        a2 = float(a2)
        observation.append(a2)
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
            self.sess.run(self.replace_target_op2)

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]


        if self.memory_counter2 > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter2, size=self.batch_size)
        batch_memory2 = self.memory2[sample_index, :]

        
        
        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -(self.n_features+1):],  
                self.s: batch_memory[:, :(self.n_features+1)],  
            })

        q_advisornext, q_advisor = self.sess.run(
            [self.q_advisor_next, self.q_advisor],
            feed_dict={
                self.s_: batch_memory2[:, -(self.n_features+1):],  
                self.s: batch_memory2[:, :(self.n_features+1)],  
            })

        

        q_target = q_eval.copy()
        q_target2 = q_advisor.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features+1].astype(int)
        reward = batch_memory[:, self.n_features+2]
        
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
        
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        reward = batch_memory2[:, self.n_features+1]
        e = batch_memory2[:, self.n_features+2].astype(int)
        q_target2[batch_index, e] = reward + self.gamma * q_advisornext[batch_index, e]
        
        
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :(self.n_features+1)],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)  
        

        
        
        

        _, self.cost2 = self.sess.run([self._train_op2, self.loss2],
                                     feed_dict={self.s: batch_memory2[:, :(self.n_features+1)],
                                                self.q_target2: q_target2})
        
        
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



    def save_model_advisor(self, s):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope2)
        saver = tf.train.Saver(model_vars)
        save_path = saver.save(self.sess, s)
        print("Model saved in path: %s" % save_path)

    def restore_model_advisor(self, s):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope2)
        saver = tf.train.Saver(model_vars)
        saver.restore(self.sess, s)
        print("Model restored")

    def save_model_advisor(self, s):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope3)
        saver = tf.train.Saver(model_vars)
        save_path = saver.save(self.sess, s)
        print("Model saved in path: %s" % save_path)

    def restore_model_advisor(self, s):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope3)
        saver = tf.train.Saver(model_vars)
        saver.restore(self.sess, s)
        print("Model restored")

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()



