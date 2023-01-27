import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import gym

np.random.seed(1)
tf.set_random_seed(1)

GAMMA = 0.9     
LR_A = 0.001    
LR_C = 0.01     


class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess
        self.n_actions = n_actions

        self.s = tf.placeholder(tf.float32, [1, n_features], "matlacstate")
        self.a = tf.placeholder(tf.int32, None, "matlacact")
        self.advantage = tf.placeholder(tf.float32, None, "matlacadvantage")


        with tf.variable_scope('matlacactorvalue'):
            self.name_scope = tf.get_variable_scope().name

            with tf.variable_scope('matlacActor'):
                l1 = tf.layers.dense(
                    inputs=self.s,
                    units=50,    
                    activation=tf.nn.relu,
                    kernel_initializer=tf.random_normal_initializer(0., .1),    
                    bias_initializer=tf.constant_initializer(0.1),  
                    name='matlacl1'
                )


                l2 = tf.layers.dense(
                    inputs=l1,
                    units=50,
                    activation=tf.nn.relu,
                    kernel_initializer=tf.random_normal_initializer(0., .1),
                    bias_initializer=tf.constant_initializer(0.1),
                    name='matlaclh1'
                )

                self.acts_prob = tf.layers.dense(
                    inputs=l2,
                    units=n_actions,    
                    activation=tf.nn.softmax,   
                    kernel_initializer=tf.random_normal_initializer(0., .1),  
                    bias_initializer=tf.constant_initializer(0.1),  
                    name='matlacacts_prob'
                )

            with tf.variable_scope('matlacexp_v'):
                log_prob = tf.log(self.acts_prob[0, self.a])
                self.exp_v = tf.reduce_mean(log_prob * self.advantage)


            with tf.variable_scope('matlactrain'):
                self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  

    def learn(self, s, a, target_value, q_values):
        s = s[np.newaxis, :]

        acts_prob = self.sess.run(self.acts_prob,{self.s: s})
        sum_value = 0
        for i in range(0, self.n_actions):
            sum_value = sum_value + acts_prob[0, i] * q_values[0,i]


        advantage = target_value - sum_value
        feed_dict = {self.s: s, self.a: a, self.advantage: advantage}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v



    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   

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



class Actor2(object):
    def __init__(self, sess, n_features, n_advisors, lr=0.001):
        self.sess = sess
        self.n_advisors = n_advisors
        self.s = tf.placeholder(tf.float32, [1, n_features], "matlacstate")
        self.e = tf.placeholder(tf.int32, None, "matlacadvisor")
        self.advantage = tf.placeholder(tf.float32, None, "matlac_advantage2")


        with tf.variable_scope('matlacactorvalue2'):
            self.name_scope = tf.get_variable_scope().name


            with tf.variable_scope('matlacActor2'):
                l1 = tf.layers.dense(
                    inputs=self.s,
                    units=50,    
                    activation=tf.nn.relu,
                    kernel_initializer=tf.random_normal_initializer(0., .1),    
                    bias_initializer=tf.constant_initializer(0.1),  
                    name='matlacl2'
                )
                
                l2 = tf.layers.dense(
                    inputs=l1,
                    units=50,
                    activation=tf.nn.relu,
                    kernel_initializer=tf.random_normal_initializer(0., .1),
                    bias_initializer=tf.constant_initializer(0.1),
                    name='matlaclh2'
                )

                self.acts_prob = tf.layers.dense(
                    inputs=l2,
                    units=n_advisors,    
                    activation=tf.nn.softmax,   
                    kernel_initializer=tf.random_normal_initializer(0., .1),  
                    bias_initializer=tf.constant_initializer(0.1),  
                    name='matlacacts_prob2'
                )

            with tf.variable_scope('matlacexp_v2'):

                log_prob = tf.log(self.acts_prob[0, self.e])
                self.exp_v = tf.reduce_mean(log_prob * self.advantage)


            with tf.variable_scope('matlactrain2'):
                self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  

    def learn(self, s, e, target_value, q_values):
        s = s[np.newaxis, :]
        acts_prob = self.sess.run(self.acts_prob,{self.s: s})
        sum_value = 0
        for i in range(0, self.n_advisors):
            sum_value = sum_value + acts_prob[0, i] * q_values[0,i]

        advantage = target_value - sum_value

        feed_dict = {self.s: s, self.e: e, self.advantage: advantage}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v


    def choose_advisor(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   


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







class Critic(object):
    def __init__(self, sess, n_features, n_actions, lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features+7], "matlacstate")
        self.a = tf.placeholder(tf.int32, None, "matlacact")
        self.q_ = tf.placeholder(tf.float32, [1, 1], "matlacq_next")
        self.r = tf.placeholder(tf.float32, None, 'matlacr')



        with tf.variable_scope('matlaccriticvalue'):
            self.name_scope = tf.get_variable_scope().name

            with tf.variable_scope('matlacCritic'):
                l1 = tf.layers.dense(
                    inputs=self.s,
                    units=50,  
                    activation=tf.nn.relu,  
                    kernel_initializer=tf.random_normal_initializer(0., .1),  
                    bias_initializer=tf.constant_initializer(0.1),  
                    name='matlacl3'
                )
                
                l2 = tf.layers.dense(
                    inputs=l1,
                    units=50,
                    activation=tf.nn.relu,
                    kernel_initializer=tf.random_normal_initializer(0., .1),
                    bias_initializer=tf.constant_initializer(0.1),
                    name='matlaclh3'
                )

                self.q = tf.layers.dense(
                    inputs=l2,
                    units=n_actions,
                    activation=None,
                    kernel_initializer=tf.random_normal_initializer(0., .1),  
                    bias_initializer=tf.constant_initializer(0.1),  
                    name='matlacV'
                )

            with tf.variable_scope('matlacsquared_TD_error'):
                self.td_error = self.r + GAMMA * self.q_ - self.q[0,self.a]
                self.loss = tf.square(self.td_error)    
            with tf.variable_scope('matlaccritictrain'):
                self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, a, opp_a, r, s_, opp_a_):

        s = list(s)

        for i in range(len(opp_a)):
            new_a = opp_a[i]
            new_a = float(new_a)
            s.append(new_a)


        s = np.array(s)
        
        s_ = list(s_)
        
        for i in range(len(opp_a_)):
            new_a = opp_a_[i]
            new_a = float(new_a)
            s_.append(new_a)
        
        s_ = np.array(s_)



        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        q_ = self.sess.run(self.q, {self.s: s_})
        q2_ = np.copy(q_)
        q_ = q_[0]
        q_ = max(q_)
        target_value = r + GAMMA * q_
        q_list = []
        q_list.append(q_)
        q_ = np.asarray(q_list, dtype=np.float32)
        q_ = np.expand_dims(q_, axis=0)
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.a: a, self.q_: q_, self.r: r})
        return target_value, q2_



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



class Critic2(object):
    def __init__(self, sess, n_features, n_advisors, lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features+7], "matlacstate")
        self.ex = tf.placeholder(tf.int32, None, "matlacact")
        self.q_ = tf.placeholder(tf.float32, [1, 1], "matlacq_next")
        self.r = tf.placeholder(tf.float32, None, 'matlacr')



        with tf.variable_scope('matlaccriticvalue2'):
            self.name_scope = tf.get_variable_scope().name

            with tf.variable_scope('matlacCritic2'):
                l1 = tf.layers.dense(
                    inputs=self.s,
                    units=50,  
                    activation=tf.nn.relu,  
                    kernel_initializer=tf.random_normal_initializer(0., .1),  
                    bias_initializer=tf.constant_initializer(0.1),  
                    name='matlacl3'
                )
                
                l2 = tf.layers.dense(
                    inputs=l1,
                    units=50,
                    activation=tf.nn.relu,
                    kernel_initializer=tf.random_normal_initializer(0., .1),
                    bias_initializer=tf.constant_initializer(0.1),
                    name='matlaclh3'
                )

                self.q = tf.layers.dense(
                    inputs=l2,
                    units=n_advisors,
                    activation=None,
                    kernel_initializer=tf.random_normal_initializer(0., .1),  
                    bias_initializer=tf.constant_initializer(0.1),  
                    name='matlacV'
                )

            with tf.variable_scope('matlacsquared_TD_error2'):
                self.td_error = self.r + GAMMA * self.q_ - self.q[0,self.ex]
                self.loss = tf.square(self.td_error)    
            with tf.variable_scope('matlaccritictrain2'):
                self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, ex, opp_a, r, s_, opp_a_):

        s = list(s)

        for i in range(len(opp_a)):
            new_a = opp_a[i]
            new_a = float(new_a)
            s.append(new_a)


        s = np.array(s)
        
        s_ = list(s_)
        
        for i in range(len(opp_a_)):
            new_a = opp_a_[i]
            new_a = float(new_a)
            s_.append(new_a)
        
        
        s_ = np.array(s_)



        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        q_ = self.sess.run(self.q, {self.s: s_})
        q2_ = np.copy(q_)

        q_ = q_[0]
        q_ = q_[ex]
        target_value = r + GAMMA * q_
        q_list = []
        q_list.append(q_)
        q_ = np.asarray(q_list, dtype=np.float32)
        q_ = np.expand_dims(q_, axis=0)



        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.ex: ex, self.q_: q_, self.r: r})
        return target_value, q2_

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
