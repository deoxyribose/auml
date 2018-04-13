"""
An implementation of Bayesian Factor Model that expresses an N x D data matrix X as
X = WZ + E
where 
W is a D x K factor loading matrix, W_dk ~ N(0,|tau||lambda_k|)
Z is a K x N latent variable matrix, Z_kn ~ N(0,1)
E is a N x D observation noise matrix, E_nd ~ N(0, sigma_d)
tau, lambda_k
sigma_d ~ Gamma
"""


import edward as ed
import numpy as np
import tensorflow as tf
import pandas as pd
from scipy import linalg
from scipy.stats import norm
from edward.models import Normal, StudentT
from tensorflow.contrib.distributions import Distribution
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, tf.__path__[0] + '/contrib/distributions/python/ops')
import bijectors as bijector

def logNormal(loc=0, scale=1):
	return ed.models.TransformedDistribution(distribution=ed.models.NormalWithSoftplusScale(loc,scale),bijector=bijector.Exp())

class low_rank_mvn_horseshoe_ard():
    def __init__(self,N,M,D):
        self.N = N
        self.M = M
        self.D = D
        K = self.D
        #########
        # P-model
        #########
        tau = StudentT(df=1., loc=0., scale=1.)
        lamda = StudentT(df=1., loc=tf.zeros([K]), scale=tf.ones([K]))
        w = Normal(tf.zeros([D, K]),    tf.reshape(tf.tile(np.abs(tau)*np.abs(lamda),[D]),[D,K]))
        
        z = Normal(tf.zeros([M, K]),    tf.ones([M, K]))
        
        mu = Normal(tf.zeros([D]),       tf.ones([D]))
        sigma = ed.models.Gamma(tf.ones([D])*0.1,tf.ones([D])*0.1)
        x = ed.models.MultivariateNormalDiag(tf.add(tf.transpose(tf.matmul(w, z, transpose_b=True)),mu), tf.reshape(tf.tile(sigma,[M]),[M,D]))

        self.Pmodel = (lamda,tau,w,z,mu,sigma)
        self.x = x
        #########
        # Q-model
        #########
        self.global_lvs = [tf.Variable(tf.random_normal([])),tf.Variable(tf.random_normal([])),
         tf.Variable(tf.random_normal([K])),tf.Variable(tf.random_normal([K])),
         tf.Variable(tf.random_normal([D, K])),tf.Variable(tf.random_normal([D, K])),
         tf.Variable(tf.random_normal([D])),tf.Variable(tf.random_normal([D])),
         tf.Variable(tf.random_normal([D])),tf.Variable(tf.random_normal([D]))]
        qtau = logNormal(self.global_lvs[0],self.global_lvs[1])
        qlamda = logNormal(self.global_lvs[2],self.global_lvs[3])
        qw = Normal(self.global_lvs[4],tf.nn.softplus(self.global_lvs[5]))
        qmu = Normal(self.global_lvs[6],tf.nn.softplus(self.global_lvs[7]))
        qsigma = logNormal(self.global_lvs[8],self.global_lvs[9])
        
        self.local_lvs = [tf.Variable(tf.random_normal([N, K])),tf.Variable(tf.random_normal([N, K]))] 
        self.idx_ph = tf.placeholder(tf.int32, M)
        qz = Normal(loc=tf.gather(self.local_lvs[0], self.idx_ph),scale=tf.nn.softplus(tf.gather(self.local_lvs[1], self.idx_ph)))
        #hidden = Dense(256, activation='tanh')(x_ph)
        #qz = Normal(total_count=1.,probs=Dense(K, activation='linear')(hidden))


        self.Qmodel = (qlamda,qtau,qw,qz,qmu,qsigma)
        
    def initialize(self, x_train):
        '''
        Initialize parameters of Q-model in the solution from PCA, and empirical means and standard deviations, for faster convergence.
        '''
        N = self.N
        D = self.D
        K = D
        x_train = x_train.astype(np.float32)
        qlamda,qtau,qw,qz,qmu,qsigma = self.Qmodel
        
        data_mean = np.mean(x_train,axis=0).astype(np.float32,copy=False)
        qmu = Normal(tf.Variable(data_mean),tf.nn.softplus(tf.Variable(tf.random_normal([D]))))
        
        #data_std = np.std(x_train,axis=0).astype(np.float32,copy=False)
        #qsigma = logNormal(tf.Variable(np.log(data_std))),tf.Variable(tf.random_normal([D])) # this is wrong, see https://en.wikipedia.org/wiki/Log-normal_distribution
        try:
            # x_train might be too big for svd
            _,S,V = np.linalg.svd(x_train-x_train.mean(0), full_matrices=False)
        except MemoryError:
            # in that case, sample 1% of it, or 1000 observations, whichever is bigger
            x_train_subsample = x_train[np.random.choice(N,max(1000,np.floor(0.01*N))),:]
            _,S,V = np.linalg.svd(x_train_subsample-x_train_subsample.mean(0), full_matrices=False)
        #qw = Normal(tf.Variable(np.dot(V,np.diag(S))/np.sqrt(N-1)),tf.nn.softplus(tf.Variable(tf.random_normal([D, K]))))
        self.Qmodel = (qlamda,qtau,qw,qz,qmu,qsigma)

#    def generator(self, arrays, batch_size):
#        # assuming arrays is a list of arrays of equal length, with rows being observations
#        while True:
#            batches = []
#            random_idx = np.random.choice(arrays[0].shape[0], batch_size, replace=False)
#            for array in arrays:
#                batches.append(array[random_idx])
#            yield batches
    
    def next_batch(self,x_train, M):
        M = self.M
        N = x_train.shape[0]
        idx_batch = np.random.choice(N, M)
        return x_train[idx_batch, :], idx_batch


    def infer(self, x_train=None, n_epoch = 100, n_print=100, n_samples=100, optimizer='rmsprop'):
        # get batch-size from P-model
        D = self.D
        M = self.M
        N = x_train.shape[0]
        # make batches and placeholder for a given batch
        self.x_ph = tf.placeholder(tf.float32, [None,D])
        n_batch = int(N / M)

        lamda,tau,w,z,mu,sigma = self.Pmodel
        qlamda,qtau,qw,qz,qmu,qsigma = self.Qmodel

        # add progress bar
        self.inference_global = ed.KLqp({lamda:qlamda,tau:qtau,w:qw,mu:qmu,sigma:qsigma}, data={self.x: self.x_ph, z:qz})
        self.inference_local = ed.KLqp({z:qz}, data={self.x: self.x_ph,lamda:qlamda,tau:qtau,w:qw,mu:qmu,sigma:qsigma})
        # global_step defines a exponentially decaying learning rate schedule 
        #self.inference.initialize(n_iter=n_batch * n_epoch, n_print=(n_batch * n_epoch)/10, n_samples=n_samples, optimizer=optimizer,global_step=tf.Variable(5,trainable=False))
        # no learning rate schedule
        scale_factor = float(N)/M
        self.inference_global.initialize(scale={self.x: scale_factor, z: scale_factor}, var_list=self.global_lvs, n_iter=n_batch * n_epoch, n_print=(n_batch * n_epoch)/10, n_samples=n_samples, optimizer=optimizer)
        self.inference_local.initialize(scale={self.x: scale_factor, z: scale_factor}, var_list=self.local_lvs, n_iter=n_batch * n_epoch, n_print=(n_batch * n_epoch)/10, n_samples=n_samples, optimizer=optimizer)
        
        sess = ed.get_session()
        init = tf.global_variables_initializer()
        init.run()
        self.iterate(x_train)
    
    def iterate(self, x_train=None):
        # Iterate gradient updates until slope of loss is more likely to be non-negative than negative.
        n_iter = self.inference_global.n_iter
        conditions_are_met = True
        loss_is_decreasing = True
        learning_curve = []
        i = 0
        while conditions_are_met:
            x_batch, idx_batch = self.next_batch(x_train, self.M)
            #print(x_batch.var())
            for _ in range(5):
                self.inference_local.update(feed_dict={self.x_ph: x_batch, self.idx_ph: idx_batch})
            info_dict = self.inference_global.update(feed_dict={self.x_ph: x_batch, self.idx_ph: idx_batch})
            window = 100 # recent defined as a window one-tenth-of-all-data wide
            if i%window == 0 and i >= window:
                print(info_dict)
                ### estimate slope of recent loss
                recent_losses = learning_curve[-window:]
                #plt.scatter(np.arange(window),recent_losses)
                # estimate slope by least squares
                m_hat = np.linalg.lstsq(np.vstack([np.arange(window), np.ones(window)]).T,recent_losses)[0][0]
                # estimate standard deviation of losses in window
                s_hat = np.array(recent_losses).std(ddof=2)
                # calculate probability that slope is less than 0
                P_negative_slope = norm.cdf(0,loc=m_hat,scale=12*s_hat**2/(window**3-window))
                # if it is more than .5, loss has been decreasing
                loss_is_decreasing = P_negative_slope > .5
            conditions_are_met = loss_is_decreasing and i < n_iter
            learning_curve.append(info_dict['loss'])
            i += 1
            
        plt.semilogy(learning_curve)
        plt.show()
    
    def prior_predictive_check(self):
        self.x_prior = ed.copy(self.x)
        pass
        
    def posterior_predictive_check(self, x_test):
        self.x_post = ed.copy(self.x, dict(zip(self.Pmodel,self.Qmodel)))
        self.pred_ll = ed.evaluate('log_likelihood', data={self.x_post: x_test})
                    
    def print_model(self):
        qsigma,qlamda,qtau,qw,qz,qmu = self.Qmodel
        # add pair plots
        print("Inferred principal axes (columns):")
        print(qw.mode().eval())
        print(qw.variance().eval())
        print("Inferred center:")
        print(qmu.mean().eval())
        print(qmu.variance().eval())