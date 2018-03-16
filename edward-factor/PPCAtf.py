import tensorflow as tf
import numpy as np
import time
import datetime
from scipy import linalg


class PPCA:
    """
    PPCA using tensorflow to maximize the observed log likelihood
    """

    REVISION = 'v01'

    def __init__(self, Y):
        """
        :param Y: datamatrix of size dxn, possible missing values as np.nan
        """
        print("Initializing PPCA")
        self.Y = Y
        self.d = self.Y.shape[0]
        self.n = self.Y.shape[1]

    def fit(self, dl=1, tol=10 ** -6, max_iter=100000, batch_size=None, debug=False):
        self.dl = dl
        self.tol = tol
        self.max_iter = max_iter
        self.debug = debug
        self.batch_size = batch_size

        # missing indicator, 0 if missing, 1 if present
        S = np.ones((self.d, self.n))
        S[np.isnan(self.Y)] = 0

        if np.sum(S) < self.d*self.n:
            self.mis = True
            self.S = S
            self.Yz = self.Y.copy()
            self.Yz[np.isnan(self.Y)] = 0
            self.__fit_missing()
        else:
            self.mis = False
            self.__fit()

    def __fit_missing(self):

        self.__build_missing_model()

        start = time.time()
        old_loss = float("inf")

        for i in range(self.max_iter):
            loss = self.train_batch_missing(iteration=i)

            if np.abs(loss - old_loss) < self.tol:
                print("break: {0:.2e}".format(np.abs(loss - old_loss)))
                break

            if i % 100 == 0:
                took = time.time() - start
                print("{0}/{1} updates, {2:.2f} s, {3:.2f} train_loss, {4:.2e} loss_diff"
                      .format(i, self.max_iter, took, loss, np.abs(loss - old_loss)))
                start = time.time()
            old_loss = loss

    def __build_missing_model(self):
        print("building missing model")
        tf.reset_default_graph()

        with tf.variable_scope('input'):

            self.y_pl = tf.placeholder(shape=[None, self.d, 1], dtype=tf.float32, name="y_placeholder")
            self.idx_pl = tf.placeholder(shape=[None, self.d, 1], dtype=tf.float32, name="idx_placeholder")
            self.N = tf.shape(self.y_pl)[0]

        with tf.variable_scope('model'):

            self.W = tf.get_variable(shape=[self.d, self.dl], dtype=tf.float32, name="W")
            init = tf.constant(1.0, dtype=tf.float32)
            self.sig2 = tf.get_variable(initializer=init, dtype=tf.float32, name="sig2")
            # self.mu = tf.get_variable(shape=[self.d, 1], dtype=tf.float32, name="mu")
            init = tf.constant(np.nanmean(self.Y, axis=1)[:, np.newaxis], dtype=tf.float32)
            self.mu = tf.get_variable(initializer=init, dtype=tf.float32, name="mu")
            # total number of observed data elements
            Dv = tf.squeeze(tf.reduce_sum(self.idx_pl, axis=1), -1, name="Dvisible_pr_obs")
            Nobs = tf.reduce_sum(Dv)

        with tf.variable_scope('Helpers'):

            # repeated identity matrix, d x d
            Id = tf.tile(tf.eye(self.d, self.d), multiples=[self.N, 1], name="Id_tiled")
            Id = tf.reshape(Id, [self.N, self.d, self.d], name="Id")
            # repeated identity matrix, dl x dl
            Idl = tf.tile(tf.eye(self.dl, self.dl), multiples=[self.N, 1], name="Idl_tiled")
            Idl = tf.reshape(Idl, [self.N, self.dl, self.dl], name="Idl")

            # repeat mu
            mur = tf.tile(self.mu, multiples=[self.N, 1], name="mu_tile")
            mur = tf.reshape(mur, [self.N, self.d, 1], name="mu_reshape")
            mur = tf.multiply(mur, self.idx_pl)

            # repeat W so it becomes [N x d x dl]
            Wr = tf.reshape(self.W, [1, self.d, self.dl])
            Wr = tf.tile(Wr, multiples=[self.N, 1, 1], name="W_tile")

            # repeat idx_pl so it becomes [N x d x dl]
            idx_plr = tf.tile(self.idx_pl, multiples=[1, 1, self.dl])

            # elementwise multiplication of Wr and idx_plr
            self.Wv = tf.multiply(Wr, idx_plr, name="elementwise_W_I")

            # M, see Bishop
            self.M = tf.add(tf.matmul(self.Wv, self.Wv, transpose_a=True, name="WtW"), self.sig2 * Idl)
            self.InvM = tf.matrix_inverse(self.M)

            # sig2 * I adjusted for missing data
            _sig2I = self.sig2 * Id
            # we need to zero missing elements
            # matrix with zeros where there are missing
            Iz = tf.matmul(self.idx_pl, self.idx_pl, transpose_b=True)
            # matrix with zeros in opposite elements
            idx0_pl = tf.ones_like(self.idx_pl, dtype=tf.float32) - self.idx_pl
            IzInv = tf.matmul(idx0_pl, idx0_pl, transpose_b=True)
            # element wise zero missing elements (and make a 1 in the diagonal instead)
            sig2I = tf.multiply(_sig2I, Iz) + tf.multiply(IzInv, Id)
            # TODO: Use sig2I in invM

        # with tf.variable_scope('C'):

            # we can now create one C for each observation
            # self.C = tf.add(tf.matmul(self.Wv, self.Wv, transpose_b=True, name="WWt"), sig2I, name="WWt_sig2I")

        with tf.variable_scope('InvC'):

            # Using tf.matrix_inverse(self.C) gives a lot of trouble when using GPUs.
            # Instead from Bishop:
            self.InvC = tf.reciprocal(self.sig2) * \
                        (Id - tf.matmul(tf.matmul(self.Wv, self.InvM), self.Wv, transpose_b=True))

        with tf.variable_scope('likelihood'):

            # self.logDetC = tf.linalg.logdet(self.C)
            self.logDetInvC = self.d*tf.log(self.sig2) - \
                              tf.linalg.logdet(Id - tf.matmul(tf.matmul(self.Wv, self.InvM), self.Wv, transpose_b=True))

            ym = self.y_pl - mur

            # Observed log likelihood

            # self.ollh = - Dv * 0.5 * tf.log(2 * tf.constant(np.pi)) \
            #             - 0.5 * self.logDetC \
            #             - 0.5 * tf.squeeze(tf.squeeze(tf.matmul(tf.matmul(ym, self.InvC, transpose_a=True), ym), -1), -1)
            self.ollh = - Dv * 0.5 * tf.log(2 * tf.constant(np.pi)) \
                        - 0.5 * self.logDetInvC \
                        - 0.5 * tf.squeeze(tf.reduce_sum(ym * tf.matmul(self.InvC, ym), 1), -1)

            # sum over observations and average over visible data
            self.ollh = tf.reciprocal(Nobs) * tf.reduce_sum(self.ollh, name='average_ollh')

        with tf.variable_scope('negative_loglikelihood'):
            # we wish to minimize the negative ollh
            self.loss = -self.ollh

        self.sess = tf.Session()
        self.global_step = tf.Variable(initial_value=0, trainable=False)

        with tf.variable_scope('trainOP'):
            self.optimizer = tf.train.AdamOptimizer()
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        tf.summary.scalar('Evaluation/negative_ollh', self.loss)
        tf.summary.scalar('Parameters/variance', self.sig2)
        tf.summary.scalar('Parameters/norm_W', tf.norm(self.W))
        tf.summary.scalar('Parameters/norm_mu', tf.norm(self.mu))

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_writer = tf.summary.FileWriter('./tensorboard/pm/miss/{}/{}/'.format(self.REVISION, timestamp), self.sess.graph)
        self.summaries = tf.summary.merge_all()

        if self.debug:
            print("y_pl\t", self.y_pl.shape)
            print("idx_pl\t", self.y_pl.shape)
            print("N\t\t", self.N.shape)
            print("W\t\t", self.W.shape)
            print("sig2\t", self.sig2.shape)
            print("mu\t\t", self.mu.shape)
            print("mur\t\t", mur.shape)
            print("Wr\t\t", Wr.shape)
            print("idx_plr\t", idx_plr.shape)
            print("Wv\t\t", self.Wv.shape)
            print("M\t\t", self.M.shape)
            print("Iz\t", Iz.shape)
            print("IzInv\t", IzInv.shape)
            print("sig2I\t", sig2I.shape)
            print("C\t\t", self.C.shape)
            print("InvC\t", self.InvC.shape)
            print("Dv\t\t", Dv.shape)
            print("Nobs\t", Nobs.shape)
            print("logDetC\t", self.logDetC.shape)
            print("ollh\t", self.ollh.shape)

    def __fit(self):

        self.mu = np.mean(self.Y, axis=1)[:, np.newaxis]
        self.Y = self.Y - self.mu

        self.__build_model()

        start = time.time()
        old_loss = float("inf")

        for i in range(self.max_iter):
            loss = self.train_batch()

            if np.abs(loss - old_loss) < self.tol:
                print("break: {0:.2e}".format(np.abs(loss - old_loss)))
                break

            if i%100 == 0:
                took = time.time() - start
                print("{0}/{1} updates, {2:.2f} s, {3:.2f} train_loss, {4:.2e} loss_diff"
                      .format(i, self.max_iter, took, loss, np.abs(loss - old_loss)))
                start = time.time()
            old_loss = loss

    def __build_model(self):
        print("Building model...")
        tf.reset_default_graph()

        with tf.variable_scope('input'):
            self.y_pl = tf.placeholder(shape=[self.d, None], dtype=tf.float32, name="y_placeholder")
            self.N = tf.cast(tf.shape(self.y_pl)[1], tf.float32)
            const = - self.N * tf.constant(0.5 * self.d * np.log(2 * np.pi), dtype=tf.float32)

        with tf.variable_scope('model'):
            self.W = tf.get_variable(shape=[self.d, self.dl], dtype=tf.float32, name="W")
            init = tf.constant(1.0, dtype=tf.float32)
            self.sig2 = tf.get_variable(initializer=init, dtype=tf.float32, name="sig2")


        with tf.variable_scope('M'):
            self.M = tf.matmul(self.W, self.W, transpose_a=True) + self.sig2 * tf.eye(self.dl)
        with tf.variable_scope('C'):
            self.C = tf.matmul(self.W, self.W, transpose_b=True) + self.sig2 * tf.eye(self.d)
        with tf.variable_scope('InvC'):
            self.InvC =  tf.reciprocal(self.sig2) * (tf.eye(self.d) - tf.matmul(tf.matmul(self.W, tf.matrix_inverse(self.M)), self.W, transpose_b=True ))
        with tf.variable_scope('Cov'):
            self.Cov = tf.reciprocal(self.N) * tf.matmul(self.y_pl, self.y_pl, transpose_b=True)

        self.logDetC = tf.linalg.logdet(self.C)
        # self.logDetInvC = self.d*tf.log(self.sig2) - tf.linalg.logdet(tf.eye(self.d) - tf.matmul(tf.matmul(self.W, tf.matrix_inverse(self.M)), self.W, transpose_b=True ))

        with tf.variable_scope('loglikelihood'):
            # observed average log likelihood
            self.ollh = tf.reciprocal(self.N * self.d) * (const - 0.5 * self.N * (self.logDetC + tf.trace(tf.matmul(self.InvC, self.Cov))))

        with tf.variable_scope('negative_loglikelihood'):
            # we wish to minimize the negative ollh
            self.loss = -self.ollh

        self.sess = tf.Session()
        self.global_step = tf.Variable(initial_value=0, trainable=False)

        with tf.variable_scope('trainOP'):
            self.optimizer = tf.train.AdamOptimizer()
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        tf.summary.scalar('Evaluation/negative_ollh', self.loss)
        tf.summary.scalar('Parameters/variance', self.sig2)
        tf.summary.scalar('Parameters/norm_W', tf.norm(self.W))

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_writer = tf.summary.FileWriter('./tensorboard/pm/nonmiss/{}/{}/'.format(self.REVISION, timestamp),
                                                  self.sess.graph)
        self.summaries = tf.summary.merge_all()

        if self.debug:
            print("y_pl\t", self.y_pl.shape)
            print("N\t", self.N.shape)
            print("W\t", self.W.shape)
            print("sig2\t", self.sig2.shape)
            print("InvC\t", self.InvC.shape)
            print("Cov\t", self.Cov.shape)
            print("ollh\t", self.ollh.shape)

    def train_batch(self):
        y_batch = self.Y
        _, _loss, _summaries, _step = \
            self.sess.run([self.train_op, self.loss, self.summaries, self.global_step],
                          {self.y_pl: y_batch})
        self.train_writer.add_summary(_summaries, _step)
        self.train_writer.flush()
        return _loss

    def train_batch_missing(self, iteration):
        # random batches.
        # TODO: implement real batch delivery
        if self.batch_size is not None:
            ix = np.random.permutation(self.n)
            ix = ix[:self.batch_size]
            y_batch = self.Yz[:, ix].T
            I_batch = self.S[:, ix].T
        else:
            y_batch = self.Yz.T
            I_batch = self.S.T

        y_batch = y_batch[:, :, np.newaxis]
        I_batch = I_batch[:, :, np.newaxis]

        if iteration == 1 and self.debug:
            # Make run-options
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            _, _loss, _summaries, _step = \
                self.sess.run([self.train_op, self.loss, self.summaries, self.global_step],
                              {self.y_pl: y_batch, self.idx_pl: I_batch},
                              options=run_options,
                              run_metadata=run_metadata
                              )

            # Add metadata and note iteration number (label of metadata)
            self.train_writer.add_run_metadata(run_metadata, "step{}".format(iteration))
            self.train_writer.add_summary(_summaries, _step)
            self.train_writer.flush()
            print('Adding run metadata for iteration', iteration)

        else:

            _, _loss, _summaries, _step = \
                self.sess.run([self.train_op, self.loss, self.summaries, self.global_step],
                              {self.y_pl: y_batch, self.idx_pl: I_batch})
            self.train_writer.add_summary(_summaries, _step)
            self.train_writer.flush()

        return _loss

    def get_params(self):
        if self.mis:
            _W, _sig2, _mu = self.sess.run([self.W, self.sig2, self.mu])
        else:
            _W, _sig2 = self.sess.run([self.W, self.sig2])
            _mu = self.mu
        return _W, _sig2, _mu

    def save(self, name):
        print("Saving session...")
        self.saver.save(self.sess, name)

    def load(self, name):
        print("Restoring session...")
        self.saver.restore(self.sess, name)

    @staticmethod
    def subspace(A, B):
        """ Use the 2-norm to compare the angle between two subspaces A and B"""
        a = linalg.orth(A)
        b = linalg.orth(B)
        b = b - a.dot( np.dot(a.T, b) )
        return np.arcsin(np.linalg.norm(b, 2))


if __name__ == '__main__':

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    d = 50
    dl = 10
    N = 1000
    W, r = np.linalg.qr(np.random.rand(d, dl))
    sig2 = 0.1
    mu = np.random.randn(d,1)
    m = 0.005


    z = np.random.randn(dl, N)
    Y = np.dot(W, z) + mu + np.sqrt(sig2) * np.random.randn(d, N)
    ix = np.random.random(Y.shape) < m
    Ynan = Y.copy()
    Ynan[ix] = np.nan
    Yz = Y.copy()
    Yz[ix] = 0

    # missing test
    pcm1 = PPCA(Ynan)
    pcm1.fit(dl=dl)
    W_est1, sig2_est, mu_est = pcm1.get_params()
    # print([W_est, sig2_est])
    # print([W, sig2])

    angle1 = pcm1.subspace(W, W_est1)


    # non-missing test
    pcm2 = PPCA(Y)
    pcm2.fit(dl=dl)
    W_est2, sig2_est2, mu_est2 = pcm2.get_params()
    # print([W_est, sig2_est])
    # print([W, sig2])

    angle2 = pcm2.subspace(W, W_est2)

    print("Missing, Estimated angle: ", angle1)
    print("Non-missing, Estimated angle: ", angle2)

    print("Angle between missing and non missing estimates: ", pcm1.subspace(W_est1, W_est2))

    # print(sig2)
    # print(sig2_est)
    # print(sig2_est2)
    #
    # print(np.mean(Y, axis=1).T)
    # print(mu_est.T)
    # print(mu_est2.T)


