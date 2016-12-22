from __future__ import print_function

import os
import sys
import timeit
import six.moves.cPickle as pickle
import numpy
import gzip

import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool

from theano.sandbox.rng_mrg import MRG_RandomStreams

class SummaryWriter(object):
  '''
  A wrapper around the Tensorflow Summary writer.
  '''
  tf = __import__('tensorflow')

  def __init__(self, logdir, flush_secs=20):
    self.tf_writer = self.tf.train.SummaryWriter(logdir, flush_secs=flush_secs)

  def add_summary(self, step, **kwargs):
    summary_values = [self.tf.Summary.Value(tag=tag, simple_value=value)
                      for (tag, value) in kwargs.iteritems()]
    self.tf_writer.add_summary(self.tf.Summary(value=summary_values), step)

    # if __name__ == '__main__':
    #   from neural_srl.shared.constants import ROOT_DIR
    #   from os.path import join
    #   writer = SummaryWriter(join(ROOT_DIR, 'temp'))
    #   writer.add_summary(1, v1=0.5, v2=2)
    #   writer.add_summary(2, v1=0.6, v2=4)

def load_data(dataset):
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(
            numpy.asarray(
                data_x,
                dtype=theano.config.floatX))
        shared_y = theano.shared(
            numpy.asarray(
                data_y,
                dtype=theano.config.floatX))
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x,
                                         valid_set_y), (test_set_x, test_set_y)]
    return rval


class HiddenLayer(object):
    def __init__(self, rng, n_in, n_out):
        W_init = numpy.asarray(rng.uniform(-numpy.sqrt(6./(n_in + n_out)), numpy.sqrt(6./(n_in + n_out)), (n_in, n_out)), dtype=theano.config.floatX)
        b_init = numpy.zeros((n_out,), dtype=theano.config.floatX)
        self.W = theano.shared(
            value=W_init,
            name='W',
            borrow=True
        )
        self.b = theano.shared(
            value=b_init,
            name='b',
            borrow=True
        )
        self.params = [self.W, self.b]

    def connect(self, input):
        self.input = input
        self.output = T.tanh(T.dot(input, self.W) + self.b)

class DropoutLayer(object):
    def __init__(self, rng, p):
        self.rng = rng
        self.p = p

    def connect(self, input):
        self.input = input
        if self.p == 0.0:
            self.output = self.input
        else:
            self.bitmask = self.rng.binomial(n=1, p=1.0-self.p, size=self.input.shape)
            self.output = self.input * self.bitmask

class LogisticRegression(object):
    def __init__(self, n_in, n_out):
        self.W = theano.shared(
            value=numpy.zeros((n_in, n_out),
                              dtype=theano.config.floatX),
            name='W',
            borrow=True
        )
        self.b = theano.shared(
            value=numpy.zeros((n_out,),
                              dtype=theano.config.floatX),
            name='b',
            borrow=True
        )
        self.params = [self.W, self.b]

    def connect(self, input):
        self.input = input
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        assert y.ndim == self.y_pred.ndim
        assert y.dtype.startswith('int')
        return T.mean(T.neq(self.y_pred, y))

class MLP(object):
    def __init__(self, rng, mask_rng, input, n_in, n_hidden, n_out, dropout_prob=0.0):
        self.input = input
        self.dropout_prob = dropout_prob
        self.dropoutLayer = DropoutLayer(mask_rng, dropout_prob)
        self.hiddenLayer = HiddenLayer(rng, n_in, n_hidden)
        self.logRegressionLayer = LogisticRegression(n_hidden, n_out)
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

        self.L1 = abs(self.hiddenLayer.W).sum() + abs(self.logRegressionLayer.W).sum()
        self.L2 = (self.hiddenLayer.W**2).sum() + (self.logRegressionLayer.W**2).sum()

    def negative_log_likelihood(self, y):
        self.dropoutLayer.connect(self.input)
        self.hiddenLayer.connect(self.dropoutLayer.output)
        self.logRegressionLayer.connect(self.hiddenLayer.output)
        return self.logRegressionLayer.negative_log_likelihood(y)

    def errors(self, y):
        self.hiddenLayer.connect(self.input * (1.0 - self.dropout_prob))
        self.logRegressionLayer.connect(self.hiddenLayer.output)
        return self.logRegressionLayer.errors(y)

def sgd_optimization_mnist_mlp(
        learning_rate=0.01,
        dropout_prob=0.0,
        L1_coeff=0.0,
        L2_coeff=0.0001,
        n_hidden=500,
        n_epochs=1000,
        dataset='mnist.pkl.gz',
        batch_size=20,
        logdir=""):
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    index = T.lscalar()

    x = T.matrix('x')
    y = T.ivector('y')

    rng = numpy.random.RandomState(seed=12345)
    mask_rng = MRG_RandomStreams(seed=12345)
    classifier = MLP(rng, mask_rng, x, n_in=28 * 28, n_hidden=n_hidden, n_out=10, dropout_prob=dropout_prob)
    cost = classifier.negative_log_likelihood(y) + classifier.L1 * L1_coeff + classifier.L2 * L2_coeff

    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    grads = [T.grad(cost=cost, wrt=param) for param in classifier.params]
    updates = [(param, param - learning_rate * g) for g, param in zip(grads, classifier.params)]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    train_loss = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # training loop
    patience = 5000
    patience_increase = 2

    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience // 2)

    best_validation_loss = numpy.inf
    test_score = 0.0
    start_time = timeit.default_timer()

    # tensorboard visualization
    dir_path = os.path.dirname(os.path.realpath(__file__))
    tensorboard_writer = SummaryWriter(logdir)

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = [
                    validate_model(i) for i in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print('-----------------------------------------------------')
                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss

                    train_losses = [train_loss(i)
                                   for i in range(n_train_batches)]
                    train_score = numpy.mean(train_losses)

                    print(
                        'epoch %i, minibatch %i/%i, train error %f %%' %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            train_score * 100.
                        )
                    )

                    test_losses = [test_model(i)
                                   for i in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(
                        (
                            'epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

                    tensorboard_writer.add_summary(epoch, train_score=train_score, validation_score=this_validation_loss, test_score=test_score)

                    # with open('best_model.pkl', 'wb') as f:
                    #     pickle.dump(classifier, f)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print('The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time)))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)

class ConvPoolLayer(object):
    def __init__(self, rng, n_out, n_in, height, width, image_shape, maxpool_shape):
        self.filter_shape=(n_out, n_in, height, width)
        self.image_shape = image_shape
        self.maxpool_shape = maxpool_shape
        fan_in = n_in * height * width
        fan_out = n_out * height * width // numpy.prod(maxpool_shape)
        W_bounds = numpy.sqrt(6.0/(fan_in+fan_out))
        W_init = numpy.asarray(rng.uniform(-W_bounds, W_bounds, (n_out, n_in, height, width)), dtype=theano.config.floatX)
        self.W = theano.shared(value=W_init, name='W', borrow=True)
        self.b = theano.shared(value=numpy.zeros((n_out,), dtype=theano.config.floatX), name='b', borrow=True)
        self.params = [self.W, self.b]

    def connect(self, input):
        self.input = input
        conv_out = conv2d(input=input, filters=self.W, filter_shape=self.filter_shape, input_shape=self.image_shape)
        pool_out = pool.pool_2d(conv_out, self.maxpool_shape, ignore_border=True)
        self.output = T.tanh(pool_out + self.b.dimshuffle('x', 0, 'x', 'x'))

class LeNet(object):
    def __init__(self, input, batch_size, nkerns, rng, mask_rng, dropout_probs):
        self.input = input
        self.dropout_probs = dropout_probs
        self.l0_in = input.reshape((batch_size, 1, 28, 28))
        self.l0_dropout = DropoutLayer(mask_rng, dropout_probs[0])
        self.l0 = ConvPoolLayer(rng, nkerns[0], 1, 5, 5, (batch_size, 1, 28, 28), (2, 2))
        self.l1_dropout = DropoutLayer(mask_rng, dropout_probs[1])
        self.l1 = ConvPoolLayer(rng, nkerns[1], nkerns[0], 5, 5, (batch_size, nkerns[0], 12, 12), (2, 2))
        self.l2_dropout = DropoutLayer(mask_rng, dropout_probs[2])
        self.l2 = HiddenLayer(rng, nkerns[1]*4*4, 500)
        self.l3 = LogisticRegression(500, 10)

        self.params = self.l0.params + self.l1.params + self.l2.params + self.l3.params

    def negative_log_likelihood(self, y):
        self.l0_dropout.connect(self.l0_in)
        self.l0.connect(self.l0_dropout.output)
        self.l1_dropout.connect(self.l0.output)
        self.l1.connect(self.l1_dropout.output)
        self.l2_dropout.connect(self.l1.output.flatten(2))
        self.l2.connect(self.l2_dropout.output)
        self.l3.connect(self.l2.output)
        return self.l3.negative_log_likelihood(y)

    def errors(self, y):
        self.l0.connect(self.l0_in * (1.0 - self.dropout_probs[0]))
        self.l1.connect(self.l0.output * (1.0 - self.dropout_probs[1]))
        self.l2.connect(self.l1.output.flatten(2) * (1.0 - self.dropout_probs[2]))
        self.l3.connect(self.l2.output)
        return self.l3.errors(y)

def sgd_optimization_mnist_lenet(
        learning_rate=0.1,
        dropout_probs=[],
        n_epochs=200,
        dataset='mnist.pkl.gz',
        nkerns=[20, 50],
        batch_size=500,
        logdir=''):

    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    index = T.lscalar()

    rng = numpy.random.RandomState(seed=1234)
    mask_rng = MRG_RandomStreams(seed=12345)

    x = T.dmatrix('x')
    y = T.ivector('y')

    classifier = LeNet(x, batch_size, nkerns, rng, mask_rng, dropout_probs)
    cost = classifier.negative_log_likelihood(y)
    params = classifier.params

    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    grads = T.grad(cost, params)
    updates = [(param, param - learning_rate * g) for g, param in zip(grads, params)]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    train_loss = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # training loop
    patience = 10000
    patience_increase = 2

    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience // 2)

    best_validation_loss = numpy.inf
    test_score = 0.0
    start_time = timeit.default_timer()

    # tensorboard visualization
    dir_path = os.path.dirname(os.path.realpath(__file__))
    tensorboard_writer = SummaryWriter(logdir)

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = [
                    validate_model(i) for i in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print('-----------------------------------------------------')
                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss

                    train_losses = [train_loss(i)
                                   for i in range(n_train_batches)]
                    train_score = numpy.mean(train_losses)

                    print(
                        'epoch %i, minibatch %i/%i, train error %f %%' %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            train_score * 100.
                        )
                    )

                    test_losses = [test_model(i)
                                   for i in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(
                        (
                            'epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

                    tensorboard_writer.add_summary(epoch, train_score=train_score, validation_score=this_validation_loss, test_score=test_score)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print('The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time)))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)


if __name__ == "__main__":
    logdir = sys.argv[1]
    print("log to ", logdir)
    logdir = os.path.join(os.getcwd(), logdir)
    # sgd_optimization_mnist_mlp(dropout_prob=0.40, logdir=logdir)
    sgd_optimization_mnist_lenet(dropout_probs=[0.3,0.1,0.1], logdir=logdir)
