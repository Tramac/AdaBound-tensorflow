"""AdaBound for Keras."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.optimizers import Optimizer
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import math_ops


class AdaBound(Optimizer):
    """AdaBound optimizer.

    Default parameters follow those provided in the original paper.

    Arguments:
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        final_lr: float >= 0. final learning rate.
        gamma: float >= 0. Convergence speed of the bound functions.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        amsbound: boolean. Whether to use the AMSBound variant of this algorithm
            from the paper "Adaptive Gradient Methods with Dynamic Bound of Learning Rate".
    """

    def __init__(self,
                 lr=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 final_lr=0.1,
                 gamma=0.001,
                 epsilon=1e-8,
                 decay=0.,
                 amsbound=False,
                 **kwargs):
        super(AdaBound, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.final_lr = K.variable(final_lr, name='final_lr')
            self.gamma = K.variable(gamma, name='gamma')
            self.decay = K.variable(decay, name='decay')
            self.amsbound = K.variable(amsbound, name='amsbound')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = K.variable(epsilon)
        self.initial_decay = decay
        self.base_lr = lr

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [state_ops.assign_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (
                    1. / (1. + self.decay * math_ops.cast(self.iterations,
                                                          K.dtype(self.decay))))

        t = math_ops.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (
                K.sqrt(1. - math_ops.pow(self.beta_2, t)) /
                (1. - math_ops.pow(self.beta_1, t)))

        final_lr = self.final_lr * lr / self.base_lr
        lower_bound = final_lr * (1. - 1. / (self.gamma * t + 1))
        upper_bound = final_lr * (1. + 1. / (self.gamma * t))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsbound:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * math_ops.square(g)
            if self.amsbound:
                vhat_t = math_ops.maximum(vhat, v_t)
                p_t = p - m_t * K.clip(lr_t / (K.sqrt(vhat_t) + self.epsilon), lower_bound, upper_bound)
                self.updates.append(state_ops.assign(vhat, vhat_t))
            else:
                p_t = p - m_t * K.clip(lr_t / (K.sqrt(v_t) + self.epsilon), lower_bound, upper_bound)

            self.updates.append(state_ops.assign(m, m_t))
            self.updates.append(state_ops.assign(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(state_ops.assign(p, new_p))
        return self.updates

    def get_config(self):
        config = {
            'lr': float(K.get_value(self.lr)),
            'beta1': float(K.get_value(self.beta_1)),
            'beta2': float(K.get_value(self.beta_2)),
            'final_lr': float(K.get_value(self.final_lr)),
            'gamma': float(K.get_value(self.gamma)),
            'decay': float(K.get_value(self.decay)),
            'epsilon': self.epsilon,
            'amsbound': self.amsbound
        }
        base_config = super(AdaBound, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
