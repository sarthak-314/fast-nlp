import tensorflow as tf
import tensorflow_addons as tfa

import matplotlib.pyplot as plt
import wandb
import math

from ..core import WORKING_DIR, red, green, blue

class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
    'Applies warmup schedule on the given lr schedule function'
    def __init__(self, warmup_lr, lr_scheduler, warmup_steps, power=1.0):
        super().__init__()
        self.warmup_lr = warmup_lr
        self.lr_scheduler = lr_scheduler
        self.warmup_steps = warmup_steps
        self.power = power

    def __call__(self, step):
        with tf.name_scope('WarmUp') as name:
            global_step_float = tf.cast(step, tf.float32)
            warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
            warmup_percent_done = global_step_float / warmup_steps_float
            warmup_learning_rate = self.warmup_lr * tf.math.pow(warmup_percent_done, self.power)
            return tf.cond(
                global_step_float < warmup_steps_float,
                lambda: warmup_learning_rate,
                lambda: self.lr_scheduler(step - self.warmup_steps),
                name=name,
            )

    def get_config(self):
        return {
            'warmup_lr': self.warmup_lr,
            'lr_scheduler': self.lr_scheduler,
            'warmup_steps': self.warmup_steps,
            'power': self.power,
        }

class CosineDecayRestarts(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Cosine decay with restarts
    gamma, step_multiply, min_lr_ratio=1e-2,
    """

    def __init__(self, lr, first_decay_steps, step_gamma=2, gamma=1, min_lr_ratio=0):
        super().__init__()
        self.initial_learning_rate = lr
        self.first_decay_steps = first_decay_steps
        self._t_mul = step_gamma
        self._m_mul = gamma
        self.alpha = min_lr_ratio

    def __call__(self, step):
        with tf.name_scope('SGDRDecay') as name:
            initial_learning_rate = tf.convert_to_tensor(
                self.initial_learning_rate, name="initial_learning_rate")
            dtype = initial_learning_rate.dtype
            first_decay_steps = tf.cast(self.first_decay_steps, dtype)
            alpha = tf.cast(self.alpha, dtype)
            t_mul = tf.cast(self._t_mul, dtype)
            m_mul = tf.cast(self._m_mul, dtype)

            global_step_recomp = tf.cast(step, dtype)
            completed_fraction = global_step_recomp / first_decay_steps

        def compute_step(completed_fraction, geometric=False):
            """Helper for `cond` operation."""
            if geometric:
                i_restart = tf.floor(
                    tf.math.log(1.0 - completed_fraction * (1.0 - t_mul)) /
                    tf.math.log(t_mul))

                sum_r = (1.0 - t_mul**i_restart) / (1.0 - t_mul)
                completed_fraction = (completed_fraction - sum_r) / t_mul**i_restart

            else:
                i_restart = tf.floor(completed_fraction)
                completed_fraction -= i_restart
            return i_restart, completed_fraction

        i_restart, completed_fraction = tf.cond(
            tf.equal(t_mul, 1.0),
            lambda: compute_step(completed_fraction, geometric=False),
            lambda: compute_step(completed_fraction, geometric=True))

        m_fac = m_mul**i_restart
        cosine_decayed = 0.5 * m_fac * (1.0 + tf.cos(
            tf.constant(math.pi) * completed_fraction))
        decayed = (1 - alpha) * cosine_decayed + alpha
        return tf.multiply(initial_learning_rate, decayed, name=name)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "first_decay_steps": self.first_decay_steps,
            "t_mul": self._t_mul,
            "m_mul": self._m_mul,
            "alpha": self.alpha,
            "name": 'CosineLRDecay', 
        }

def plot_first_epoch(lr_scheduler, train_steps, checkpoints_per_epoch):
    plt.rcParams['figure.figsize'] = (20,3) # TODO: Move globally
    steps = list(range(0, train_steps, train_steps//checkpoints_per_epoch+1))
    _ = plt.plot([lr_scheduler(x) for x in range(train_steps)], markevery=steps, marker='o')


def adamw_optimizer_factory(HP, lr_scheduler): 
    optimizer = tfa.optimizers.AdamW(
        beta_1=HP.beta_1, 
        beta_2=HP.beta_2, 
        epsilon=HP.epsilon, 
        weight_decay=HP.max_weight_decay, 
        clipnorm=HP.max_grad_norm,
        learning_rate=lr_scheduler,
    )
    if HP.average_decay > 0: 
        print(f'Using EMA with decay {blue(HP.average_decay)}')
        optimizer = tfa.optimizers.MovingAverage(
            optimizer, 
            average_decay=HP.average_decay, 
            dynamic_decay=True, 
        )
    else: 
        print('Skipping EMA')
    return optimizer

def lr_scheduler_factory(HP, train_steps):
    lr_scheduler = CosineDecayRestarts(
        HP.max_lr, 
        int(HP.decay_epochs*train_steps)+1, 
        HP.step_gamma, 
        HP.lr_gamma, 
        HP.min_lr/HP.max_lr, 
    )
    lr_scheduler = WarmUp(
        warmup_lr=HP.max_lr, 
        lr_scheduler=lr_scheduler, 
        warmup_steps=int(train_steps*HP.warmup_epochs)-1, 
        power=HP.warmup_power, 
    )
    return lr_scheduler



