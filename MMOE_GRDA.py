from __future__ import print_function

import pickle
import json
import operator
from functools import reduce
from train.models.tf_util import activate
import tensorflow as tf
try:
    import horovod.tensorflow as hvd
    distribute_tmp = True
except Exception:
    print("have no horovod package")
    distribute_tmp = False
from models.MMOE_ALL import MMOE_ALL
from models.MMOE_CrossScenario import MMOE_CrossScenario
from models.grda import GRDA
from train.models.tf_util import get_params_count


class MMOE_GRDA(MMOE_CrossScenario): 
    def __init__(self, config, dataset_argv, embedding_size, expert_argv, tower_argv,
                 init_argv, ptmzr_argv, reg_argv, dense_flags, regression_task=False,
                 trans_argv=None, u_init_argv=None, loss_mode='full', merge_multi_hot=False, batch_norm=True):
        self.ptmzr = None
        self.gradients = None
        super().__init__(config, dataset_argv, embedding_size, expert_argv, tower_argv,
                 init_argv, ptmzr_argv, reg_argv, dense_flags, regression_task,
                 trans_argv, u_init_argv, loss_mode, merge_multi_hot, batch_norm)
     

    def update_optimizer(self, ptmzr_argv):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            learning_rate, opt = self.get_lr_and_opt(ptmzr_argv)
            gate_opt = GRDA(learning_rate=learning_rate)

            var_list = []
            var_gate = []
            for g_v in tf.global_variables():
                if 'gate' in g_v.name:
                    var_gate.append(g_v)
                else:
                    var_list.append(g_v)


            self.gradients = opt.compute_gradients(self.loss, var_list=var_list)
            gate_gradients = gate_opt.compute_gradients(self.loss, var_list=var_gate)
            for i, (g, v) in enumerate(self.gradients):
                if g is not None:
                    self.gradients[i] = (tf.clip_by_value(g, -1, 1), v)
                    
            self.ptmzr = [opt.apply_gradients(self.gradients), gate_opt.apply_gradients(gate_gradients)]
            log = 'optimizer: %s, learning rate: %g, epsilon: %g\n' % (ptmzr_argv[0], ptmzr_argv[1], ptmzr_argv[2])
        self.log += log
        params_count = get_params_count()
        self.log += 'total count of trainable variables is: %d' % params_count