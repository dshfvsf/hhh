from __future__ import print_function
from train.models.tf_util import get_params_count
import tensorflow as tf

try:
    import horovod.tensorflow as hvd

    distribute_tmp = True
except Exception:
    print("have no horovod package")
    distribute_tmp = False
from models.MMOE_ALL import MMOE_ALL


class MMOE_ALL_PUB(MMOE_ALL):

    def __init__(self, config, dataset_argv, embedding_size, expert_argv, tower_argv,
                 init_argv, ptmzr_argv, reg_argv, dense_flags, regression_task=False,
                 trans_argv=None, u_init_argv=None, loss_mode='full', merge_multi_hot=False, batch_norm=True):
        self.share_emb_variables_before = None
        self.update_emb_placeholder = None
        self.update_loss_weights_op = None
        self.loss_weights = None
        self.loss_gradnorm = None
        self.task_losses = None
        self.loss = 0
        self.gradients = None
        self.ptmzr = None
        self.ptmzr_list = []
        self.update_emb_list = []
        self.loss_list = []
        super(MMOE_ALL_PUB, self).__init__(config, dataset_argv, embedding_size, expert_argv, tower_argv,
                                           init_argv, ptmzr_argv, reg_argv, dense_flags,
                                           regression_task=regression_task,
                                           trans_argv=trans_argv, u_init_argv=u_init_argv, loss_mode=loss_mode,
                                           merge_multi_hot=merge_multi_hot, batch_norm=batch_norm)
        # 获取共享emb
        self.get_share_params()
        # 共享emb assign
        self.update_emb_op = self.share_emb_variables_before.assign(self.update_emb_placeholder)

    # 获取共享的参数,获取emb
    def get_share_params(self):
        self.share_emb_variables_before = [variables for variables in tf.trainable_variables() if
                                           variables.name == "share_emb:0"][0]

        self.update_emb_placeholder = tf.placeholder(self.share_emb_variables_before.dtype,
                                                     shape=self.share_emb_variables_before.get_shape())

    def loss_part(self, final_layer_y, ptmzr_argv):
        self.loss_weight_variable()
        self.loss = 0
        for i, final_y in enumerate(final_layer_y):
            loss_i = tf.nn.sigmoid_cross_entropy_with_logits(logits=final_y, labels=self.lbl_values[:, i])  # [B]
            if self.has_task_mask:
                loss_i = tf.multiply(loss_i, self.lbl_masks[:, i])
            if self.mean_all_samples:
                loss_i = tf.reduce_mean(loss_i)
            else:
                num_samples = tf.reduce_sum(self.lbl_masks[:, i])
                num_samples = tf.maximum(num_samples, tf.constant(1.0))
                loss_i = tf.reduce_sum(loss_i) / num_samples
            regular_loss = tf.contrib.layers.l1_regularizer(self.l1_lambda)(self.cross_w[i]) \
                           + tf.contrib.layers.l1_regularizer(self.l1_lambda)(self.cross_b[i]) \
                           + self._lambda * tf.nn.l2_loss(self.embed_v)
            one_task_loss = loss_i + regular_loss
            self.loss_list.append(one_task_loss)
            self.loss += loss_i
        self.loss += tf.contrib.layers.l1_regularizer(self.l1_lambda)(self.cross_w[0]) \
                     + tf.contrib.layers.l1_regularizer(self.l1_lambda)(self.cross_b[0]) \
                     + tf.contrib.layers.l1_regularizer(self.l1_lambda)(self.cross_w[1]) \
                     + tf.contrib.layers.l1_regularizer(self.l1_lambda)(self.cross_b[1]) \
                     + self._lambda * tf.nn.l2_loss(self.embed_v)

    def update_optimizer(self, ptmzr_argv):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            learning_rate, opt = self.get_lr_and_opt(ptmzr_argv)
            vars_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            # 拆分各个任务的loss
            for _, loss in enumerate(self.loss_list):
                task_grads = opt.compute_gradients(loss)
                for i, (g, v) in enumerate(task_grads):
                    if g is not None:
                        task_grads[i] = (tf.clip_by_value(g, -1, 1), v)
                task_ptmzr = opt.apply_gradients(task_grads)
                self.ptmzr_list.append(task_ptmzr)
                share_emb_variables_after = [variables for variables in tf.trainable_variables() if
                                             variables.name == "share_emb:0"][0]
                self.update_emb_list.append(share_emb_variables_after)
            # 原逻辑
            self.gradients = opt.compute_gradients(self.loss, var_list=vars_list)
            for i, (g, v) in enumerate(self.gradients):
                if g is not None:
                    self.gradients[i] = (tf.clip_by_value(g, -1, 1), v)
            self.ptmzr = opt.apply_gradients(self.gradients)
            log = 'optimizer: %s, learning rate: %g, epsilon: %g\n' % (ptmzr_argv[0], ptmzr_argv[1], ptmzr_argv[2])
        self.log += log
        params_count = get_params_count()
        self.log += 'total count of trainable variables is: %d' % params_count
