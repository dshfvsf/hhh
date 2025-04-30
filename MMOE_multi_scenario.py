from __future__ import print_function

import operator
from functools import reduce
from train.models.tf_util import init_var_map, get_params_count
import tensorflow as tf
import numpy as np

try:
    import horovod.tensorflow as hvd

    distribute_tmp = True
except Exception:
    print("have no horovod package")
    distribute_tmp = False
from models.MMOE_ALL import MMOE_ALL
from train.models.tf_util import get_domain_mask_noah
from train.layer.CoreLayer import DomainTower


class MMOE_multi_scenario(MMOE_ALL):

    def __init__(self, config, dataset_argv, embedding_size, expert_argv, tower_argv,
                 init_argv, ptmzr_argv, reg_argv, dense_flags,
                 trans_argv=None, u_init_argv=None,
                 multi_domain_argv=None,
                 loss_mode='full', merge_multi_hot=False,
                 batch_norm=True, distill=False,
                 checkpoint=None, sess=None, regression_task=False,
                 use_cross=True, use_linear=False, use_fm=False
                 ):
        self.cross_w = []
        self.cross_b = []
        self.out_w = []
        self.out_b = []
        self.h_w = []
        self.h_b = []
        self.embed_v = None
        self.lbl_hldr = None
        self.keep_prob = None
        self._lambda = None
        self.l1_lambda, self.l2_lambda = None, None
        self.all_tower_deep_layers = None
        self.fields_num = None

        # domain tower
        self.use_domain_tower = getattr(config, 'USE_DOMAIN_TOWER', False)
        self.domain_tower = None

        self.experts = None
        self.gates = []
        self.num_tasks = None
        self.has_task_mask = None
        self.num_experts = None

        self.num_multihot = None
        self.embedding_dim = None
        self.num_cross_layer = None
        self.tower_act_func = None
        self.features_size = None
        self.multi_hot_flags = None
        self.expert_hidden_layers = None
        self.gate_hidden_layers = None
        self.lbl_values = None
        self.lbl_masks = None
        self.cagrad_w = None
        self.log = None
        self.num_experts_task = None
        self.num_experts_shared = None
        self.expert_act_func = None

        self.use_trm = None

        self.data_member_part(config, distill, init_argv, dataset_argv, reg_argv,
                              checkpoint, sess, ptmzr_argv, merge_multi_hot,
                              regression_task, use_cross, use_linear, use_fm,
                              batch_norm, dense_flags, multi_domain_argv)
        self.parameter_init(config, dataset_argv, embedding_size,
                            reg_argv, expert_argv, tower_argv, trans_argv, merge_multi_hot)

        self.init_input_layer(embedding_size, init_argv, regression_task)

        self.init_placeholder(self.dense_flag)

        pred_dict, label_dict = self.domain_forward(training=True)
        self.train_preds = pred_dict
        self.loss = self.get_mmoe_all_domain_loss_sum(pred_dict, label_dict, self._lambda)

        self.eval_preds, _ = self.domain_forward(training=False)
        self.sigmoid_identity_eval_node()
        self.save_and_update_optimizer()

        print("MMOE_multi_scenario model init finish")

    def data_member_part(self, config, distill, init_argv, dataset_argv, reg_argv,
                         checkpoint, sess, ptmzr_argv, merge_multi_hot,
                         regression_task, use_cross, use_linear, use_fm,
                         batch_norm, dense_flags, multi_domain_argv):
        self.config = config
        self.distill = distill
        self.init_argv = init_argv
        (self.features_size,
         self.fields_num,
         self.dense_num,
         self.multi_hot_flags,
         self.multi_hot_len) = dataset_argv
        self.keep_prob, self.l2_lambda, self.l1_lambda = reg_argv
        self.checkpoint = checkpoint
        self.sess = sess
        self.use_cross = use_cross
        self.use_linear = use_linear
        self.use_fm = use_fm
        self.use_sfps = self.config.USE_SFPS
        self.ptmzr_argv = ptmzr_argv
        self.merge_multi_hot = merge_multi_hot
        self.regression_task = regression_task
        self.batch_norm = batch_norm
        self.dense_flags = dense_flags
        self.dense_flag = reduce(operator.or_, dense_flags)

        self.gate_act_func = 'softmax'
        self.domain_dict, self.domain_col_idx, self.domain_flags = multi_domain_argv

    def domain_forward(self, training=False):
        if training:
            id_hldr = self.id_hldr
            wt_hldr = self.wt_hldr
            domain_hldr = tf.gather(self.id_hldr, self.domain_col_idx, axis=1)
            if self.dense_num != 0:
                dense_hldr = self.dense_hldr
            if self.use_domain_tower:
                self.domain_tower = DomainTower(self.config, self.num_multihot, self.num_onehot, self.embedding_dim,
                                                self.init_argv, use_final=True)
        else:
            id_hldr = self.eval_id_hldr
            wt_hldr = self.eval_wt_hldr
            domain_hldr = tf.gather(self.eval_id_hldr, self.domain_col_idx, axis=1)
            if self.dense_num != 0:
                dense_hldr = self.eval_dense_hldr

        # construct input embedding layer
        vx_embed = self.construct_embedding(wt_hldr, id_hldr, self.embed_v, self.merge_multi_hot)
        nn_input = tf.reshape(vx_embed, [-1, self.embedding_dim])

        bottom_outputs = self.calculate_expert_gate(nn_input, self.batch_norm, self.gate_act_func,
                                                    self.num_experts, self.num_tasks, training=training)

        if self.use_domain_tower:
            domain_bias = self.domain_tower(vx_embed, training)

        x_stack_list, label_list = [], []
        domain_bias_list = []
        for i in range(self.num_tasks):
            domain_idx = i
            domain_mask = get_domain_mask_noah(self.domain_dict.get(domain_idx), domain_hldr)
            domain_label = tf.boolean_mask(self.lbl_hldr, domain_mask) if training else None

            tower_i_input = tf.boolean_mask(bottom_outputs[i], domain_mask)
            print('domain {} tower input: {}'.format(domain_idx, tower_i_input))

            x_l, final_hl = self.forward(tower_i_input,
                                         self.cross_w[i],
                                         self.cross_b[i],
                                         self.h_w[i],
                                         self.h_b[i],
                                         dense_hldr if self.dense_flag else None,
                                         self.expert_hidden_layers[-1],
                                         self.tower_act_func,
                                         self.dense_flags[i],
                                         training=training,
                                         batch_norm=self.batch_norm,
                                         name_scope="tower%d" % (i + 1))
            x_stack = final_hl if self.dense_flags[i] else tf.concat([x_l, final_hl], 1)
            x_stack_list.append(x_stack)
            label_list.append(domain_label)
            if self.use_domain_tower:
                domain_bias_list.append(tf.boolean_mask(domain_bias, domain_mask))

        final_layer_y = self.final_layer(x_stack_list, self.init_argv, batch_norm=False, training=training)

        predict_dict, label_dict = {}, {}
        for domain_idx in self.domain_dict:
            predict_dict[domain_idx] = final_layer_y[int(domain_idx)]
            label_dict[domain_idx] = label_list[int(domain_idx)]
            if self.use_domain_tower:
                predict_dict[domain_idx] += domain_bias_list[int(domain_idx)]

        return predict_dict, label_dict

    def get_mmoe_all_domain_loss_sum(self, pred_dict, label_dict, _lambda):
        loss = 0
        for idx, d_pred in pred_dict.items():
            d_label = label_dict.get(idx)
            with tf.variable_scope("d_{}_loss".format(idx)):
                all_sample_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_pred, labels=d_label)
                loss += tf.reduce_mean(all_sample_loss, name='loss')
        loss += _lambda * tf.nn.l2_loss(self.embed_v)
        return loss


    def label_hldl_placeholder(self):
        # single label
        self.lbl_hldr = tf.placeholder(tf.float32)

    def sigmoid_identity_eval_node(self):
        prefix_name = 'predictionNode'

        for idx in self.eval_preds:
            eval_preds = tf.nn.sigmoid(self.eval_preds.get(idx))
            self.eval_preds[idx] = tf.identity(eval_preds, name='{}_{}'.format(prefix_name, idx))

    def save_and_update_optimizer(self):
        # same as dcn
        var_list = []
        for g_v in tf.global_variables():
            var_list.append(g_v)
        self.saver = tf.train.Saver(var_list=var_list)
        if self.checkpoint:
            self.saver.restore(self.sess, self.checkpoint)
            self.extend_param(self.sess, self.features_dim)

        # optimizer with gradient clipped
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            learning_rate, opt = self.get_lr_and_opt(self.ptmzr_argv)

            vars_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            if self.weight_method == 'gradnorm':
                vars_list.remove(self.loss_weights.experimental_ref().deref())
            self.gradients = opt.compute_gradients(self.loss, var_list=vars_list)
            # clip gradient
            for i, (g, v) in enumerate(self.gradients):
                if g is not None:
                    self.gradients[i] = (tf.clip_by_value(g, -1, 1), v)
            self.ptmzr = opt.apply_gradients(self.gradients)
            log = 'optimizer: %s, learning rate: %g, epsilon: %g, decay_rate: %g, decay_steps: %g\n' % (
            self.ptmzr_argv[0],
            self.ptmzr_argv[1],
            self.ptmzr_argv[2],
            self.ptmzr_argv[3],
            self.ptmzr_argv[4])
        self.log += log
        params_count = get_params_count()
        self.log += 'total count of trainable variables is: %d' % params_count
