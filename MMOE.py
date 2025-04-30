from __future__ import print_function

import pickle
import json
import operator
from functools import reduce
from train.models.tf_util import build_optimizer, init_var_map, \
    get_field_index, get_field_num, split_mask, split_param, sum_multi_hot, \
    activate, get_params_count
import tensorflow as tf
try:
    import horovod.tensorflow as hvd
    distribute_tmp = True
except Exception:
    print("have no horovod package")
    distribute_tmp = False


class MMOE:

    def __init__(self, dataset_argv, embedding_size, expert_argv, tower_argv,
                 init_argv, ptmzr_argv, reg_argv, dense_flags,
                 regression_task=False,
                 trans_argv=None,
                 u_init_argv=None,
                 loss_mode='full',
                 merge_multi_hot=False, batch_norm=True):
        self.out_w = []
        self.out_b = []
        self.lbl_hldr_1 = None
        self.cross_w = []
        self.cross_b = []
        self.h_w = []
        self.h_b = []
        self.experts = None
        self.gates = []
        self.embed_v = None
        self.lbl_hldr_2 = None
        self.keep_prob = None
        self._lambda = None
        self.l1_lambda = None
        self.all_tower_deep_layers = None
        self.fields_num = None
        self.num_tasks = None
        self.num_experts = None
        self.num_multihot = None
        self.experts = None
        self.embedding_dim = None
        self.num_cross_layer = None
        self.tower_act_func = None
        self.features_size = None
        self.multi_hot_flags = None
        self.log = None
        self.num_expert_units = None
        self.use_bidden = config.USE_BIDDEN
        self.parameter_init(dataset_argv,
                            embedding_size,
                            reg_argv,
                            expert_argv,
                            tower_argv,
                            trans_argv,
                            merge_multi_hot)
        gate_act_func = 'softmax'

        self.init_input_layer(embedding_size,
                              init_argv, u_init_argv,
                              regression_task)

        dense_flag = reduce(operator.or_, dense_flags)
        self.init_placeholder(dense_flag)

        self.bottom_outputs = self.calculate_expert_gate(self.wt_hldr,
                                                         self.id_hldr,
                                                         self.embed_v,
                                                         merge_multi_hot,
                                                         self.experts,
                                                         batch_norm,
                                                         gate_act_func,
                                                         self.num_expert_units,
                                                         self.embedding_dim,
                                                         self.num_experts, training=True)

        x_stacks = self.tower_layer(self.bottom_outputs, self.dense_hldr if dense_flag else None,
                                    dense_flags, batch_norm, training=True)

        final_layer_y = self.final_layer(x_stacks, init_argv, batch_norm, training=True)

        self.train_preds(final_layer_y)
        self.loss_part(final_layer_y)

        self.eval_bottom_outputs = self.calculate_expert_gate(self.eval_wt_hldr,
                                                              self.eval_id_hldr,
                                                              self.embed_v,
                                                              merge_multi_hot,
                                                              self.experts,
                                                              batch_norm,
                                                              gate_act_func,
                                                              self.num_expert_units,
                                                              self.embedding_dim,
                                                              self.num_experts,
                                                              training=False)

        eval_x_stacks = self.tower_layer(self.eval_bottom_outputs,
                                         self.eval_dense_hldr if dense_flag else None,
                                         dense_flags, batch_norm,
                                         training=False)

        eval_final_layer_y = self.final_layer(eval_x_stacks, init_argv,
                                              batch_norm, training=False)

        self.eval_train_preds(eval_final_layer_y)

        self.update_optimizer(ptmzr_argv)

    def parameter_init(self, dataset_argv, embedding_size,
                       reg_argv,
                       expert_argv, tower_argv, trans_argv,
                       merge_multi_hot=False):
        (self.features_size,
         self.fields_num,
         self.dense_num,
         self.multi_hot_flags,
         multi_hot_len) = dataset_argv

        one_hot_flags = [not flag for flag in self.multi_hot_flags]

        # num_tower_units: dcn deep-layers []
        self.num_expert_units, self.num_experts, self.num_tasks, self.expert_act_func = expert_argv
        tower_deep_layers, self.num_cross_layer, self.tower_act_func = tower_argv
        self.keep_prob, self._lambda, self.l1_lambda = reg_argv

        num_onehot = int(sum(one_hot_flags))
        if multi_hot_len != 0:
            self.num_multihot = int(sum(self.multi_hot_flags) / multi_hot_len)
        else:
            print('zero')

        if merge_multi_hot:
            self.embedding_dim = int((self.num_multihot + num_onehot) * embedding_size)
        else:
            self.embedding_dim = int(self.fields_num * embedding_size)

        self.all_tower_deep_layers = [self.num_expert_units] + tower_deep_layers

        self.dense_deep_layers = [self.num_expert_units + self.dense_num] + tower_deep_layers



    def init_input_layer(self, embedding_size,
                         init_argv, u_init_argv,
                         regression_task=False):
        # init input layer
        init_acts = [('embed', [self.features_size, embedding_size], 'random'),
                     ('experts',
                      [self.embedding_dim, self.num_expert_units, self.num_experts],
                      'random'),
                     ('cross_w', [self.num_cross_layer, self.num_expert_units], 'random'),
                     ('cross_b', [self.num_cross_layer, self.num_expert_units], 'random')]

        # add gate layer
        for i in range(self.num_tasks):
            init_acts.append(('gate%d' % (i + 1),
                              [self.embedding_dim, self.num_experts], 'random'))

        # add classification tower layers
        for i in range(len(self.all_tower_deep_layers) - 1):
            init_acts.extend([('h%d_w' % (i + 1),
                               self.all_tower_deep_layers[i: i + 2], 'random'),
                              ('h%d_b' % (i + 1),
                               [self.all_tower_deep_layers[i + 1]], 'random')])
        # add regression tower layers
        if regression_task:
            for i in range(len(self.dense_deep_layers) - 1):
                init_acts.extend([('h%d_w_regression' % (i + 1),
                                   self.dense_deep_layers[i: i + 2], 'random'),
                                  ('h%d_b_regression' % (i + 1),
                                   [self.dense_deep_layers[i + 1]], 'random')])

        var_map, log = init_var_map(init_argv, init_acts)

        self.log = json.dumps(locals(), default=str, sort_keys=True, indent=4)
        self.log += log
        self.input_variable(var_map)
        self.expert_gate_variable(var_map)

    def input_variable(self, var_map):
        self.embed_v = tf.Variable(var_map['embed'], name="share_emb")

        for _ in range(self.num_tasks):
            self.cross_w.append(tf.Variable(var_map['cross_w']))
            self.cross_b.append(tf.Variable(var_map['cross_b']))

        for i in range(len(self.all_tower_deep_layers) - 1):
            if i == 0:
                for _ in range(self.num_tasks):
                    self.h_w.append([tf.Variable(var_map['h%d_w' % (i + 1)])])
                    self.h_b.append([tf.Variable(var_map['h%d_b' % (i + 1)])])
            else:
                for j in range(self.num_tasks):
                    self.h_w[j].append(tf.Variable(var_map['h%d_w' % (i + 1)]))
                    self.h_b[j].append(tf.Variable(var_map['h%d_b' % (i + 1)]))

    def expert_gate_variable(self, var_map):

        for i in range(self.num_tasks):
            self.gates.append(tf.Variable(var_map['gate%d' % (i + 1)]))
        self.experts = tf.Variable(var_map['experts'])

    def init_placeholder(self, dense_flag):
        self.wt_hldr = tf.placeholder(tf.float32, shape=[None, self.fields_num])
        self.id_hldr = tf.placeholder(tf.int64, shape=[None, self.fields_num])

        self.eval_wt_hldr = tf.placeholder(tf.float32, [None, self.fields_num],
                                           name='wt')
        self.eval_id_hldr = tf.placeholder(tf.int64, [None, self.fields_num],
                                           name='id')

        if dense_flag > 0:
            self.dense_hldr = tf.placeholder(tf.float32,
                                             shape=[None, self.dense_num])
            self.eval_dense_hldr = tf.placeholder(tf.float32,
                                                  [None, self.dense_num],
                                                  name='dense')

        self.global_step = tf.placeholder(tf.int64, name='global_step')
        self.label_hldl_placeholder()

    def label_hldl_placeholder(self):
        self.lbl_hldr_1 = tf.placeholder(tf.float32)
        self.lbl_hldr_2 = tf.placeholder(tf.float32)

    def calculate_expert_gate(self, wt_hldr, id_hldr,
                              embed_v, merge_multi_hot,
                              experts, batch_norm,
                              gate_act_func,
                              num_expert_units,
                              embedding_dim, num_experts,
                              training):
        # construct input embedding layer
        vx_embed = self.construct_embedding(wt_hldr, id_hldr,
                                            embed_v,
                                            merge_multi_hot)

        # first forward: input -> experts
        shared_expert_output = self.single_forward(vx_embed, experts,
                                                   self.expert_act_func,
                                                   embedding_dim,
                                                   num_expert_units,
                                                   num_experts,
                                                   dimension=3,
                                                   training=training,
                                                   batch_norm=batch_norm,
                                                   name_scope="expert")

        gates_outputs = []
        for index, gate in enumerate(self.gates):
            gate_output = self.single_forward(vx_embed, gate,
                                              gate_act_func,
                                              embedding_dim, num_expert_units,
                                              num_experts,
                                              dimension=2,
                                              training=training,
                                              batch_norm=batch_norm,
                                              name_scope="gate%d" % index)
            gates_outputs.append(gate_output)

        bottom_outputs = []
        for gate_output in gates_outputs:
            expanded_gate_output = tf.expand_dims(gate_output, axis=1)
            repeated_expanded_gate_output = self.repeat_elements(
                expanded_gate_output, num_expert_units, axis=1)
            gate_x_expert_output = tf.multiply(shared_expert_output,
                                               repeated_expanded_gate_output)
            gate_x_expert_output_sum = tf.reduce_sum(gate_x_expert_output,
                                                     axis=2)

            bottom_outputs.append(gate_x_expert_output_sum)
        return bottom_outputs

    def tower_layer(self, bottom_outputs, dense_hldr, dense_flags, batch_norm, training):
        x_stacks = []
        for i in range(self.num_tasks):
            x_l, final_hl = self.forward(bottom_outputs[i],
                                         self.cross_w[i],
                                         self.cross_b[i],
                                         self.h_w[i],
                                         self.h_b[i],
                                         dense_hldr,
                                         self.num_expert_units,
                                         self.tower_act_func,
                                         dense_flags[i],
                                         training=training,
                                         batch_norm=batch_norm,
                                         name_scope="tower%d" % (i + 1))
            x_stack = final_hl if dense_flags[i] else tf.concat([x_l, final_hl], 1)

            x_stacks.append(x_stack)

        return x_stacks

    def final_variable(self, x_stacks, init_argv):
        init_acts_final = [('out_b', [1], 'zero')]
        for i in range(self.num_tasks):
            init_acts_final.append(('out_w_%d' % i, [int(x_stacks[i].shape[1]), 1], 'random'))

        var_map, log = init_var_map(init_argv, init_acts_final)

        self.log += log

        for i in range(self.num_tasks):
            self.out_w.append(tf.Variable(var_map['out_w_%d' % i]))
            self.out_b.append(tf.Variable(var_map['out_b']))

    def final_layer(self, x_stacks, init_argv, batch_norm, training):
        if training:
            self.final_variable(x_stacks, init_argv)
        final_layer_y = []
        for i in range(self.num_tasks):
            y = self.final_forward(x_stacks[i],
                                   self.out_w[i],
                                   self.out_b[i],
                                   training=training,
                                   batch_norm=batch_norm,
                                   name_scope="task%d" % (i + 1))
            final_layer_y.append(y)

        return final_layer_y

    def train_preds(self, final_layer_y):
        self.train_preds_ctr = tf.sigmoid(final_layer_y[0], name='predicitons_ctr')
        self.train_preds_cvr = tf.sigmoid(final_layer_y[1], name='predicitons_cvr')

        self.train_preds_ctcvr = tf.multiply(self.train_preds_ctr,
                                             self.train_preds_cvr,
                                             name='ctcvr_predictions')

    def eval_train_preds(self, eval_final_layer_y):
        self.eval_preds_ctr = tf.sigmoid(eval_final_layer_y[0])
        self.eval_out_ctr = tf.identity(self.eval_preds_ctr, name='ctr_prediction_node')
        self.eval_preds_cvr = tf.sigmoid(eval_final_layer_y[1])
        self.eval_out_cvr = tf.identity(self.eval_preds_cvr, name='cvr_prediction_node')
        self.eval_preds_ctcvr = tf.multiply(self.eval_preds_ctr,
                                            self.eval_preds_cvr,
                                            name='predictionNode')

    def loss_part(self, final_layer_y):
        log_loss_ctr = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=final_layer_y[0],
                                                    labels=self.lbl_hldr_1),
            name='ctr_loss')

        log_loss_ctcvr = tf.reduce_mean(
            -self.lbl_hldr_2 * tf.log(self.train_preds_ctcvr) - (1 - self.lbl_hldr_2) * tf.log(
                1 - self.train_preds_ctcvr))

        self.loss = tf.add(log_loss_ctr, log_loss_ctcvr)

        self.loss += tf.contrib.layers.l1_regularizer(self.l1_lambda)(self.cross_w[0]) \
                     + tf.contrib.layers.l1_regularizer(self.l1_lambda)(self.cross_b[0]) \
                     + tf.contrib.layers.l1_regularizer(self.l1_lambda)(self.cross_w[1]) \
                     + tf.contrib.layers.l1_regularizer(self.l1_lambda)(self.cross_b[1]) \
                     + self._lambda * tf.nn.l2_loss(self.embed_v)

    def update_optimizer(self, ptmzr_argv):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            learning_rate = tf.train.exponential_decay(learning_rate=ptmzr_argv[1], global_step=self.global_step,
                                                       decay_rate=ptmzr_argv[3], decay_steps=ptmzr_argv[4],
                                                       staircase=False)
            if distribute_tmp:
                opt = tf.train.AdamOptimizer(learning_rate=learning_rate * hvd.size(), epsilon=ptmzr_argv[2])
                opt = hvd.DistributedOptimizer(opt, compression=hvd.Compression.fp16, sparse_as_dense=True)
            else:
                opt = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=ptmzr_argv[2])
            self.gradients = opt.compute_gradients(self.loss)
            for i, (g, v) in enumerate(self.gradients):
                if g is not None:
                    self.gradients[i] = (tf.clip_by_value(g, -1, 1), v)
            self.ptmzr = opt.apply_gradients(self.gradients)
            log = 'optimizer: %s, learning rate: %g, epsilon: %g\n' % (ptmzr_argv[0], ptmzr_argv[1], ptmzr_argv[2])
        self.log += log
        params_count = get_params_count()
        self.log += 'total count of trainable variables is: %d' % params_count

    # construct the embedding layer
    def construct_embedding(self, wt_hldr, id_hldr,
                            embed_v, merge_multi_hot=False,
                            is_reduce=True):
        mask = tf.expand_dims(wt_hldr, 2)
        if merge_multi_hot and self.num_multihot > 0:
            # *_hot_mask is weight(values that follow the ids in the dataset,
            # different from weight of param) that used
            one_hot_mask, multi_hot_mask = split_mask(
                mask, self.multi_hot_flags, self.num_multihot)

            one_hot_v, multi_hot_v = split_param(
                embed_v, id_hldr, self.multi_hot_flags)

            # fm part (reduce multi-hot vector's length to k*1)
            multi_hot_vx = sum_multi_hot(
                multi_hot_v, multi_hot_mask, self.num_multihot, is_reduce=is_reduce)
            one_hot_vx = tf.multiply(one_hot_v, one_hot_mask)
            vx_embed = tf.concat([one_hot_vx, multi_hot_vx], axis=1)
        else:
            vx_embed = tf.multiply(tf.gather(embed_v, id_hldr), mask)
        return vx_embed

    def single_forward(self, vx_embed, x_tensor,
                       act_func,
                       embedding_dim,
                       num_expert_units,
                       num_experts, dimension,
                       training,
                       batch_norm=False,
                       name_scope="expert"):
        hidden_output = tf.reshape(vx_embed, [-1, embedding_dim])
        print('shape of hidden_output in single-forward of %s is %s' % (
            name_scope, hidden_output.shape))
        print('shape of x_tensor in single-forward of %s is %s' % (
            name_scope, x_tensor.shape))

        hidden_output = tf.tensordot(hidden_output, x_tensor, axes=1)
        if dimension == 3:
            hidden_output.set_shape(
                (None, num_expert_units, num_experts))
        else:
            hidden_output.set_shape((None, num_experts))

        if batch_norm:
            hidden_output = tf.layers.batch_normalization(
                hidden_output, training=training, reuse=not training,
                name=(name_scope + "_single")
            )
        hidden_output = activate(act_func, hidden_output)
        if training:
            hidden_output = tf.nn.dropout(hidden_output, keep_prob=self.keep_prob)

        print('shape of hidden_output in batch_normalization of %s is %s' % (
            name_scope, hidden_output.shape))

        return hidden_output

    def forward(self, vx_embed,
                cross_w, cross_b, h_w, h_b,
                dense_feature,
                num_expert_units,
                act_func,
                dense_flag=False,
                training=True,
                batch_norm=False,
                name_scope="bn"):
        x_l = []
        if not dense_flag:
            # embedding layer
            x_0 = tf.reshape(vx_embed, [-1, num_expert_units])
            # cross layer
            x_l = x_0
            for i in range(self.num_cross_layer):
                xlw = tf.tensordot(x_l, cross_w[i], axes=1)
                x_l = x_0 * tf.expand_dims(xlw, -1) + cross_b[i] + x_l
                x_l.set_shape((None, num_expert_units))

        # get final hidden layer output
        final_hl = self.deep_forward(vx_embed, h_w, h_b,
                                     dense_feature, dense_flag,
                                     act_func,
                                     num_expert_units,
                                     training, batch_norm, name_scope)

        return x_l, final_hl

    def deep_forward(self, vx_embed, h_w, h_b, dense_feature, dense_flag, act_func,
                     num_expert_units, training,
                     batch_norm=False, name_scope="bn"):
        if dense_flag:
            hidden_output = tf.concat([dense_feature,
                                       tf.reshape(vx_embed, [-1,
                                                             num_expert_units])],
                                      1)
        else:
            hidden_output = tf.reshape(vx_embed, [-1, num_expert_units])

        for i in range(len(h_w)):
            hidden_output = tf.tensordot(hidden_output, h_w[i], axes=1) + h_b[i]
            hidden_output.set_shape((None, h_w[i].shape[1]))
            if batch_norm:
                print("setting bn for training stage")
                hidden_output = tf.layers.batch_normalization(hidden_output,
                                                              training=training,
                                                              reuse=not training,
                                                              name=(name_scope + "_%d") % i)
            if i < len(h_w) - 1:
                hidden_output = activate(act_func, hidden_output)
                if training:
                    hidden_output = tf.nn.dropout(hidden_output,
                                                  keep_prob=self.keep_prob)

        return hidden_output

    @staticmethod
    def final_forward(final_layer, out_w, out_b,
                      training, batch_norm=False,
                      name_scope="bn"):
        hidden_output = tf.matmul(final_layer, out_w) + out_b
        if batch_norm:
            hidden_output = tf.layers.batch_normalization(
                hidden_output, training=training, reuse=not training,
                name=(name_scope)
            )

        return tf.reshape(hidden_output, [-1])

    @staticmethod
    def repeat_elements(x, rep, axis):
        x_shape = x.get_shape().as_list()

        # slices along the repeat axis
        splits = tf.split(x, x_shape[axis], axis)

        # repeat each slice the given number of reps
        x_rep = [s for s in splits for i in range(rep)]

        return tf.concat(x_rep, axis)
