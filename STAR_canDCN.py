# encoding=utf-8
from __future__ import print_function
import tensorflow as tf

from models.multi_scenario_base import MultiScenarioBase
from models.STAR import STAR
from models.canDCN import canDCN
from train.models.tf_util import (build_optimizer, init_var_map, get_field_index, get_field_num,
                                  split_mask, split_param, split_param_4d, sum_pooling_multi_hot, sum_pooling,
                                  split_embedding, activate, get_domain_mask_noah)
from train.layer.CoreLayer import DomainTower
from common_util.util import get_other_domain_key


class STAR_canDCN(STAR, canDCN, MultiScenarioBase):

    def __init__(self, config, dataset_argv, architect_argv, init_argv,
                 ptmzr_argv, reg_argv, distilled_argv,
                 loss_mode='full', merge_multi_hot=False,
                 batch_norm=True, distill=False,
                 checkpoint=None, sess=None, regression_task=False,
                 use_cross=True, use_linear=False, use_fm=False,
                 star_argv=None, auxiliary_network_layers=None,
                 diff_feats_argv=None,
                 use_can=True, one_hot_idx_list=None, multi_hot_idx_list=None,
                 can_as_mlp_input=False, can_as_can_module=True, dense_as_mlp_input=True, dense_as_module=False,
                 ):
        super(MultiScenarioBase, self).__init__(self)
        self.is_diff_features, self.share_feat_idxs, \
            self.bias_feat_idxs_list, self.domain_idx_2_diff_feat = diff_feats_argv
        # ---------
        self.parsing_parameters(config, distilled_argv, distill, init_argv, dataset_argv,
                                architect_argv, reg_argv, checkpoint, sess, ptmzr_argv,
                                merge_multi_hot, regression_task,
                                use_cross, use_linear,
                                one_hot_idx_list, multi_hot_idx_list,
                                use_fm, use_can,
                                can_as_mlp_input, can_as_can_module,
                                dense_as_mlp_input, dense_as_module,
                                batch_norm)
        self.use_sfps = config.USE_SFPS
        self.keep_prob, self._lambda, self.l1_lambda = reg_argv

        # for star
        self.h_w, self.h_b = [], []
        self.domain_w, self.domain_b = {}, {}
        self.h_ax_w, self.h_ax_b = [], []
        self.final_var_map = None
        self.auxiliary_network_layers = auxiliary_network_layers
        self.domain_dict, self.domain_col_idx, self.domain_flags = star_argv
        self.use_bn = batch_norm

        self.other_domain_key = get_other_domain_key(config)  # 分配没有进行分组的list_id
        self.other_domain_list_id = config.other_domain_list_id
        # 默认没有在domain_dict中提到的list_id也参与训练
        self.is_train_with_other_domain = getattr(config, 'is_train_with_other_domain', True)

        # for domain tower
        self.use_domain_tower = getattr(config, 'USE_DOMAIN_TOWER', False)
        self.domain_tower = None

        # for candcn
        self.dnn_input_dim = None
        self.out_w_ctr, self.out_b_ctr = [], []
        self.out_w_price, self.out_b_price = [], []
        self.out_w_p, self.out_b_p = None, None
        self.final_output_p = None
        self.out_w_pos, self.out_b_pos = [], []
        self.out_w_pos2, self.out_b_pos2 = None, None
        self.final_output_pos = None
        self.final_input_dim = None
        # OOM may occur when is_split_can4star==True. Not fixed yet
        self.is_split_can4star = getattr(self.config, 'is_split_can4star', False)
        self.cross_input_dim = None
        self.cross_w, self.cross_b = None, None

        self.set_position_and_filter_config()
        self.sfps_init_func()
        self.init_placeholder_noah()

        if self.fields_num != 0:
            self.compute_embedding_dim()
            self.sparse_variable_part()

        self.get_domain_indicator_embedding_dim()
        self.get_auxiliary_network_layers()

        self.dnn_variable_part()
        self.init_weights_star()
        if self.use_can:
            self.can_variable_init4star()
        if self.use_domain_tower:
            self.domain_tower = DomainTower(config, self.num_multihot, self.num_onehot, self.embedding_dim, init_argv,
                                            use_final=True)

        pred_dict, label_dict = self.forward(is_training=True)
        self.train_preds = pred_dict
        self.loss = self.get_star_all_domain_loss_sum(pred_dict, label_dict, self._lambda)

        self.eval_preds, _ = self.forward(is_training=False)
        self.sigmoid_identity_eval_node()
        self.save_and_optimizer()

        print("STAR_canDCN model init finish")

    def forward(self, is_training=False):
        if is_training:
            id_hldr = self.id_hldr
            wt_hldr = self.wt_hldr
            domain_hldr = self.domain_hldr
            if self.dense_num != 0:
                dense_hldr = self.dense_hldr
            else:
                dense_hldr = None
        else:
            id_hldr = self.eval_id_hldr
            wt_hldr = self.eval_wt_hldr
            domain_hldr = self.eval_domain_hldr
            self.domain_col_idx = self.config.domain_col_idx
            self.eval_domain_hldr = tf.gather(self.eval_id_hldr, self.domain_col_idx, axis=1)
            if self.dense_num != 0:
                dense_hldr = self.eval_dense_hldr
            else:
                dense_hldr = None

        if self.fields_num != 0:
            vx_embed = self.get_embedding_component(id_hldr, wt_hldr, training=is_training)

        if not self.is_split_can4star:
            candcn_output = self.candcn_forward(candcn_input={'id_hldr': id_hldr,
                                                              'wt_hldr': wt_hldr,
                                                              'dense_hldr': dense_hldr,
                                                              'vx_embed': vx_embed},
                                                keyword='candcn_part',
                                                training=is_training)

        predict_dict, label_dict = {}, {}
        for idx in self.domain_dict:
            domain_mask = get_domain_mask_noah(self.domain_dict.get(idx), domain_hldr)
            if (self.is_train_with_other_domain or not is_training) and idx == self.other_domain_key:
                # 未分类的list_id也要计算
                all_domain_mask = get_domain_mask_noah(self.all_domain_list_id, domain_hldr)
                domain_mask = tf.math.logical_or(domain_mask, tf.math.logical_not(all_domain_mask))

            domain_embed = tf.boolean_mask(vx_embed, domain_mask)
            domain_label = tf.boolean_mask(self.lbl_hldr, domain_mask) if is_training else None

            domain_input = {'id_hldr': tf.boolean_mask(id_hldr, domain_mask),
                            'wt_hldr': tf.boolean_mask(wt_hldr, domain_mask),
                            'dense_hldr': tf.boolean_mask(dense_hldr, domain_mask) if self.dense_num != 0 else None,
                            'vx_embed': domain_embed}

            if self.is_split_can4star:
                candcn_domain_output = self.candcn_forward(candcn_input=domain_input,
                                                           domain_idx=idx,
                                                           keyword='d_{}_candcn_part'.format(idx),
                                                           training=is_training)
            else:
                candcn_domain_output = tf.boolean_mask(candcn_output, domain_mask)

            domain_sm = self.domain_star_candcn(domain_embed, candcn_domain_output, idx, training=is_training)
            pn_domain_id_embed = self.get_pn_domain_id_embed(domain_embed)
            domain_sa = self.auxiliary_network(pn_domain_id_embed, training=is_training)
            domain_predict = tf.add(domain_sm, domain_sa)
            if self.use_domain_tower:
                domain_bias = self.domain_tower(domain_embed, is_training)
                domain_predict = tf.add(domain_predict, domain_bias)

            predict_dict[idx] = domain_predict
            label_dict[idx] = domain_label

        return predict_dict, label_dict

    def domain_star_candcn(self, hidden_output, candcn_domain_output, domain_idx, training=False):
        with tf.variable_scope("d_{}_STAR_CanDCN".format(domain_idx), reuse=tf.AUTO_REUSE):
            for layer in range(len(self.domain_w.get(domain_idx))):
                _w, _b = self.get_fcn_w_b(layer, domain_idx)
                hidden_output = self.split_embed(hidden_output, layer, domain_idx)
                if self.use_bn:
                    _bn_w = tf.layers.batch_normalization(_w, training=training, reuse=not training,
                                                          name="d_{}_w_bn_{}".format(domain_idx, layer))
                    hidden_output = tf.matmul(hidden_output, _bn_w) + _b
                else:
                    hidden_output = tf.matmul(hidden_output, _w) + _b
                if self.use_bn:
                    hidden_output = tf.layers.batch_normalization(hidden_output, training=training, reuse=not training,
                                                                  name="d_{}_star_bn_{}".format(domain_idx, layer))
                hidden_output = activate(self.act_func, hidden_output)
                if training:
                    hidden_output = tf.nn.dropout(hidden_output, keep_prob=self.keep_prob)
            hidden_output = tf.concat((hidden_output, candcn_domain_output), axis=1)  # deep part concat with candcn
        hidden_output = self.final_forward(hidden_output, "d_{}_CanDCN_final".format(domain_idx), training=training)
        return hidden_output

    def candcn_forward(self, candcn_input, domain_idx=None, keyword='candcn_part', training=False):
        if domain_idx is None:
            print('candcn_forward for all domains')
        else:
            print('candcn_forward for domain {}'.format(domain_idx))
        with tf.variable_scope(keyword, reuse=tf.AUTO_REUSE):
            cross_input_list = []
            dnn_input_list = []
            final_input_list = []
            id_hldr, wt_hldr, dense_hldr = candcn_input['id_hldr'], candcn_input['wt_hldr'], candcn_input['dense_hldr']
            vx_embed = candcn_input['vx_embed']

            dnn_input_list.append(candcn_input)
            if self.use_cross:
                cross_input_list.append(vx_embed)

            if self.use_can:
                can_embed_output_list = self.construct_can_embedding(id_hldr=id_hldr,
                                                                     domain_idx=domain_idx)
                if self.can_as_mlp_input:
                    dnn_input_list.append(can_embed_output_list)
                    if self.use_cross:
                        cross_input_list.append(can_embed_output_list)
                elif self.can_as_can_module:
                    final_input_list.append(can_embed_output_list)

            if self.dense_num != 0 and self.dense_as_mlp_input:
                dnn_input_list.append(dense_hldr)

            # cross module
            if len(cross_input_list) != 0:
                cross_input = tf.concat(cross_input_list, axis=1)
                if training and (self.cross_input_dim is None):
                    self.cross_variable_init(cross_input)
                cross_output = self.cross_forward(cross_input, domain_idx)
                final_input_list.append(cross_output)

            # candcn final module
            candcn_output = tf.concat(final_input_list, axis=1)
        return candcn_output

    def construct_can_embedding(self, id_hldr, domain_idx=None):
        can_input_one_hot_v, can_input_multi_hot_v = split_param(
            self.can_input_embed.get(domain_idx) if domain_idx is not None else self.can_input_embed,
            id_hldr,
            self.multi_hot_flags)
        can_mlp_one_hot_v_111, can_mlp_multi_hot_v_111 = split_param_4d(
            self.can_mlp_embed_111.get(domain_idx) if domain_idx is not None else self.can_mlp_embed_111,
            id_hldr,
            self.multi_hot_flags)
        can_mlp_one_hot_v_222, can_mlp_multi_hot_v_222 = split_param_4d(
            self.can_mlp_embed_222.get(domain_idx) if domain_idx is not None else self.can_mlp_embed_222,
            id_hldr,
            self.multi_hot_flags)
        out_seq = []

        h = tf.einsum('aik,ajkl->aijl', can_input_multi_hot_v, can_mlp_one_hot_v_111)
        h = tf.nn.tanh(h)
        out_seq.append(h)
        h = tf.einsum('aijk,ajkl->aijl', h, can_mlp_one_hot_v_222)
        out_seq.append(h)
        out_seq = tf.concat(out_seq, 3)

        item_item = tf.reshape(out_seq, [-1, sum(self.can_sizes[1]) * sum(self.multi_hot_flags) * self.num_onehot])

        return item_item

    def final_forward(self, hidden_output, key_word, training):
        if training:
            self.final_variable_init(hidden_output, key_word)
        with tf.variable_scope(key_word, reuse=tf.AUTO_REUSE):
            out_w = tf.get_variable('out_w', initializer=lambda: self.final_var_map.get('{}_out_w'.format(key_word)))
            out_b = tf.get_variable('out_b', initializer=lambda: self.final_var_map.get('{}_out_b'.format(key_word)))
            hidden_output = tf.matmul(hidden_output, out_w) + out_b
            output_y = tf.reshape(hidden_output, [-1])
        return output_y

    def can_variable_init4star(self, keywords=None):
        # 只用一套can embedding
        if self.is_split_can4star:
            self.can_input_embed, self.can_mlp_embed_111, self.can_mlp_embed_222 = {}, {}, {}
            print('split candcn for star')
            for idx in self.domain_dict:
                init_acts = [('d_{}_can_mlp_embed_111'.format(idx), [self.features_dim] + self.can_sizes[0], 'random'),
                             ('d_{}_can_mlp_embed_222'.format(idx), [self.features_dim] + self.can_sizes[1], 'random'),
                             ('d_{}_can_input_embed'.format(idx), [self.features_dim] + self.can_sizes[2], 'random')]
                var_map, log = init_var_map(self.init_argv, init_acts)

                self.log += log

                self.can_input_embed[idx] = tf.Variable(var_map['d_{}_can_input_embed'.format(idx)])
                self.can_mlp_embed_111[idx] = tf.Variable(var_map['d_{}_can_mlp_embed_111'.format(idx)])
                self.can_mlp_embed_222[idx] = tf.Variable(var_map['d_{}_can_mlp_embed_222'.format(idx)])

        else:  # not split candcn for different domains
            init_acts = [('can_mlp_embed_111', [self.features_dim] + self.can_sizes[0], 'random'),
                         ('can_mlp_embed_222', [self.features_dim] + self.can_sizes[1], 'random'),
                         ('can_input_embed', [self.features_dim] + self.can_sizes[2], 'random')]

            var_map, log = init_var_map(self.init_argv, init_acts)

            self.log += log

            self.can_input_embed = tf.Variable(var_map['can_input_embed'])
            self.can_mlp_embed_111 = tf.Variable(var_map['can_mlp_embed_111'])
            self.can_mlp_embed_222 = tf.Variable(var_map['can_mlp_embed_222'])

        print("can_input_embed: {}".format(self.can_input_embed))
        print("can_mlp_embed_111: {}".format(self.can_mlp_embed_111))
        print("can_mlp_embed_222: {}".format(self.can_mlp_embed_222))

    def cross_variable_init(self, cross_input):
        self.cross_input_dim = cross_input.shape.as_list()[1]
        if self.is_split_can4star:
            self.cross_w, self.cross_b = {}, {}
            for domain_idx in self.domain_dict:
                with tf.variable_scope("d_{}_cross_w_b".format(domain_idx), reuse=tf.AUTO_REUSE):
                    init_acts = [('d_{}_cross_w'.format(domain_idx),
                                  [self.num_cross_layer, self.cross_input_dim], 'random'),
                                 ('d_{}_cross_b'.format(domain_idx),
                                  [self.num_cross_layer, self.cross_input_dim], 'random')]
                    var_map, _log = init_var_map(self.init_argv, init_acts)
                    self.log += _log
                    self.cross_w[domain_idx] = tf.Variable(var_map['d_{}_cross_w'.format(domain_idx)],
                                                           name='d_{}_cross_w'.format(domain_idx))
                    self.cross_b[domain_idx] = tf.Variable(var_map['d_{}_cross_b'.format(domain_idx)],
                                                           name='d_{}_cross_b'.format(domain_idx))
        else:
            init_acts = [('cross_w', [self.num_cross_layer, self.cross_input_dim], 'random'),
                         ('cross_b', [self.num_cross_layer, self.cross_input_dim], 'random')]

            var_map, log = init_var_map(self.init_argv, init_acts)

            self.log += log

            self.cross_w = tf.Variable(var_map['cross_w'])
            self.cross_b = tf.Variable(var_map['cross_b'])

        print('cross_w: {}'.format(self.cross_w))
        print('cross_b: {}'.format(self.cross_b))

    def cross_forward(self, cross_input, domain_idx=None):
        # embedding layer
        x_0 = cross_input
        # cross layer
        x_l = x_0
        if domain_idx is not None:
            print('cross_forward for domain {}'.format(domain_idx))
        if domain_idx is not None:
            cross_b = self.cross_b[domain_idx]
            cross_w = self.cross_w[domain_idx]
        else:
            cross_b = self.cross_b
            cross_w = self.cross_w
        for i in range(self.num_cross_layer):
            xlw = tf.tensordot(x_l, cross_w.get(i), axes=1)
            x_l = x_0 * tf.expand_dims(xlw, -1) + cross_b.get(i) + x_l
            x_l.set_shape((None, self.cross_input_dim))
        print('cross component init: input shape: %s, output shape: %s' % (
            cross_input.shape.as_list(), x_l.shape.as_list()))
        return x_l

    def final_variable_init(self, final_input, key_word):
        if self.final_input_dim is None:
            self.final_input_dim = final_input.shape.as_list()[1]
        init_acts_final = [
            ('{}_out_w'.format(key_word), [self.final_input_dim, 1], 'random'),
            ('{}_out_b'.format(key_word), [1], 'zero'),
        ]

        final_var_map_tmp, log = init_var_map(self.init_argv, init_acts_final)
        self.final_var_map.update(final_var_map_tmp)
        self.log += log

    def init_weights_star(self):
        # sub domain
        self.init_weights_domain_w_b()
        # auxiliary network
        self.init_weights_domain_auxiliary_net()
        # final
        self.final_auxiliary_map_init()

    def final_auxiliary_map_init(self):
        init_acts_final = []
        init_acts_final.extend([
            ('auxiliary_final_out_w', [int(self.auxiliary_network_layers[-1]), 1], 'random'),
            ('auxiliary_final_out_b', [1], 'zero')
        ])
        self.final_var_map, log1 = init_var_map(self.init_argv, init_acts_final)

