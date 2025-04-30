# encoding=utf-8
from __future__ import print_function
import tensorflow as tf
from train.models.tf_util import init_var_map
from models.STAR import STAR


class STAR_Diff_Feats(STAR):

    def __init__(self, config, dataset_argv, architect_argv, init_argv,
                 ptmzr_argv, reg_argv, distilled_argv,
                 loss_mode='full', merge_multi_hot=False,
                 batch_norm=True, distill=False,
                 checkpoint=None, sess=None, regression_task=False,
                 use_cross=True, use_linear=False, use_fm=False,
                 star_argv=None, auxiliary_network_layers=None,
                 diff_feats_argv=None,
                 ):
        # ------------------------------
        super(STAR_Diff_Feats, self).__init__(config, dataset_argv, architect_argv, init_argv,
                                              ptmzr_argv, reg_argv, distilled_argv,
                                              loss_mode=loss_mode, merge_multi_hot=merge_multi_hot,
                                              batch_norm=batch_norm, distill=distill,
                                              checkpoint=checkpoint, sess=sess, regression_task=regression_task,
                                              use_cross=use_cross, use_linear=use_linear, use_fm=use_fm,
                                              star_argv=star_argv, auxiliary_network_layers=auxiliary_network_layers,
                                              diff_feats_argv=diff_feats_argv)

    def get_all_deep_layer(self):
        super(STAR_Diff_Feats, self).get_all_deep_layer()
        self.all_deep_layer[0] = len(self.share_feat_idxs)

    def init_weights_domain_w_b(self):
        share_deep_layer = self.all_deep_layer  # 此时 all_deep_layer 第一层为共享特征长度
        for domain_idx in self.domain_dict:
            spec_domain_w, spec_domain_b = [], []
            gen_idx = self.domain_idx_2_diff_feat[domain_idx]
            with tf.variable_scope("d_{}_w_b".format(domain_idx), reuse=tf.AUTO_REUSE):
                for i in range(len(share_deep_layer) - 1):
                    spec_first_layer_dim = len(self.bias_feat_idxs_list[gen_idx]) - len(self.share_feat_idxs)
                    spec_deep_layer = [spec_first_layer_dim + self.dense_num] + self.deep_layers
                    if i == 0:
                        # share + spec
                        init_acts = [('d_h{}_w_share'.format(i + 1), share_deep_layer[i: i + 2], 'random'),
                                     ('d_h{}_b_share'.format(i + 1), [share_deep_layer[i + 1]], 'random'),
                                     ('d_{}_h{}_w_spec'.format(domain_idx, i + 1), spec_deep_layer[i: i + 2], 'random'),
                                     ('d_{}_h{}_b_spec'.format(domain_idx, i + 1), [spec_deep_layer[i + 1]], 'random')]
                        var_map, _log = init_var_map(self.init_argv, init_acts)
                        self.log += _log
                        share_w = tf.Variable(var_map['d_h{}_w_share'.format(i + 1)], 'ramdom',
                                              name='d_h{}_w_share'.format(i + 1))
                        share_b = tf.Variable(var_map['d_h{}_b_share'.format(i + 1)], 'ramdom',
                                              name='d_h{}_b_share'.format(i + 1))
                        spec_w = tf.Variable(var_map['d_{}_h{}_w_spec'.format(domain_idx, i + 1)],
                                             name='d_{}_h{}_w_spec'.format(domain_idx, i + 1))
                        spec_b = tf.Variable(var_map['d_{}_h{}_b_spec'.format(domain_idx, i + 1)],
                                             name='d_{}_h{}_b_spec'.format(domain_idx, i + 1))
                        _domain_w = [share_w, spec_w]
                        _domain_b = [share_b, spec_b]
                    else:
                        init_acts = [
                            ('d_{}_h{}_w_spec'.format(domain_idx, i + 1), spec_deep_layer[i: i + 2], 'random'),
                            ('d_{}_h{}_b_spec'.format(domain_idx, i + 1), [spec_deep_layer[i + 1]], 'random')]
                        var_map, _log = init_var_map(self.init_argv, init_acts)
                        self.log += _log
                        _domain_w = tf.Variable(var_map['d_{}_h{}_w_spec'.format(domain_idx, i + 1)],
                                                name='d_{}_h{}_w_spec'.format(domain_idx, i + 1))
                        _domain_b = tf.Variable(var_map['d_{}_h{}_b_spec'.format(domain_idx, i + 1)],
                                                name='d_{}_h{}_b_spec'.format(domain_idx, i + 1))
                    spec_domain_w.append(_domain_w)
                    spec_domain_b.append(_domain_b)
            self.domain_w[domain_idx] = spec_domain_w
            self.domain_b[domain_idx] = spec_domain_b

    def get_fcn_w_b(self, layer, domain_idx):
        if layer == 0:
            share_d_w, spec_d_w = self.domain_w.get(domain_idx)[layer]
            share_d_b, spec_d_b = self.domain_b.get(domain_idx)[layer]
            _w = tf.concat([tf.multiply(self.h_w[layer], share_d_w), spec_d_w], axis=0)
            _b = tf.add(tf.add(self.h_b[layer], share_d_b), spec_d_b)
        else:
            _w = tf.multiply(self.h_w[layer], self.domain_w.get(domain_idx)[layer])
            _b = tf.add(self.h_b[layer], self.domain_b.get(domain_idx)[layer])
        return _w, _b
