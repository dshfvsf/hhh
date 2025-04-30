# encoding=utf-8
from __future__ import print_function
import tensorflow as tf

from common_util.util import get_other_domain_key
from train.models.tf_util import init_var_map, activate, get_domain_mask_light
from models.multi_scenario_base import MultiScenarioBase
from models.DCN_SFPS import DCN_SFPS
from train.layer.CoreLayer import DomainTower


class STAR(MultiScenarioBase, object):

    def __init__(self, config, dataset_argv, architect_argv, init_argv,
                 ptmzr_argv, reg_argv, distilled_argv,
                 loss_mode='full', merge_multi_hot=False,
                 batch_norm=True, distill=False,
                 checkpoint=None, sess=None, regression_task=False,
                 use_cross=True, use_linear=False, use_fm=False,
                 star_argv=None, auxiliary_network_layers=None,
                 diff_feats_argv=None, model_name='STAR'
                 ):
        super(STAR, self).__init__()
        self.model_name = model_name
        self.is_diff_features, self.share_feat_idxs, \
            self.bias_feat_idxs_list, self.domain_idx_2_diff_feat = diff_feats_argv
        # ---------
        self.data_member_part(config, distilled_argv, distill, init_argv, dataset_argv, architect_argv, reg_argv,
                              checkpoint, sess, ptmzr_argv, merge_multi_hot, regression_task, use_cross, use_linear,
                              use_fm, batch_norm)
        self.h_w, self.h_b = [], []
        self.domain_w, self.domain_b = {}, {}
        self.h_ax_w, self.h_ax_b = [], []
        self.final_var_map = None
        self.auxiliary_network_layers = auxiliary_network_layers
        self.domain_dict, self.domain_col_idx, self.domain_flags = star_argv
        self.use_bn = True

        self.other_domain_key = get_other_domain_key(config)  # 分配没有进行分组的list_id
        self.other_domain_list_id = config.other_domain_list_id
        # 默认没有在domain_dict中提到的list_id也参与训练
        self.is_train_with_other_domain = getattr(config, 'is_train_with_other_domain', True)
        if self.is_train_with_other_domain:
            print("WARNING! List_id which not in the domain_org_dict will also be used in training stage!")

        # domain bias tower
        self.use_domain_tower = getattr(self.config, 'USE_DOMAIN_TOWER', False)
        self.domain_tower = None

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

        if self.use_domain_tower:
            self.domain_tower = DomainTower(config, self.num_multihot, self.num_onehot, self.embedding_dim, init_argv,
                                            use_final=True)

        pred_dict, label_dict = self.multi_domain_forward(self.wt_hldr, self.id_hldr, self.domain_hldr,
                                                          is_training=True)
        self.train_preds = pred_dict
        self.loss = self.get_star_all_domain_loss_sum(pred_dict, label_dict, self._lambda)

        if not self.use_sfps:
            self.eval_part()
            self.save_and_optimizer()
        else:
            self.optimizer()
            self.eval_part()
            self.sfps_save_part()
            self.saver_func()

        print("{} model init finish".format(self.model_name))

    def eval_part(self):
        if not self.config.split_sub_network:
            print('"config.split_sub_network" is False in Evaluation.')
            self.eval_preds, _ = self.multi_domain_forward(self.eval_wt_hldr, self.eval_id_hldr, self.eval_domain_hldr,
                                                           is_training=False)
        else:
            self.eval_preds, _ = self.multi_domain_forward_split(self.eval_wt_hldrs, self.eval_id_hldrs,
                                                                 is_training=False)
        self.sigmoid_identity_eval_node()

    def sfps_save_part(self):
        _sfps_save_preds, _ = self.multi_domain_forward_split(self.sfps_wt_hldrs, self.sfps_emb_hldrs,
                                                              is_training=False, is_save=True)
        self.sfps_save_preds = {}
        for idx, d_sfps_save_pred in _sfps_save_preds.items():
            self.sfps_save_preds[idx] = tf.sigmoid(d_sfps_save_pred, name='predictionNode_{}'.format(idx))

    def get_star_all_domain_loss_sum(self, pred_dict, label_dict, _lambda):
        loss = 0
        for idx, d_pred in pred_dict.items():
            d_label = label_dict.get(idx)
            with tf.variable_scope("d_{}_loss".format(idx)):
                all_sample_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_pred, labels=d_label)
                loss += tf.reduce_mean(all_sample_loss, name='loss')
        loss += _lambda * tf.nn.l2_loss(self.embed_v)
        return loss

    def multi_domain_forward(self, wt_hldr, id_hldr, domain_hldr, is_training=False):
        predict_dict, label_dict = {}, {}
        all_ids, all_domain_idxs = [], []
        vx_embed = self.construct_embedding(wt_hldr, id_hldr, self.merge_multi_hot, train=is_training)
        vx_embed = tf.reshape(vx_embed, [-1, self.embedding_dim])

        for idx in self.domain_dict:
            if (self.is_train_with_other_domain or not is_training) and idx == self.other_domain_key:
                domain_mask = get_domain_mask_light(self.domain_dict.get(idx) + self.other_domain_list_id, domain_hldr)
            else:
                domain_mask = get_domain_mask_light(self.domain_dict.get(idx), domain_hldr)

            domain_embed = tf.boolean_mask(vx_embed, domain_mask)
            domain_label = tf.boolean_mask(self.lbl_hldr, domain_mask) if is_training else None
            domain_predict = self.star_sub_netword_forward(domain_embed, idx, is_training)
            domain_ids = tf.cast(tf.where(domain_mask), tf.int32)
            if self.use_domain_tower:
                domain_bias = self.domain_tower(domain_embed, is_training)
                domain_predict = tf.add(domain_predict, domain_bias)
            predict_dict[idx] = domain_predict
            label_dict[idx] = domain_label
            all_ids.append(tf.reshape(domain_ids, [-1]))
            all_domain_idxs.append(idx)

        if is_training or self.config.multi_domain_algo:
            # 训练阶段分domain来进行梯度反向传播；或者split_sub_network=True时，pred也需要分domain存储
            return predict_dict, label_dict
        else:
            """
            Function: 在eval阶段，multi_domain_algo=False，则将pred按输入的顺序进行重排，保证其与label的顺序对应
            逻辑解释：
            通常multi_domain_algo和SPLIT_MODEL同为True或同为False。
            在model_training.py中，multi_domain_algo=True会将eval_lbl分domain存储，SPLIT_MODEL=True会将输出节点分开存储。
            """
            print(f'Warning! '
                  f'"config.multi_domain_algo" is False during evaluation. Merge all predictions.')
            all_predictions = [predict_dict.get(domain_idx) for domain_idx in all_domain_idxs]
            return tf.dynamic_stitch(all_ids, all_predictions), None

    def multi_domain_forward_split(self, wt_hldrs, id_hldrs, is_training=False, is_save=False):
        # eval / save
        predict_dict, label_dict = {}, {}
        for idx in self.domain_dict:
            domain_embed = self.construct_embedding(wt_hldrs[idx], id_hldrs[idx], self.merge_multi_hot,
                                                    train=is_training, is_save=is_save)
            domain_embed = tf.reshape(domain_embed, [-1, self.embedding_dim])
            domain_predict = self.star_sub_netword_forward(domain_embed, idx, is_training)
            if self.use_domain_tower:
                domain_bias = self.domain_tower(domain_embed, is_training)
                domain_predict = tf.add(domain_predict, domain_bias)
            predict_dict[idx] = domain_predict
        return predict_dict, label_dict

    def star_sub_netword_forward(self, domain_embed, idx, is_training):
        pn_embed = domain_embed
        pn_domain_id_embed = self.get_pn_domain_id_embed(pn_embed)
        domain_sm = self.domain_star_fcn(pn_embed, idx, training=is_training)
        domain_sa = self.auxiliary_network(pn_domain_id_embed, training=is_training)
        domain_predict = tf.add(domain_sm, domain_sa)
        return domain_predict

    def get_pn_domain_id_embed(self, pn_embed):
        with tf.variable_scope('get_domain_indicator', reuse=tf.AUTO_REUSE):
            batch_domain_param = tf.transpose(pn_embed, [1, 0])
            pn_domain_id_embed = tf.transpose(tf.boolean_mask(batch_domain_param, self.domain_indicator_flags), [1, 0])
            pn_domain_id_embed = tf.reshape(pn_domain_id_embed, [-1, self.domain_indicator_embedding_dim])
        return pn_domain_id_embed

    def auxiliary_network(self, inp, training=False):
        hidden_output = inp
        for i, _ in enumerate(self.h_ax_w):
            hidden_output = tf.matmul(hidden_output, self.h_ax_w[i]) + self.h_ax_b[i]
            if i < len(self.h_w) - 1:
                hidden_output = activate(self.act_func, hidden_output)
                if training:
                    hidden_output = tf.nn.dropout(hidden_output, keep_prob=self.keep_prob)
        hidden_output = self.final_forward_star(hidden_output, "auxiliary_final")
        return hidden_output

    def get_fcn_w_b(self, layer, domain_idx):
        _w = tf.multiply(self.h_w[layer], self.domain_w.get(domain_idx)[layer])  # h_w: 共享特征的长度   domain_w: 特有特征长度
        _b = tf.add(self.h_b[layer], self.domain_b.get(domain_idx)[layer])
        return _w, _b

    def domain_star_fcn(self, inp, domain_idx, training=False):
        hidden_output = inp
        print("domain_w len: {}, domain_idx: {}".format(len(self.domain_w), domain_idx))
        with tf.variable_scope("d_{}_FCN".format(domain_idx), reuse=tf.AUTO_REUSE):
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
        hidden_output = self.final_forward_star(hidden_output, "d_{}_FCN_final".format(domain_idx))
        return hidden_output

    def final_forward_star(self, hidden_output, key_word):
        with tf.variable_scope(key_word, reuse=tf.AUTO_REUSE):
            out_w = tf.get_variable('out_w', initializer=lambda: self.final_var_map.get('{}_out_w'.format(key_word)))
            out_b = tf.get_variable('out_b', initializer=lambda: self.final_var_map.get('{}_out_b'.format(key_word)))
            hidden_output = tf.matmul(hidden_output, out_w) + out_b
            output_y = tf.reshape(hidden_output, [-1])
        return output_y

    def get_loss(self, train_y, label, _lambda, is_domain=False):
        all_sample_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=train_y, labels=label)
        loss = tf.reduce_mean(all_sample_loss, keep_dims=is_domain, name='loss')
        loss = loss + _lambda * tf.nn.l2_loss(self.embed_v)
        return loss

    def init_weights_star(self):
        # sub domain
        self.init_weights_domain_w_b()
        # auxiliary network
        self.init_weights_domain_auxiliary_net()
        # final
        self.final_var_map_init()

    def init_weights_domain_w_b(self):
        # sub domain
        for domain_idx in self.domain_dict:
            each_domain_w, each_domain_b = [], []
            with tf.variable_scope("d_{}_w_b".format(domain_idx), reuse=tf.AUTO_REUSE):
                for i in range(len(self.all_deep_layer) - 1):
                    init_acts = [
                        ('d_{}_h{}_w'.format(domain_idx, i + 1), self.all_deep_layer[i: i + 2], 'random'),
                        ('d_{}_h{}_b'.format(domain_idx, i + 1), [self.all_deep_layer[i + 1]], 'random')
                    ]
                    var_map, _log = init_var_map(self.init_argv, init_acts)
                    self.log += _log
                    each_domain_w.append(tf.Variable(var_map['d_{}_h{}_w'.format(domain_idx, i + 1)],
                                                     name='d_{}_h{}_w'.format(domain_idx, i + 1)))
                    each_domain_b.append(tf.Variable(var_map['d_{}_h{}_b'.format(domain_idx, i + 1)],
                                                     name='d_{}_h{}_b'.format(domain_idx, i + 1)))
            self.domain_w[domain_idx] = each_domain_w
            self.domain_b[domain_idx] = each_domain_b

    def init_weights_domain_auxiliary_net(self):
        # auxiliary network
        with tf.variable_scope("ax_w_b"):
            for i in range(len(self.auxiliary_network_layers) - 1):
                init_acts = [
                    ('h{}_ax_w'.format(i + 1), self.auxiliary_network_layers[i: i + 2], 'random'),
                    ('h{}_ax_b'.format(i + 1), [self.auxiliary_network_layers[i + 1]], 'random')
                ]
                var_map, _log = init_var_map(self.init_argv, init_acts)
                self.log += _log
                self.h_ax_w.append(tf.Variable(var_map['h{}_ax_w'.format(i + 1)], name='h{}_ax_w'.format(i + 1)))
                self.h_ax_b.append(tf.Variable(var_map['h{}_ax_b'.format(i + 1)], name='h{}_ax_b'.format(i + 1)))

    def final_var_map_init(self):
        init_acts_final = []
        init_acts_final.extend([
            ('auxiliary_final_out_w', [int(self.auxiliary_network_layers[-1]), 1], 'random'),
            ('auxiliary_final_out_b', [1], 'zero')
        ])
        for idx in self.domain_dict:
            key_word = "d_{}_FCN_final".format(idx)
            init_acts_final.extend([
                ('{}_out_w'.format(key_word), [int(self.all_deep_layer[-1]), 1], 'random'),
                ('{}_out_b'.format(key_word), [1], 'zero'),
            ])
        self.final_var_map, log1 = init_var_map(self.init_argv, init_acts_final)

    def get_auxiliary_network_layers(self):
        self.auxiliary_network_layers = [self.domain_indicator_embedding_dim] + self.auxiliary_network_layers

    def split_embed(self, inp, layer, domain_idx):
        """第一层 将共享的放在前面，特有的放在后面"""
        if self.is_diff_features and layer == 0:
            gen_idx = self.domain_idx_2_diff_feat[domain_idx]
            share_embed = tf.gather(inp, self.share_feat_idxs, axis=1)
            spec_feat_idxs = [idx for idx in self.bias_feat_idxs_list[gen_idx] if idx not in self.share_feat_idxs]
            if spec_feat_idxs:
                spec_embed = tf.gather(inp, spec_feat_idxs, axis=1)
                concat_embed = tf.concat([share_embed, spec_embed], axis=1)
                return concat_embed
            else:
                return share_embed
        else:
            return inp
