# encoding=utf-8
from __future__ import print_function
import tensorflow as tf

from train.models.tf_util import init_var_map, activate, get_domain_mask_noah
from models.STAR import STAR
from models.DCN_SFPS import DCN_SFPS


class STAR_DCN_EPNet(STAR):
    def __init__(self, config, dataset_argv, architect_argv, init_argv,
                 ptmzr_argv, reg_argv, distilled_argv,
                 loss_mode='full', merge_multi_hot=False,
                 batch_norm=True, distill=False,
                 checkpoint=None, sess=None, regression_task=False,
                 use_cross=True, use_linear=False, use_fm=False,
                 star_argv=None, auxiliary_network_layers=None,
                 diff_feats_argv=None,
                 ):
        # cross part
        self.use_cross = config.USE_CROSS
        self.domain_cross_w, self.domain_cross_b = {}, {}
        self.final_var_map = None
        print("use_cross: {}".format(self.use_cross))

        # epnet part
        self.use_epnet = config.USE_EPNET
        self.domain_col_idx = config.domain_col_idx
        self.epnet_input_col_idx = config.epnet_input_col_idx
        self.dnn_input_stop_grad = config.dnn_input_stop_grad
        print("use_epnet: {}".format(self.use_epnet))
        print("epnet_input_col_idx: {}".format(self.epnet_input_col_idx))
        print("dnn_input_stop_grad: {}".format(self.dnn_input_stop_grad))

        super(STAR_DCN_EPNet, self).__init__(config, dataset_argv, architect_argv, init_argv,
                                             ptmzr_argv, reg_argv, distilled_argv,
                                             loss_mode, merge_multi_hot,
                                             batch_norm, distill,
                                             checkpoint, sess, regression_task,
                                             use_cross, use_linear, use_fm,
                                             star_argv, auxiliary_network_layers,
                                             diff_feats_argv, model_name='STAR_DCN_EPNet')

    def star_sub_netword_forward(self, domain_embed, idx, is_training):
        pn_embed = domain_embed
        pn_domain_id_embed = self.get_pn_domain_id_embed(pn_embed)
        domain_sm = self.domain_star_cross_deep(pn_embed, idx, training=is_training)
        domain_sa = self.auxiliary_network(pn_domain_id_embed, training=is_training)
        domain_predict = tf.add(domain_sm, domain_sa)
        return domain_predict

    def star_cross_layer(self, cross_w, cross_b, cross_input, num_cross_layer):
        # embedding layer
        x_0 = cross_input
        # cross layer
        x_l = x_0
        for i in range(num_cross_layer):
            xlw = tf.tensordot(x_l, cross_w[i], axes=1)
            x_l = x_0 * tf.expand_dims(xlw, -1) + cross_b[i] + x_l
            x_l.set_shape((None, self.embedding_dim))

        return x_l

    def domain_star_cross_deep(self, inp, domain_idx, training=False):
        hidden_output = inp
        cross_result = None
        with tf.variable_scope("d_{}_DCN".format(domain_idx), reuse=tf.AUTO_REUSE):
            if self.use_cross:
                # cross part
                print('domain_cross_w_d_{} is {}'.format(domain_idx, self.domain_cross_w.get(domain_idx)))
                cross_w_layers = self.domain_cross_w.get(domain_idx)
                cross_b_layers = self.domain_cross_b.get(domain_idx)
                cross_input = tf.reshape(inp, [-1, self.embedding_dim])
                cross_result = self.star_cross_layer(cross_w_layers, cross_b_layers, cross_input, self.num_cross_layer)

            # deep part
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
        # 以下一行是新加的，用于concat cross结果与隐层输出
        hidden_output = tf.concat([cross_result, hidden_output], 1) if cross_result is not None else hidden_output
        hidden_output = self.final_forward_star(hidden_output, "d_{}_DCN_final".format(domain_idx))
        return hidden_output

    def init_weights_star(self):
        # sub domain
        self.init_weights_domain_cross_w_b()
        self.init_weights_domain_w_b()
        # auxiliary network
        self.init_weights_domain_auxiliary_net()
        # final
        self.final_var_map_init_star_dcn()

    def init_weights_domain_cross_w_b(self):
        if not self.use_cross:
            return
        # sub domain
        for domain_idx in self.domain_dict:
            with tf.variable_scope("d_{}_cross_w_b".format(domain_idx), reuse=tf.AUTO_REUSE):
                init_acts = [('d_{}_cross_w'.format(domain_idx), [self.num_cross_layer, self.embedding_dim], 'random'),
                             ('d_{}_cross_b'.format(domain_idx), [self.num_cross_layer, self.embedding_dim], 'random')]
                var_map, _log = init_var_map(self.init_argv, init_acts)
                self.log += _log
                self.domain_cross_w[domain_idx] = tf.Variable(var_map['d_{}_cross_w'.format(domain_idx)],
                                                    name='d_{}_cross_w'.format(domain_idx))
                self.domain_cross_b[domain_idx] = tf.Variable(var_map['d_{}_cross_b'.format(domain_idx)],
                                                    name='d_{}_cross_b'.format(domain_idx))
        print('self.domain_cross_w is {}'.format(self.domain_cross_w))
        print('self.domain_cross_b is {}'.format(self.domain_cross_b))

    def final_var_map_init_star_dcn(self):
        init_acts_final = []
        init_acts_final.extend([
            ('auxiliary_final_out_w', [int(self.auxiliary_network_layers[-1]), 1], 'random'),
            ('auxiliary_final_out_b', [1], 'zero')
        ])
        final_input_dim = int(self.all_deep_layer[-1]) + (int(self.embedding_dim) if self.use_cross else 0)
        for idx in self.domain_dict:
            key_word = "d_{}_DCN_final".format(idx)
            init_acts_final.extend([
                ('{}_out_w'.format(key_word), [final_input_dim, 1], 'random'),
                ('{}_out_b'.format(key_word), [1], 'zero'),
            ])
        self.final_var_map, log1 = init_var_map(self.init_argv, init_acts_final)


    def construct_embedding(self, wt_hldr, id_hldr, merge_multi_hot=False, train=True, is_save=False):
        """
        construct the embedding layer and epnet part
        """
        if not self.use_sfps:
            # DCN.construct_embedding
            vx_embed = super(DCN_SFPS, self).construct_embedding(wt_hldr, id_hldr, merge_multi_hot)
        else:
            if not is_save:
                # train / eval
                vx_embed = super(STAR, self).construct_embedding(wt_hldr, id_hldr,
                                                                 merge_multi_hot, train)
            else:
                # save
                vx_embed = super(STAR, self).sfps_save_construct_embedding(id_hldr, wt_hldr,
                                                                           merge_multi_hot, train)

        if self.use_epnet:
            epnet_input_embed = self.get_single_embeds(id_hldr, self.epnet_input_col_idx)
            print("original epnet_input_embed: {}".format(epnet_input_embed))
            epnet_out = self.epnet_part(vx_embed, epnet_input_embed)
            vx_embed = tf.reshape(epnet_out, [-1, self.embedding_dim // self.embedding_size, self.embedding_size])

        return vx_embed

    def epnet_part(self, vx_embed, epnet_input_embed):
        vx_embed = tf.reshape(vx_embed, [-1, self.embedding_dim])
        print("vx_embed: {}".format(vx_embed))
        print("embedding_dim: {}".format(self.embedding_dim))
        print("feature_nums: {}".format(self.num_multihot + self.num_onehot))

        # EPNet的输入中主特征部分需要阻隔梯度
        vx_embed_sg = tf.stop_gradient(vx_embed)
        epnet_input = tf.concat([vx_embed_sg, epnet_input_embed], axis=1)
        print("EPNet gate_nu input: {}".format(epnet_input))

        # gate nu 0
        with tf.variable_scope('epnet', reuse=tf.AUTO_REUSE):
            hidden_out = tf.layers.dense(epnet_input, self.embedding_dim, activation='relu', name='ep_gate_nu_0')
            epnet_output = tf.layers.dense(hidden_out, self.embedding_dim, activation='sigmoid', name='ep_gate_nu_1')
            print("EPNet gate_nu hidden_out: {}".format(epnet_output))

            epnet_embed = vx_embed * epnet_output * 2
            print("EPNet embedding out: {}".format(epnet_embed))

        return epnet_embed

    def get_single_embeds(self, id_hldr, col_idxs):
        embeddings = []
        for col_idx in col_idxs:
            single_id_hldr = tf.slice(id_hldr, [0, col_idx], [-1, 1])  # batch size, 1
            batch_param = tf.gather(self.embed_v, single_id_hldr)  # batch size, 1, embedding size
            batch_param = tf.reshape(batch_param, [-1, self.embedding_size])
            embeddings.append(batch_param)
        return tf.concat(embeddings, axis=1)
