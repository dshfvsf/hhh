# encoding=utf-8
from __future__ import print_function
import numpy as np
import tensorflow as tf
import os
import pandas as pd

from train.models.tf_util import init_var_map
from train.models.SharedBottomCross import SharedBottomCross

np.set_printoptions(threshold=np.inf)


class SharedBottomCrossCDN(SharedBottomCross):
    def __init__(self, config, dataset_argv, architect_argv, init_argv, ptmzr_argv,
                 reg_argv, autodis_argv=None, log_file=None, loss_mode='full', merge_multi_hot=False,
                 batch_norm=True,
                 use_inter=True, use_bridge=True, use_autodis=True, debug_argv=None,
                 checkpoint=None, sess=None, feat_field_map=None, hcmd_argv=None, domain_list=None,
                 index_list_feat_id=0, list_to_domain_map=None):

        self._init_shared_bottom_cross(architect_argv)

        self.use_dense_features = config.USE_DENSE_FEATURES
        self.dense_num = 0
        self.scope_name = 'dcn'
        self.act_func = config.ACT_FUNC
        self.batch_norm = config.BATCH_NORM

        self.feature_num = None
        self.app_features = None
        self.alpha = None
        self.domain_gate_w = None
        self.domain_gate_b = None
        self.all_deep_layer_gen = None
        self.domain_h_w_gen = None
        self.domain_h_b_gen = None
        self.all_deep_layer_mem = None
        self.domain_h_w_mem = None
        self.domain_h_b_mem = None
        self.mem_idx = config.mem_idx
        self.gen_idx = config.gen_idx
        self.in_dim_gen = None
        self.in_dim_mem = None

        init_model_args = config, dataset_argv, architect_argv, init_argv, ptmzr_argv, reg_argv, autodis_argv, \
            log_file, loss_mode, merge_multi_hot, batch_norm, use_inter, use_bridge, use_autodis, debug_argv, \
            checkpoint, sess, feat_field_map, hcmd_argv, domain_list, index_list_feat_id, list_to_domain_map

        self._init_model(init_model_args)

        print(self.domain_list)

    def _init_cdn(self):
        self.feature_num = self.num_multihot + self.num_onehot
        self.app_features = tf.convert_to_tensor(self.get_app_features(), dtype=tf.float32)
        self.alpha = tf.cast(1. - (self.global_step / (self.config.gamma * self.config.N_EPOCH)) ** 2, dtype=tf.float32)
        self.extra_data_member_part()
        self.domain_gate_w, self.domain_gate_b = self.init_moe_vars()
        self.all_deep_layer_gen, self.domain_h_w_gen, self.domain_h_b_gen = self.dnn_moe_variable_part(self.in_dim_gen)
        self.all_deep_layer_mem, self.domain_h_w_mem, self.domain_h_b_mem = self.dnn_moe_variable_part(self.in_dim_mem)

    def multi_domain_forward(self, wt_hldr, id_hldr, merge_multi_hot, is_training=False):
        domain_out = {}
        domain_moe_out = {}
        vx_embed = self.construct_embedding(wt_hldr, id_hldr, merge_multi_hot)
        vx_embed = tf.reshape(vx_embed, [-1, self.embedding_dim])

        shared_out = vx_embed
        if self.num_cross_layer > 0 and self.cross_combine.startswith("stack"):
            shared_cross = self.cross_layer(vx_embed, self.num_cross_layer)
            if self.cross_combine == "stack_concat":
                shared_out = tf.concat([vx_embed, shared_cross], 1)
            else:
                shared_out = shared_cross

            print("[SharedBottomCross], cross_combine: {}, shared_out: {}".format(self.cross_combine, shared_out.shape))

        domain_feat = tf.cast(id_hldr[:, self.index_list_feat_id], tf.int32)
        domain_w = self.domain_w
        domain_b = self.domain_b

        if self.use_star_param_sharing is not None:
            domain_w, domain_b = self.star_param_sharing_part()

        for idx, list_feats in self.domain_list_feat_map.items():
            domain_mask = self._get_domain_mask(list_feats, domain_feat)
            domain_id_hldr = tf.boolean_mask(id_hldr, domain_mask)
            domain_concat_embed = tf.boolean_mask(shared_out, domain_mask)

            domain_sm = self.deep_and_cross_network(domain_concat_embed, domain_w, domain_b, is_training, idx)

            domain_moe_out[idx] = tf.reshape(
                self.dnn_moe_forward(domain_concat_embed, domain_id_hldr, is_training=is_training,
                                     scope='domain_{}_moe'.format(idx), domain_idx=idx), [-1, ])

            domain_sa = self.auxiliary_network(domain_concat_embed, is_training)
            if domain_sa is not None:
                print("[SharedBottomCross]: domain_sa: {}, domain_sm: {}".format(domain_sa.shape, domain_sm.shape))
                domain_predict = tf.add(domain_sm, domain_sa)

            else:
                print("[SharedBottomCross]: domain_sm: {}".format(domain_sm.shape))
                domain_predict = domain_sm

            domain_out[idx] = tf.reshape(domain_predict, [-1, ])

            domain_out[idx] += domain_moe_out[idx]

        return domain_out

    def extra_data_member_part(self):
        in_dim = self.embedding_dim + self.dense_num * int(self.use_dense_features)
        self.in_dim_gen = in_dim - len(self.mem_idx) * self.embedding_size
        self.in_dim_mem = in_dim - len(self.gen_idx) * self.embedding_size

    def init_moe_vars(self):
        gate_layer_sizes = [1, 1]
        domain_gate_w, domain_gate_b = [], []
        for domain_idx in self.domain_list:
            init_acts_all = []
            init_acts_all += self._init_acts(gate_layer_sizes, var_name_template='d{}_moe_gate'.format(domain_idx))
            var_map, log = init_var_map(self.init_argv, init_acts_all)
            gate_w, gate_b = self._init_vars(gate_layer_sizes, var_map, 'd{}_moe_gate'.format(domain_idx))
            domain_gate_w.append(gate_w)
            domain_gate_b.append(gate_b)

        return domain_gate_w, domain_gate_b

    @staticmethod
    def _init_acts(layer_sizes, var_name_template):
        init_acts = []
        for i in range(len(layer_sizes) - 1):
            init_acts.extend([
                ("{}_{}_w".format(var_name_template, i + 1), layer_sizes[i: i + 2], 'random'),
                ("{}_{}_b".format(var_name_template, i + 1), [layer_sizes[i + 1]], 'random')
            ])

        return init_acts

    @staticmethod
    def _init_vars(layer_sizes, var_map, var_name_template):
        var_list_w, var_list_b = [], []
        with tf.variable_scope(var_name_template):
            for i in range(len(layer_sizes) - 1):
                w_name = '{}_{}_w'.format(var_name_template, i + 1)
                b_name = '{}_{}_b'.format(var_name_template, i + 1)
                var_list_w.append(tf.Variable(var_map[w_name], name=w_name))
                var_list_b.append(tf.Variable(var_map[b_name], name=b_name))

        return var_list_w, var_list_b

    def dnn_moe_variable_part(self, in_dim):
        all_deep_layer = [in_dim] + self.deep_layers + [1]
        domain_h_w, domain_h_b = [], []
        for idx in self.domain_list:
            init_acts = []
            for i in range(len(all_deep_layer) - 1):
                init_acts.extend([('moe_d_{}_h{}_w'.format(idx, i + 1), all_deep_layer[i: i + 2], 'random'),
                                  ('moe_d_{}_h{}_b'.format(idx, i + 1), [all_deep_layer[i + 1]], 'random')])
            var_map, log = init_var_map(self.init_argv, init_acts)
            self.log += log
            h_w = []
            h_b = []
            for i in range(len(all_deep_layer) - 1):
                h_w.append(tf.Variable(var_map['moe_d_{}_h{}_w'.format(idx, i + 1)]))
                h_b.append(tf.Variable(var_map['moe_d_{}_h{}_b'.format(idx, i + 1)]))
            domain_h_w.append(h_w)
            domain_h_b.append(h_b)
        return all_deep_layer, domain_h_w, domain_h_b

    def expert_forward(self, embed_and_idxs, scope, h_w=None, h_b=None, is_training=False):
        embed, exclude_idx = embed_and_idxs
        all_idxs = list(range(embed.shape[1]))
        sel_idx = [idx for idx in all_idxs if idx not in exclude_idx]
        print("Selected idxs", sel_idx)
        embed = tf.reshape(tf.gather(embed, sel_idx, axis=1), [-1, len(sel_idx) * self.embedding_size])
        nn_input = embed
        final_hl = self.mlp(nn_input, h_w, h_b, is_training=is_training, scope=scope)
        return final_hl

    def dnn_moe_forward(self, embed, id_hldr, scope, domain_idx, is_training=False):
        embed = tf.reshape(embed, [-1, self.feature_num, self.embedding_size])
        # gate
        idx = tf.cast(id_hldr[:, 4], dtype=tf.int32)
        max_idx = self.app_features.shape[0]
        zeros = tf.zeros_like(idx, dtype=tf.int32)
        zeros_fl = tf.zeros_like(idx, dtype=tf.float32)
        freq = tf.where(idx < max_idx, tf.gather(self.app_features, tf.where(idx < max_idx, idx, zeros)), zeros_fl)
        gate = self.mlp(tf.expand_dims(freq, 1), self.domain_gate_w[domain_idx], self.domain_gate_b[domain_idx],
                        is_training=is_training, scope=scope)
        w1 = tf.sigmoid(gate * self.config.gate_cons)

        with tf.variable_scope("gen_exp"):
            gen_embed_and_idxs = embed, self.mem_idx
            final_hl_gen = self.expert_forward(gen_embed_and_idxs, h_w=self.domain_h_w_gen[domain_idx],
                                               h_b=self.domain_h_b_gen[domain_idx], is_training=is_training,
                                               scope=scope)
        with tf.variable_scope("mem_exp"):
            mem_embed_and_idxs = embed, self.gen_idx
            final_hl_mem = self.expert_forward(mem_embed_and_idxs, h_w=self.domain_h_w_mem[domain_idx],
                                               h_b=self.domain_h_b_mem[domain_idx], is_training=is_training,
                                               scope=scope)

        final_hl = w1 * final_hl_gen + (1. - w1) * final_hl_mem

        return final_hl

    def get_app_features(self):
        app_features_file = os.path.join('/opt/huawei/dataset/', self.config.FINETUNE_ID_DATA, 'app_features.pkl')
        df = pd.read_pickle(app_features_file)    # ['App.ID', 'App.IDX', 'App.Cat', 'App.Dev', 'App.size', 'clicks']
        df.drop(labels=['App.ID', 'App.Cat', 'App.Dev', 'App.size'], axis=1, inplace=True)
        df["App.IDX"] = df["App.IDX"].astype(int)
        df = df.set_index("App.IDX")
        df = df.reindex(list(range(df.index.max() + 1)), fill_value=0)
        if self.config.freq_norm == "percentile":
            df["clicks_norm"] = df["clicks"].rank(pct=True)
        elif self.config.freq_norm == "norm":
            df["clicks_norm"] = (df['clicks'] - df['clicks'].min()) / (df['clicks'].max() - df['clicks'].min())
        else:
            raise Exception("Freq_norm as set in config must be either percentile or norm")
        return df["clicks_norm"].values
