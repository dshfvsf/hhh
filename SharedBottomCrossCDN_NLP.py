# encoding=utf-8
from __future__ import print_function
import numpy as np
import tensorflow as tf
import os
import pandas as pd
from sklearn.preprocessing import normalize

from train.models.tf_util import init_var_map, split_mask, split_param, sum_multi_hot
from train.models.SharedBottomCrossCDN import SharedBottomCrossCDN

np.set_printoptions(threshold=np.inf)


class SharedBottomCrossCDN_NLP(SharedBottomCrossCDN):
    def __init__(self, config, dataset_argv, architect_argv, init_argv, ptmzr_argv,
                 reg_argv, autodis_argv=None, log_file=None, loss_mode='full', merge_multi_hot=False,
                 batch_norm=True,
                 use_inter=True, use_bridge=True, use_autodis=True, debug_argv=None,
                 checkpoint=None, sess=None, feat_field_map=None, hcmd_argv=None, domain_list=None,
                 index_list_feat_id=0, list_to_domain_map=None):

        self.mem_idx_nlp, self.gen_idx_nlp = config.mem_idx_nlp, config.gen_idx_nlp
        self.embed_v, self.pre_embed_v = None, None
        self.shared_layer_shape = None

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
                          log_file, loss_mode, merge_multi_hot, batch_norm, use_inter, use_bridge, use_autodis, \
                          debug_argv, checkpoint, sess, feat_field_map, hcmd_argv, domain_list, index_list_feat_id, \
                          list_to_domain_map

        self._init_model(init_model_args)

        print(self.domain_list)

    def multi_domain_forward(self, wt_hldr, id_hldr, merge_multi_hot, is_training=False):
        domain_out = {}
        domain_moe_out = {}
        vx_embed = self.construct_embedding(wt_hldr, id_hldr, merge_multi_hot)
        vx_pre_embed = self.construct_pre_embedding(wt_hldr, id_hldr, merge_multi_hot)
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
            domain_pre_embed = tf.boolean_mask(vx_pre_embed, domain_mask)
            domain_sm = self.deep_and_cross_network(domain_concat_embed, domain_w, domain_b, is_training, idx)

            domain_moe_out[idx] = tf.reshape(
                self.dnn_moe_forward((domain_concat_embed, domain_pre_embed), domain_id_hldr, is_training=is_training,
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

    def expert_forward(self, embeds_and_idxs, scope, h_w=None, h_b=None, is_training=False):
        (embeds), (idxes) = embeds_and_idxs
        embed, pre_embed = embeds
        exclude_idx, include_pre_idx = idxes
        all_idxs = list(range(embed.shape[1]))
        sel_idx = [idx for idx in all_idxs if idx not in exclude_idx]
        all_pre_idxs = list(range(pre_embed.shape[1]))
        sel_pre_idx = [idx for idx in all_pre_idxs if idx in include_pre_idx]
        print("Selected idxs", sel_idx)
        print("Selected idxs", sel_pre_idx)

        embed = tf.reshape(tf.gather(embed, sel_idx, axis=1), [-1, len(sel_idx) * self.embedding_size])
        if len(sel_pre_idx) == 0:
            pre_embed = []
            nn_input = embed
        else:
            pre_embed = tf.reshape(tf.gather(pre_embed, sel_pre_idx, axis=1),
                                   [-1, len(sel_pre_idx) * self.embedding_size])
            nn_input = tf.concat([embed, pre_embed], 1)
        final_hl = self.mlp(nn_input, h_w, h_b, is_training=is_training, scope=scope)
        return final_hl

    def dnn_moe_forward(self, embeds, id_hldr, scope, domain_idx, is_training=False):
        embed, pre_embed = embeds
        embed = tf.reshape(embed, [-1, self.feature_num, self.embedding_size])
        pre_embed = tf.reshape(pre_embed, [-1, self.feature_num, self.embedding_size])
        embeds = embed, pre_embed
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
            mem_idxes = self.mem_idx, self.gen_idx_nlp
            final_hl_gen = self.expert_forward((embeds, mem_idxes), h_w=self.domain_h_w_gen[domain_idx],
                                               h_b=self.domain_h_b_gen[domain_idx], is_training=is_training,
                                               scope=scope)
        with tf.variable_scope("mem_exp"):
            gen_idxes = self.gen_idx, self.mem_idx_nlp
            final_hl_mem = self.expert_forward((embeds, gen_idxes), h_w=self.domain_h_w_mem[domain_idx],
                                               h_b=self.domain_h_b_mem[domain_idx], is_training=is_training,
                                               scope=scope)

        final_hl = w1 * final_hl_gen + (1. - w1) * final_hl_mem

        return final_hl

    def _init_weights(self, input_dim, embedding_size):
        init_acts = []
        for domain_idx in self.domain_list:
            for j in range(len(self.domain_layers_shape) - 1):
                init_acts.extend([
                    ('d_{}_h{}_w'.format(domain_idx, j + 1), self.domain_layers_shape[j: j + 2], 'random'),
                    ('d_{}_h{}_b'.format(domain_idx, j + 1), [self.domain_layers_shape[j + 1]], 'zeros')
                ])

        self.shared_layer_shape = [self.embedding_dim, 256]
        init_acts.extend([
            ('shared_layer', self.shared_layer_shape, 'random')
        ])
        var_map, log = init_var_map(self.init_argv, init_acts)
        self.log += log
        self.init_mlp_weights(var_map)

        print('pre init')
        random_uniform_init = self.pre_embedding_init(input_dim)
        print('pre init done')
        # init embedding
        init_acts = [
            ('embed', [input_dim, embedding_size], 'random')
        ]
        var_map, log = init_var_map(self.init_argv, init_acts)
        self.embed_v = tf.Variable(var_map['embed'], name='embedding', validate_shape=False)
        self.pre_embed_v = tf.Variable(random_uniform_init, name='pre_embedding', validate_shape=False,
                                       trainable=False)

    def pre_embedding_init(self, input_dim):
        pre_id_emb = self.get_pre_id_emb()
        featuremap_index = self.get_featuremap_index()
        z = 0
        random_emb = np.random.uniform(self.config.MIN_VALUE, self.config.MAX_VALUE,
                                       [self.config.pre_model_size])
        init_emb = pre_id_emb.get("unknown", random_emb)
        random_uniform_init = np.tile(init_emb, (input_dim, 1))
        random_uniform_init = random_uniform_init.astype('f')
        for index in featuremap_index:
            id_ = featuremap_index[index]
            if id_ in pre_id_emb:
                z += 1
                random_uniform_init[int(index)] = pre_id_emb[id_]
        print('match appid size', z)
        random_uniform_init = normalize(random_uniform_init, norm='l2', axis=1)
        return random_uniform_init

    def get_featuremap_index(self):
        featuremap_dir = os.path.join(self.config.DATA_DIR, 'model/featureMap.txt')
        print('featuremap dir ', featuremap_dir)
        featuremap_index = {}
        with open(featuremap_dir, 'r') as f:
            for samples in f:
                sample = samples.strip().split('\t')
                prefix_ = sample[0].split(',')
                if len(prefix_) < 2 or len(prefix_[1]) == 0:
                    print(prefix_)
                    continue
                id_ = prefix_[1]
                index = sample[1]
                featuremap_index[index] = prefix_[1]
        print('length emb', len(featuremap_index))
        return featuremap_index

    def get_pre_id_emb(self):
        print("start converting embedding")
        appids_path = os.path.join(self.config.pre_model_dir, self.config.appids_file)
        if os.path.exists(appids_path):
            print("appids file path: ", appids_path)
        else:
            raise FileNotFoundError("{} does not exist!".format(appids_path))
        df = pd.read_csv(appids_path)
        appids = df['appid'].to_numpy()
        embed_path = os.path.join(self.config.pre_model_dir, self.config.embedding_name)
        if os.path.exists(embed_path):
            print("pre trained embedding path: ", embed_path)
        else:
            raise FileNotFoundError("{} does not exist!".format(embed_path))
        embed = np.load(embed_path)
        zipped_embed = zip(appids, embed)
        name_embedding = {}
        for _, (app_id, emb) in enumerate(zipped_embed):
            name_embedding[str(app_id)] = emb.tolist()
        print('length emb', len(name_embedding))
        return name_embedding

    def extra_data_member_part(self):
        self.mem_idx, self.gen_idx = self.config.mem_idx, self.config.gen_idx
        self.set_in_dim()

    def set_in_dim(self):
        in_dim = self.embedding_dim + self.dense_num * int(self.use_dense_features)
        self.in_dim_gen = in_dim + (len(self.gen_idx_nlp) - len(self.mem_idx)) * self.embedding_size
        self.in_dim_mem = in_dim + (len(self.mem_idx_nlp) - len(self.gen_idx)) * self.embedding_size

    def construct_pre_embedding(self, wt_hldr, id_hldr, merge_multi_hot=False):
        mask = tf.expand_dims(wt_hldr, 2)
        if merge_multi_hot and self.num_multihot > 0:
            # *_hot_mask is weight(values that follow the ids in the dataset, different from weight of param) that used
            if self.config.DYNAMIC_LENGTH:
                one_hot_mask, multi_hot_mask = split_mask(mask, self.multi_hot_flags, self.multi_hot_variable_len)
                if self.config.POSITION_EMBEDDING:
                    d_position_value = []
                    for index in self.multi_hot_variable_len:
                        d_position_value.extend(list(range(index)))
            else:
                one_hot_mask, multi_hot_mask = split_mask(mask, self.multi_hot_flags, self.num_multihot)
                if self.config.POSITION_EMBEDDING:
                    d_position_value = []
                    d_position_value.extend(list(range(self.multi_hot_len)) * self.num_multihot)

            one_hot_v, multi_hot_v = split_param(self.pre_embed_v, id_hldr, self.multi_hot_flags)
            if self.config.POSITION_EMBEDDING:
                d_position_embed = tf.gather(self.posi_embed, d_position_value)
                multi_hot_v = multi_hot_v + tf.expand_dims(d_position_embed, 0)

            one_hot_vx = tf.multiply(one_hot_v, one_hot_mask)
            if self.config.DYNAMIC_LENGTH:
                multi_hot_vx = sum_multi_hot(multi_hot_v, multi_hot_mask, self.multi_hot_variable_len)
            else:
                multi_hot_vx = sum_multi_hot(multi_hot_v, multi_hot_mask, self.num_multihot)
            vx_embed = tf.concat([one_hot_vx, multi_hot_vx], axis=1)
        else:
            vx_embed = tf.multiply(tf.gather(self.pre_embed_v, id_hldr), mask)
        return vx_embed
