# encoding=utf-8
from __future__ import print_function
import json
import numpy as np
import tensorflow as tf

from train.models.tf_util import init_var_map
from train.models.multi_scenario_base import MultiScenarioBase, my_print

np.set_printoptions(threshold=np.inf)


class SharedBottom(MultiScenarioBase):
    def __init__(self, config, dataset_argv, architect_argv, init_argv, ptmzr_argv,
                 reg_argv, autodis_argv=None, log_file=None, loss_mode='full', merge_multi_hot=False,
                 batch_norm=True,
                 use_inter=True, use_bridge=True, use_autodis=True, debug_argv=None,
                 checkpoint=None, sess=None, feat_field_map=None, hcmd_argv=None, domain_list=None,
                 index_list_feat_id=0, list_to_domain_map=None):

        self.para_init()

        self.embedding_size, self.num_cross_layer, self.deep_layers, act_func = architect_argv[:4]
        self.random_domain_index_range = 0 
        if len(architect_argv) >= 5:
            self.random_domain_index_range = architect_argv[4]

        self.target_domain_out = None

        init_model_args = config, dataset_argv, architect_argv, init_argv, ptmzr_argv, reg_argv, autodis_argv, \
            log_file, loss_mode, merge_multi_hot, batch_norm, use_inter, use_bridge, use_autodis, debug_argv, \
            checkpoint, sess, feat_field_map, hcmd_argv, domain_list, index_list_feat_id, list_to_domain_map

        self._init_model(init_model_args)

    def _init_aux(self):
        pass

    def _init_cdn(self):
        pass

    def _init_model(self, init_model_args):

        config, dataset_argv, architect_argv, init_argv, ptmzr_argv, reg_argv, autodis_argv, log_file, loss_mode, \
            merge_multi_hot, batch_norm, use_inter, use_bridge, use_autodis, debug_argv, checkpoint, sess, \
            feat_field_map, hcmd_argv, domain_list, index_list_feat_id, list_to_domain_map = init_model_args

        self.config = config
        self.use_sfps = config.USE_SFPS
        self.log_file = log_file
        if loss_mode != 'full':
            raise NotImplementedError()
        self.new_param_start_index = 0
        self.log = json.dumps(locals(), default=str, sort_keys=True, indent=4)
        self.init_argv = init_argv
        input_dim, input_dim4lookup, self.multi_hot_flags, self.multi_hot_len, self.domain_flags,\
            self.item_flags, self.user_flags = dataset_argv
        self.multi_hot_variable_len = config.MULTI_HOT_VARIABLE_LEN
        self.one_hot_flags = [not flag for flag in self.multi_hot_flags]
        self.use_inter = use_inter
        self.use_autodis = use_autodis

        self.pertubation, self.conf_orth, self.conf_backbone, self.conf_bias = hcmd_argv

        self.keep_prob, _lambda, l1_lambda = reg_argv
        self.ptmzr_argv = ptmzr_argv

        self.num_onehot = sum(self.one_hot_flags)

        try:
            if self.config.DYNAMIC_LENGTH:
                self.num_multihot = len(self.multi_hot_variable_len)
            else:
                self.num_multihot = int(sum(self.multi_hot_flags) / self.multi_hot_len)

        except ZeroDivisionError:
            print("You can't divide by 0!")

        self.fetch_dict = {}

        # ---------------------------------
        my_print("num_multihot: {}, num_onehot: {}, embedding_size:{}".format(self.num_multihot, self.num_onehot,
                                                                              self.embedding_size), self.log_file)

        self.global_step = tf.placeholder(tf.int64, name='global_step')

        self.domain_indicator_flags = []
        for i in self.domain_flags:
            self.domain_indicator_flags.extend([i] * self.embedding_size)
        self.domain_indicator_embedding_dim = sum([i for i in self.domain_indicator_flags if i])

        self.item_indicator_flags = []
        for i in self.item_flags:
            self.item_indicator_flags.extend([i] * self.embedding_size)
        self.item_indicator_dim = sum([i for i in self.item_indicator_flags if i])

        my_print("domain_indicator_flags:\n{}".format(self.domain_indicator_flags), self.log_file)  # DEBUG

        self.init_placeholder(input_dim4lookup)
        self.get_embedding_dim(merge_multi_hot, self.embedding_size, input_dim4lookup)

        '''
        reverse the map from list_id -> domains
                        to   domain  -> list_ids
        '''
        self.domain_list_feat_map = {} 
        for list_id, domains in list_to_domain_map.items():
            for domain in domains:
                if domain not in self.domain_list_feat_map:
                    self.domain_list_feat_map[domain] = []

                self.domain_list_feat_map[domain].append(list_id)

        print("######### domain_list_feat_map: {}".format(self.domain_list_feat_map))

        self.domain_list = [i for i in range(len(self.domain_list_feat_map))]

        self.init_weights(input_dim, self.embedding_size)
        self._init_aux()
        self.index_list_feat_id = index_list_feat_id
        self.list_to_domain_map = list_to_domain_map

        self._init_cdn()

        '''
        training outputs for all the domains and,
        sum of the domain losses
        '''
        domain_out = self.multi_domain_forward(self.wt_hldr, self.id_hldr,
                                               merge_multi_hot, is_training=True)
        self.all_domain_loss_sum = self.get_all_domain_loss_sum(domain_out, self.lbl_hldr, self.id_hldr, _lambda)

        '''
        evaluation outputs for all the domains
        '''
        self.eval_domain_out = self.multi_domain_forward(self.eval_wt_hldr, self.eval_id_hldr,
                                                         merge_multi_hot, is_training=False)

        for idx in self.eval_domain_out:
            self.eval_domain_out[idx] = tf.nn.sigmoid(self.eval_domain_out[idx])

        if self.random_domain_index_range != 0: 
            '''
            this is only used for ensemble models: randomly select an output as the prediction for the main domain
            `eval_domain_out[0]` will be modified
            '''
            print("[Ensemble-SharedBottom] Take random domain output")
            out_all_tensor = tf.concat([tf.expand_dims(v, 1) for v in self.eval_domain_out.values()], axis=1)
            jj = tf.transpose(tf.cast(
                    tf.random.categorical(
                           [[1.0] * self.random_domain_index_range],
                           tf.shape(out_all_tensor)[0]), tf.int32))
            ii = tf.expand_dims(tf.range(tf.shape(out_all_tensor)[0]), 1)
            random_indices = tf.concat([ii, jj], axis=1)
            self.eval_domain_out[0] = tf.gather_nd(params=out_all_tensor, indices=random_indices)

        try:
            self.target_domain_out = tf.identity(self.eval_domain_out[0], name='predictionNode')
        except KeyError:
            print('error')

        self.eval_domain_label = self.get_domain_labels(self.eval_id_hldr, self.eval_lbl_hldr)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops), tf.variable_scope('Optimizer'):
            learning_rate = tf.train.exponential_decay(learning_rate=self.ptmzr_argv[1], global_step=self.global_step,
                                                       decay_rate=self.ptmzr_argv[3], decay_steps=self.ptmzr_argv[4],
                                                       staircase=False)
            self.trainer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=self.ptmzr_argv[2])

            self.domain_opz = self.trainer.minimize(self.all_domain_loss_sum)
            log = 'optimizer: %s, learning rate: %g, epsilon: %g\n' % (ptmzr_argv[0], ptmzr_argv[1], ptmzr_argv[2])
            my_print(log, self.log_file)
        my_print("model init finish", self.log_file)


    def multi_domain_forward(self, wt_hldr, id_hldr, merge_multi_hot, is_training=False):
        domain_out = {}
        vx_embed = self.construct_embedding(wt_hldr, id_hldr, merge_multi_hot)
        vx_embed = tf.reshape(vx_embed, [-1, self.embedding_dim])
        
        shared_out = vx_embed
        domain_feat = tf.cast(id_hldr[:, self.index_list_feat_id], tf.int32)
        for idx, list_feats in self.domain_list_feat_map.items():
            domain_mask = self._get_domain_mask(list_feats, domain_feat)
            domain_concat_embed = tf.boolean_mask(shared_out, domain_mask)
            try:
                domain_out[idx] = tf.reshape(
                    self.mlp(domain_concat_embed, self.domain_w[idx], self.domain_b[idx], is_training=is_training,
                             scope='domain_%s' % idx, is_output=True), [-1, ])
            except KeyError:
                print('error')

        return domain_out

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

        # init embedding
        init_acts = [
            ('embed', [input_dim, embedding_size], 'random')
        ]
        var_map, log = init_var_map(self.init_argv, init_acts)
        self.embed_v = tf.Variable(var_map['embed'], name='embedding', validate_shape=False)


    def init_weights(self, input_dim, embedding_size):
        # init domain and final layers
        self.domain_layers_shape = [self.embedding_dim] + self.deep_layers + [1]  # final pred node
        self._init_weights(input_dim, embedding_size)

    def init_mlp_weights(self, var_map):

        for domain_idx in self.domain_list:
            each_domain_w = []
            each_domain_b = []
            for i in range(len(self.domain_layers_shape) - 1):
                each_domain_w.append(tf.Variable(var_map['d_{}_h{}_w'.format(domain_idx, i + 1)],
                                                 name='d_{}_h{}_w'.format(domain_idx, i + 1)))
                each_domain_b.append(tf.Variable(var_map['d_{}_h{}_b'.format(domain_idx, i + 1)],
                                                 name='d_{}_h{}_b'.format(domain_idx, i + 1)))
            self.domain_w[domain_idx] = each_domain_w
            self.domain_b[domain_idx] = each_domain_b

        self.shared_layer = tf.Variable(var_map['shared_layer'], name='shared_layer')
