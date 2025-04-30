# encoding=utf-8
from __future__ import print_function
import numpy as np
import tensorflow as tf

from train.models.tf_util import init_var_map, activate
from train.models.SharedBottom import SharedBottom
np.set_printoptions(threshold=np.inf)


class SharedBottomCross(SharedBottom):
    def __init__(self, config, dataset_argv, architect_argv, init_argv, ptmzr_argv,
                reg_argv, autodis_argv=None, log_file=None, loss_mode='full', merge_multi_hot=False,
                batch_norm=True,
                use_inter=True, use_bridge=True, use_autodis=True, debug_argv=None,
                checkpoint=None, sess=None, feat_field_map=None, hcmd_argv=None, domain_list=None,
                index_list_feat_id=0, list_to_domain_map=None):

        self._init_shared_bottom_cross(architect_argv)

        init_model_args = config, dataset_argv, architect_argv, init_argv, ptmzr_argv, reg_argv, autodis_argv, \
            log_file, loss_mode, merge_multi_hot, batch_norm, use_inter, use_bridge, use_autodis, debug_argv, \
            checkpoint, sess, feat_field_map, hcmd_argv, domain_list, index_list_feat_id, list_to_domain_map

        self._init_model(init_model_args)
        
        print(self.domain_list)

    def _init_shared_bottom_cross(self, architect_argv):
        self.domain_w = {}
        self.domain_b = {}
        self.shared_w = []
        self.shared_b = []
        self.out_w = {}
        self.out_b = {}
        self.cross_w = {}
        self.cross_b = {}
        self.domain_layers_shape = []
        self.h_ax_w = []
        self.h_ax_b = []
        self.embed_v = None
        self.shared_layer_shape = []
        self.shared_layer = None
        self.embedding_size, self.num_cross_layer, self.deep_layers, \
        self.auxiliary_network_layer, self.cross_combine, self.cross_domain, \
        self.domain_loss_weight = architect_argv[:7]

        print(f"[SharedBottomCross] {architect_argv}, {self.num_cross_layer}")

        # weight of l2 norm on mlp weights
        self.lambda_mlp_l2 = 0.0
        if len(architect_argv) > 7:
            self.lambda_mlp_l2 = architect_argv[7]

        '''
        use_star_param_sharing: str 
        - `None` (default ): do not use parameter sharing like STAR
        - `"multiply"`: multiply the weights -- same as STAR 
        - `"add"`: add the weights
        '''
        self.use_star_param_sharing = None
        if len(architect_argv) > 8:
            self.use_star_param_sharing = architect_argv[8]

        # weight of l2 norm on shared mlp weights
        self.lambda_mlp_shared_l2 = 0.0
        if self.use_star_param_sharing is not None and len(architect_argv) > 9:
            self.lambda_mlp_shared_l2 = architect_argv[9]

        self.random_domain_index_range = 0
        if len(architect_argv) > 10:
            self.random_domain_index_range = architect_argv[10]

    def _init_aux(self):
        # new code: initialize weights for the cross module 
        self.init_aux_weights()
        if self.num_cross_layer > 0:
            if self.cross_domain:
                self.cross_variable_part_domains()
            else:
                self.cross_variable_part()

        if self.num_cross_layer > 0:
            self.final_variable_part(self.embedding_dim + self.deep_layers[-1])

        # end initialization of cross and aux variables
    
    def get_all_domain_loss_sum(self, domain_out, lbl_hldr, id_hldr, _lambda=0.0001):
        if self.domain_loss_weight is None:
            return super().get_all_domain_loss_sum(domain_out, lbl_hldr, id_hldr, _lambda)
            
        n_domains = len(self.domain_list)
        if self.domain_loss_weight[0] == "custom":
            weights = self.domain_loss_weight[1:]

        elif self.domain_loss_weight[0] == "random":
            weights = tf.nn.softmax(tf.random.normal([n_domains]))
        
        elif self.domain_loss_weight[0] == 'cagrad' and len(self.domain_loss_weight) == 2 + n_domains:
            weights = self.domain_loss_weight[2:]
        
        else:
            weights = [1.0 / n_domains for _ in range(n_domains)]

        loss = _lambda * tf.nn.l2_loss(self.embed_v)
        loss_list = self.get_all_domain_loss(domain_out, lbl_hldr, id_hldr)
        for i, l in enumerate(loss_list):
            loss += l * weights[i] 
        
        print("[SharedBottomCross] Weights for domain losses:", weights)
        if self.domain_loss_weight[0] == "cagrad":
            var_list = tf.trainable_variables()
            r = tf.stop_gradient(
                    tf.reduce_sum([tf.norm(g) for g in tf.gradients(loss, var_list) if g is not None]) / 
                    tf.reduce_sum([tf.norm(g) for g in tf.gradients(loss_list[0], var_list) if g is not None]))
            c = self.domain_loss_weight[1]

            loss += r * loss_list[0] * c

        print(f"[SharedBottomCross] lambda_mlp_l2 {self.lambda_mlp_l2}, "\
              f"lambda_mlp_shared_l2 {self.lambda_mlp_shared_l2}, "\
              f"use_star_param_sharing {self.use_star_param_sharing}")
        if self.lambda_mlp_l2 > 0:
            for _, domain_w in self.domain_w.items():
                for w in domain_w:
                    loss += self.lambda_mlp_l2 * tf.nn.l2_loss(w)

        if self.use_star_param_sharing is not None and self.lambda_mlp_shared_l2 > 0:
            for w in self.shared_w:
                loss += self.lambda_mlp_shared_l2 * tf.nn.l2_loss(w)

        return loss

                
    def auxiliary_network(self, input_4, training=False):
        if len(self.h_ax_w) == 0:
            return None

        hidden_output = input_4
        for i in range(len(self.h_ax_w)):
            hidden_output = tf.matmul(hidden_output, self.h_ax_w[i]) + self.h_ax_b[i]
            if i < len(self.h_ax_w) - 1:
                hidden_output = activate('relu', hidden_output)
                if training:
                    hidden_output = tf.nn.dropout(hidden_output, keep_prob=self.keep_prob)
        return hidden_output

    def star_param_sharing_part(self):
        """
        when STAR type parameter sharing is enabled, weights/biases of mlps are
        the products/sums of the shared weights/biases and the domain-specific
        weights/biases
        """
        domain_w = {}
        domain_b = {}

        print(f"[SharedBottomCross] construct parameters for each domain using {self.use_star_param_sharing}")
        for domain_idx in self.domain_list:
            domain_w[domain_idx] = []
            domain_b[domain_idx] = []
            try:
                for i in range(len(self.domain_w[domain_idx])):
                    print(f"domain {domain_idx}, " \
                          f"layer {i}, {self.shared_w[i].shape}, " \
                          f"{self.domain_w[domain_idx][i].shape}")
                    if self.use_star_param_sharing == "multiply":
                        domain_w[domain_idx].append(tf.multiply(self.domain_w[domain_idx][i], self.shared_w[i]))
                    else:
                        domain_w[domain_idx].append(tf.add(self.domain_w[domain_idx][i], self.shared_w[i]))

                    domain_b[domain_idx].append(tf.add(self.domain_b[domain_idx][i], self.shared_b[i]))

            except KeyError:
                print('error')

        return domain_w, domain_b

    def deep_and_cross_network(self, domain_concat_embed, domain_w, domain_b, is_training, idx):
        if self.num_cross_layer > 0 and self.cross_combine == 'parallel':
            if self.cross_domain:
                domain_cross = self.cross_layer_domain(domain_concat_embed, self.num_cross_layer, domain_idx=idx)
            else:
                domain_cross = self.cross_layer(domain_concat_embed, self.num_cross_layer)

            try:
                domain_hidden = self.mlp(domain_concat_embed, domain_w[idx][:-1], domain_b[idx][:-1],
                                         is_training=is_training,
                                         scope='domain_%s' % idx, is_output=False)
            except KeyError:
                print('error')

            domain_cat = tf.concat([domain_cross, domain_hidden], 1)
            print('[SharedBottomCross], cross_combine: {}, domain: {}, domain_cross: {}, domain_cat: {}'.format(
                self.cross_combine, idx, domain_cross.shape, domain_cat.shape))
            try:
                hidden_output = tf.matmul(tf.nn.leaky_relu(domain_cat), self.out_w[idx]) + self.out_b[idx]
            except KeyError:
                print('error')

            domain_sm = tf.reshape(hidden_output, [-1, 1])

        else:
            try:
                domain_sm = self.mlp(domain_concat_embed, domain_w[idx], domain_b[idx],
                                     is_training=is_training,
                                     scope='domain_%s' % idx, is_output=True)
            except KeyError:
                print('error')

        return domain_sm

    def multi_domain_forward(self, wt_hldr, id_hldr, merge_multi_hot, is_training=False):
        domain_out = {}
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
            domain_concat_embed = tf.boolean_mask(shared_out, domain_mask)

            domain_sm = self.deep_and_cross_network(domain_concat_embed, domain_w, domain_b, is_training, idx)

            domain_sa = self.auxiliary_network(domain_concat_embed, is_training)
            if domain_sa is not None:
                print("[SharedBottomCross]: domain_sa: {}, domain_sm: {}".format(domain_sa.shape, domain_sm.shape))
                domain_predict = tf.add(domain_sm, domain_sa)

            else:
                print("[SharedBottomCross]: domain_sm: {}".format(domain_sm.shape))
                domain_predict = domain_sm

            domain_out[idx] = tf.reshape(domain_predict, [-1, ])


        return domain_out


    def init_aux_weights(self):
        # --------------------------

        if self.auxiliary_network_layer is None or len(self.auxiliary_network_layer) == 0:
            print("[SharedBottomCross] empty auxiliary_network_layer {}".format(self.auxiliary_network_layer))
            return

        self.auxiliary_network_layer = [self.embedding_dim] + self.auxiliary_network_layer + [1]

        print("[SharedBottomCross]: initialize aux network {}".format(self.auxiliary_network_layer))

        init_acts = []
        for i in range(len(self.auxiliary_network_layer) - 1):
            init_acts.extend([
                ('h{}_ax_w'.format(i + 1), self.auxiliary_network_layer[i: i + 2], 'random'),
                ('h{}_ax_b'.format(i + 1), [self.auxiliary_network_layer[i + 1]], 'random')
            ])
        var_map, log = init_var_map(self.init_argv, init_acts)
        self.log += log
        with tf.variable_scope("ax_w_b"):
            for i in range(len(self.auxiliary_network_layer) - 1):
                self.h_ax_w.append(tf.Variable(var_map['h{}_ax_w'.format(i + 1)], name='h{}_ax_w'.format(i + 1)))
                self.h_ax_b.append(tf.Variable(var_map['h{}_ax_b'.format(i + 1)], name='h{}_ax_b'.format(i + 1)))

    def cross_variable_part(self):
        init_acts = [('cross_w', [self.num_cross_layer, self.embedding_dim], 'random'),
                     ('cross_b', [self.num_cross_layer, self.embedding_dim], 'random'), ]

        var_map, log = init_var_map(self.init_argv, init_acts)

        self.log += log
        self.cross_w = tf.Variable(var_map['cross_w'])
        self.cross_b = tf.Variable(var_map['cross_b'])

    def cross_layer(self, cross_input, num_cross_layer):
        '''
            cross operator shared by all the domains
            used parameters self.cross_w and self.cross_b
        '''
        # embedding layer
        x_0 = cross_input
        # cross layer
        x_l = x_0
        for i in range(num_cross_layer):
            try:
                xlw = tf.tensordot(x_l, self.cross_w[i], axes=1)
            except KeyError:
                print('error')
            try:
                x_l = x_0 * tf.expand_dims(xlw, -1) + self.cross_b[i] + x_l
            except KeyError:
                print('error')
            x_l.set_shape((None, self.embedding_dim))

        return x_l


    def cross_variable_part_domains(self):
        print("[SharedBottomCross]cross_variable_part_domains")
        init_acts = []
        for idx in self.domain_list:
            init_acts.extend([
                ('cross_w_{}'.format(idx), [self.num_cross_layer, self.embedding_dim], 'random'),
                ('cross_b_{}'.format(idx), [self.num_cross_layer, self.embedding_dim], 'random')])

        var_map, log = init_var_map(self.init_argv, init_acts)
        print(log)
        self.log += log

        self.cross_w = {}
        self.cross_b = {}
        for idx in self.domain_list:
            self.cross_w[idx] = tf.Variable(var_map['cross_w_{}'.format(idx)])
            self.cross_b[idx] = tf.Variable(var_map['cross_b_{}'.format(idx)])


    def cross_layer_domain(self, cross_input, num_cross_layer, domain_idx):
        '''
            cross operator shared by all the domains
            used parameters self.cross_w and self.cross_b
        '''

        # embedding layer
        x_0 = cross_input
        # cross layer
        x_l = x_0
        for i in range(num_cross_layer):
            try:
                xlw = tf.tensordot(x_l, self.cross_w[domain_idx][i], axes=1)
            except KeyError:
                print('error')
            try:
                x_l = x_0 * tf.expand_dims(xlw, -1) + self.cross_b[domain_idx][i] + x_l
            except KeyError:
                print('error')
            x_l.set_shape((None, self.embedding_dim))

        return x_l

    def final_variable_part(self, dim_input):

        init_acts_final = []
        for domain_idx in self.domain_list:
            init_acts_final.extend(
                    [('out_w_{}'.format(domain_idx), [dim_input, 1], 'random'),
                     ('out_b_{}'.format(domain_idx), [1], 'zero')])

        var_map, log = init_var_map(self.init_argv, init_acts_final)
        self.log += log

        self.out_w = {}
        self.out_b = {}
        for domain_idx in self.domain_list:
            self.out_w[domain_idx] = tf.Variable(var_map['out_w_{}'.format(domain_idx)])
            self.out_b[domain_idx] = tf.Variable(var_map['out_b_{}'.format(domain_idx)])

    def init_weights(self, input_dim, embedding_size):
        ## new code:
        if self.num_cross_layer > 0 and self.cross_combine == 'stack_concat':
            # bottom layer is extended for extra input from the cross layers
            self.domain_layers_shape = [self.embedding_dim * 2] + self.deep_layers + [1]  # final pred node
        else:
            self.domain_layers_shape = [self.embedding_dim] + self.deep_layers + [1]  # final pred node
        
        self._init_weights(input_dim, embedding_size)


    def init_mlp_weights(self, var_map):
        self.shared_w = []
        self.shared_b = []
        if self.use_star_param_sharing:
            for i in range(len(self.domain_layers_shape) - 1):
                self.shared_w.append(tf.Variable(var_map['s_h{}_w'.format(i + 1)],
                                                    name='s_h{}_w'.format(i + 1)))
                self.shared_b.append(tf.Variable(var_map['s_h{}_b'.format(i + 1)],
                                                    name='s_h{}_b'.format(i + 1)))
                                        
        self.domain_w = {}
        self.domain_b = {}
        for domain_idx in self.domain_list:
            each_domain_w = []
            each_domain_b = []
            for i in range(len(self.domain_layers_shape) - 1):
                each_domain_w.append( 
                    tf.Variable(var_map['d_{}_h{}_w'.format(domain_idx, i + 1)],
                                   name='d_{}_h{}_w'.format(domain_idx, i + 1)))

                each_domain_b.append(
                    tf.Variable(var_map['d_{}_h{}_b'.format(domain_idx, i + 1)],
                                   name='d_{}_h{}_b'.format(domain_idx, i + 1)))

            self.domain_w[domain_idx] = each_domain_w
            self.domain_b[domain_idx] = each_domain_b


        self.shared_layer = tf.Variable(var_map['shared_layer'], name='shared_layer')

    def _init_weights(self, input_dim, embedding_size):
        init_acts = []

        for j in range(len(self.domain_layers_shape) - 1):
            init_acts.extend([
                ('s_h{}_w'.format(j + 1), self.domain_layers_shape[j: j + 2], 'random'),
                ('s_h{}_b'.format(j + 1), [self.domain_layers_shape[j + 1]], 'zeros')
            ])

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