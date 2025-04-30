# encoding=utf-8
from __future__ import print_function
import json 
import tensorflow as tf

from train.models.tf_util import init_var_map, activate, get_domain_mask_noah, split_mask, split_param, sum_multi_hot
from models.multi_scenario_base import MultiScenarioBase
from models.DCN_SFPS import DCN_SFPS
from train.models.DIN_UTILS import split_group_feat, split_one_and_multi_hot_feat, get_item_his_emb, \
    get_item_target_emb, din_attention, Din_attention_params
from common_util.util import get_spec_feature_embed_flags


class SARNet(MultiScenarioBase, object):

    def __init__(self, config, dataset_argv, architect_argv, init_argv,
                 ptmzr_argv, reg_argv, distilled_argv,
                 loss_mode='full', merge_multi_hot=False,
                 batch_norm=True, distill=False,
                 checkpoint=None, sess=None, regression_task=False,
                 use_cross=True, use_linear=False, use_fm=False):
        super(SARNet, self).__init__()
        self.specific_expert_w = None 
        self.specific_expert_b = None 
        self.specific_embed_linear = None 
        self.specific_transform_layer_w = None 
        self.specific_transform_layer_b = None 
        # ---------
        self.data_member_part(config, distilled_argv, distill, init_argv, dataset_argv, architect_argv, reg_argv,
                              checkpoint, sess, ptmzr_argv, merge_multi_hot, regression_task, use_cross, use_linear,
                              use_fm, batch_norm)
        
        self.set_position_and_filter_config()
        self.sfps_init_func()

        self.init_placeholder_noah()

        if self.fields_num != 0:
            self.compute_embedding_dim()
            self.sparse_variable_part()

        self.init_weights_sar()
        self.dnn_variable_part()

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

        print("SAR model init finish")
        
    def data_member_part(self, config, distilled_argv, distill, init_argv, dataset_argv, architect_argv,
                         reg_argv, checkpoint, sess, ptmzr_argv, merge_multi_hot, regression_task, use_cross,
                         use_linear,
                         use_fm, batch_norm):
        self.config = config
        self.T, self.loss_lambda = distilled_argv
        self.distill = distill
        self.init_argv = init_argv
        (features_dim, fields_num, dense_num, multi_hot_flags, multi_hot_len, multi_hot_variable_len) = dataset_argv
        self.one_hot_flags = [not flag for flag in multi_hot_flags]
        embedding_size, num_cross_layer, deep_layers, act_func = architect_argv
        self.keep_prob, _lambda, l1_lambda = reg_argv
        self._lambda = _lambda
        self.l1_lambda = l1_lambda
        self.embedding_size = embedding_size
        self.num_cross_layer = num_cross_layer
        self.ptmzr_argv = ptmzr_argv
        self.merge_multi_hot = merge_multi_hot
        self.regression_task = regression_task
        self.deep_layers = deep_layers
        self.act_func = act_func
        self.features_dim = features_dim
        self.fields_num = fields_num
        self.dense_num = dense_num
        self.multi_hot_len = multi_hot_len
        self.multi_hot_flags = multi_hot_flags
        self.multi_hot_variable_len = multi_hot_variable_len
        self.checkpoint = checkpoint
        self.sess = sess
        
        self.use_cross = use_cross
        self.use_linear = use_linear
        self.use_fm = use_fm
        self.batch_norm = batch_norm
        self.use_sfps = config.USE_SFPS
        
        # multi-domain arg
        self.domain_dict = config.domain_dict
        self.domain_col_idx = config.domain_col_idx

        # dense parameters
        self.use_dense_features = config.USE_DENSE_FEATURES
        self.log_raw_dense = config.LOG_RAW_DENSE 

        # DIN parameters
        self.merge_method = config.MERGE_METHOD 
        self.din_act_func = config.DIN_ACT_FUNC
        self.din_hidden_layers = config.DIN_HIDDEN_LAYERS
        self.din_softmax_norm = config.DIN_SOFTMAX_NORM 
        self.target_feat_idx = config.TARGET_FEAT_IDX
        self.seq_group_size_list = config.SEQ_GROUP_SIZE_LIST
        self.seq_group_merge_method_list = config.SEQ_GROUP_MERGE_METHOD_LIST
        
        # SAR parameters
        self.use_ds_gate = config.USE_DS_GATE
        self.num_shared_experts = config.NUM_SHARED_EXPERTS
        self.use_ds_batch_norm = config.USE_DS_BATCH_NORM 
        self.shared_expert_hidden_layers = config.SHARED_EXPERT_HIDDEN_LAYERS
        self.shared_gate_hidden_layers = config.SHARED_GATE_HIDDEN_LAYERS
        self.shared_gate_act_func = config.SHARED_GATE_ACT_FUNC
        
        log_dict = locals()
        
        log_dict['domain_dict'] = self.domain_dict
        log_dict['domain_col_idx'] = self.domain_col_idx 
        log_dict['split_sub_network'] = config.split_sub_network
        
        log_dict['use_dense_features'] = self.use_dense_features
        log_dict['log_raw_dense'] = self.log_raw_dense 
        
        log_dict['merge_method'] = self.merge_method
        log_dict['din_act_func'] = self.din_act_func
        log_dict['din_softmax_norm'] = self.din_softmax_norm
        log_dict['din_hidden_layers'] = self.din_hidden_layers 
        log_dict['target_feat_idx'] = self.target_feat_idx 
        log_dict['seq_group_size_list'] = self.seq_group_size_list
        log_dict['seq_group_merge_method_list'] = self.seq_group_merge_method_list
        
        log_dict['use_ds_gate'] = self.use_ds_gate 
        log_dict['num_shared_experts'] = self.num_shared_experts
        log_dict['use_ds_batch_norm'] = self.use_ds_batch_norm 
        log_dict['shared_expert_hidden_layers'] = self.shared_expert_hidden_layers
        log_dict['shared_gate_hidden_layers'] = self.shared_gate_hidden_layers
        log_dict['shared_gate_act_func'] = self.shared_gate_act_func
        
        
        self.log = json.dumps(log_dict, default=str, sort_keys=True, indent=4)

    def eval_part(self):
        if not self.config.split_sub_network:
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
    
    def sparse_variable_part(self):
        init_acts = [('embed', [self.features_dim, self.embedding_size], 'random')]
        if self.config.POSITION_EMBEDDING:
            init_acts.append(('posi_embed', [self.config.MAX_POSI_SIZE, self.embedding_size], 'random'))
        var_map, log = init_var_map(self.init_argv, init_acts)

        self.log += log
        self.embed_v = tf.Variable(var_map['embed'])
        if self.config.POSITION_EMBEDDING:
            self.posi_embed = tf.Variable(var_map['posi_embed'])
    
    def init_weights_sar(self):
        # specific experts 
        self.init_specific_experts()
        # shared experts pass
        # transform layer
        if self.use_ds_gate:
            self.init_transform_layer()
    
    def init_specific_experts(self):
        self.specific_expert_w = []
        self.specific_expert_b = []
        if self.use_linear:
            self.specific_embed_linear = []
        for domain_idx in self.domain_dict: 
                   
            init_acts = [(f'cross_w_{domain_idx}', [self.num_cross_layer, self.embedding_dim], 'random'),
                        (f'cross_b_{domain_idx}', [self.num_cross_layer, self.embedding_dim], 'random')]
            if self.use_linear:
                init_acts.append((f'embed_linear_{domain_idx}', [self.features_dim, 1], 'random'))
            var_map, log = init_var_map(self.init_argv, init_acts)

            self.log += log
            each_domain_cross_w = tf.Variable(var_map[f'cross_w_{domain_idx}'])
            each_domain_cross_b = tf.Variable(var_map[f'cross_b_{domain_idx}'])
            self.specific_expert_w.append(each_domain_cross_w)
            self.specific_expert_b.append(each_domain_cross_b)
            if self.use_linear:
                each_domain_embed_linear = tf.Variable(var_map[f'embed_linear_{domain_idx}'], validate_shape=False)
                self.specific_embed_linear.append(each_domain_embed_linear)
                
    def init_transform_layer(self):
        self.specific_transform_layer_w = []
        self.specific_transform_layer_b = []
        for domain_idx in self.domain_dict: 
            init_acts = [(f'trans_w_{domain_idx}', [1, self.embedding_dim], 'one'),
                        (f'trans_b_{domain_idx}', [1, self.embedding_dim], 'zero')]
            var_map, log = init_var_map(self.init_argv, init_acts)

            self.log += log
            each_domain_trans_w = tf.Variable(var_map[f'trans_w_{domain_idx}'])
            each_domain_trans_b = tf.Variable(var_map[f'trans_b_{domain_idx}'])
            self.specific_transform_layer_w.append(each_domain_trans_w)
            self.specific_transform_layer_b.append(each_domain_trans_b)

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
        vx_embed = self.construct_embedding(wt_hldr, id_hldr, self.merge_multi_hot, is_training=is_training)
        vx_embed = tf.reshape(vx_embed, [-1, self.embedding_dim])
        for idx in self.domain_dict:
            print(f'This is domain_{idx}!')
            domain_mask = get_domain_mask_noah(self.domain_dict.get(idx), domain_hldr)
            domain_embed = tf.boolean_mask(vx_embed, domain_mask)
            domain_label = tf.boolean_mask(self.lbl_hldr, domain_mask) if is_training else None
            domain_predict = self.sar_sub_netword_forward(domain_embed, idx, is_training)
            predict_dict[idx] = domain_predict
            label_dict[idx] = domain_label
        return predict_dict, label_dict
    
    def multi_domain_forward_split(self, wt_hldrs, id_hldrs, is_training=False, is_save=False):
        # eval / save
        predict_dict, label_dict = {}, {}
        for idx in self.domain_dict:
            print(f'This is domain_{idx}!')
            domain_embed = self.construct_embedding(wt_hldrs[idx], id_hldrs[idx], self.merge_multi_hot,
                                                    is_training=is_training, is_save=is_save)
            domain_embed = tf.reshape(domain_embed, [-1, self.embedding_dim])
            domain_predict = self.sar_sub_netword_forward(domain_embed, idx, is_training)
            predict_dict[idx] = domain_predict
        return predict_dict, label_dict

    def sar_sub_netword_forward(self, domain_embed, idx, is_training):
        # domain-specific transform layer
        if self.use_ds_gate: 
            domain_embed = self.domain_transform_layer(domain_embed, idx, is_training)
        # domain-specific experts
        specific_embed = self.domain_specific_expert_layer(domain_embed, idx, is_training, self.num_cross_layer)
        # shared experts 
        shared_embed = self.shared_expert_layer(domain_embed, idx, is_training, scope='shared')
        # domain final layer
        print(f'specific_embed.shape: {specific_embed.shape}')
        print(f'shared_embed.shape: {shared_embed.shape}')
        domain_predict = self.domain_final_forward(tf.concat([specific_embed, shared_embed], axis=-1), idx, 
                                                   batch_norm=self.batch_norm, 
                                                   training=is_training, scope=f'final_forward_{idx}')

        return domain_predict
    
    def domain_transform_layer(self, domain_embed, idx, is_training):
        # the shape of domain_embed: [B, D]
        domain_trans_w = self.specific_transform_layer_w[idx] # [1, D]
        domain_trans_b = self.specific_transform_layer_b[idx] # [1, D]
        return tf.multiply(domain_embed, domain_trans_w) + domain_trans_b
    
    def domain_specific_expert_layer(self, domain_embed, idx, is_training, num_cross_layer):
        domain_cross_w = self.specific_expert_w[idx]
        domain_cross_b = self.specific_expert_b[idx]
        # embedding layer
        x_0 = domain_embed
        # cross layer
        x_l = x_0
        for i in range(num_cross_layer):
            xlw = tf.tensordot(x_l, domain_cross_w[i], axes=1)
            x_l = x_0 * tf.expand_dims(xlw, -1) + domain_cross_b[i] + x_l
            x_l.set_shape((None, self.embedding_dim))
        return x_l
    
    def shared_expert_layer(self, domain_embed, idx, is_training, scope):
        shared_expert_output = self.expert_forward(
            idx, 
            domain_embed,
            self.num_shared_experts, 
            batch_norm=self.batch_norm,
            training=is_training,
            reuse=tf.AUTO_REUSE, 
            scope=f'{scope}_expert'
        ) # [B, N_E, H_E]
        gate_output = self.gate_forward(
            idx, 
            domain_embed, 
            self.shared_gate_act_func,
            batch_norm=self.batch_norm,
            training=is_training, 
            reuse=tf.AUTO_REUSE, 
            scope=f'{scope}_gate'
        ) # [B, N_E]
        gate_output = tf.expand_dims(gate_output, axis=1) # [B, N_E] -> [B, 1, N_E]
        shared_expert_output = tf.squeeze(tf.matmul(gate_output, shared_expert_output), 
                                          axis=-2) # [B, 1, H_E] -> [B, H_E]        
        return shared_expert_output 
        
    def expert_forward(self, domain_idx, nn_input, num_experts, batch_norm, training, reuse, scope="shared_expert"):
        with tf.variable_scope(scope, reuse=reuse):
            shared_expert_output_list = []
            for i in range(num_experts):
                shared_expert_output_list.append(
                    self.ds_mlp_module(nn_input, 
                                       hidden_layers=self.shared_expert_hidden_layers,
                                       act_func='relu', 
                                       dropout_prob=(1.0 - self.keep_prob),
                                       batch_norm=batch_norm, 
                                       training=training, 
                                       reuse=reuse,
                                       name_scope=f'expert_{i}_mlp',
                                       domain_scope=f'domain_{domain_idx}', 
                                       ds_batch_norm=self.use_ds_batch_norm) # [B, H_E]
                ) 
            shared_expert_output = tf.stack(shared_expert_output_list, axis=-2) # [B, N_E, H_E]
        return shared_expert_output
        
    def gate_forward(self, domain_idx, nn_input, act_func, batch_norm, training, reuse, scope='shared_gate'):
        if len(self.shared_gate_hidden_layers) >= 2:
            nn_input = self.ds_mlp_module(
                nn_input=nn_input, 
                hidden_layers=self.shared_gate_hidden_layers[:-1],
                act_func='relu', 
                dropout_prob=(1.0 - self.keep_prob),
                batch_norm=batch_norm, 
                training=training, 
                reuse=reuse,
                name_scope=f"{scope}_1",
                domain_scope=f'domain_{domain_idx}',
                ds_batch_norm=self.use_ds_batch_norm
            )
    
        gate_output = self.ds_mlp_module(
            nn_input=nn_input, 
            hidden_layers=self.shared_gate_hidden_layers[-1:],
            act_func=act_func, 
            dropout_prob=0.0,
            batch_norm=False, 
            training=training, 
            reuse=reuse,
            name_scope=f"{scope}_2", 
            domain_scope=f'domain_{domain_idx}',
            ds_batch_norm=self.use_ds_batch_norm
        )
        return gate_output
    
    def domain_final_forward(self, nn_input, domain_idx, batch_norm, training, scope="final_forward"):
        all_deep_layers = [nn_input.shape.as_list()[-1]] + self.deep_layers
        with tf.variable_scope(scope, reuse=(not training)):
            final_output = self.ds_mlp_module(nn_input, hidden_layers=all_deep_layers,
                                              act_func='relu', dropout_prob=(1.0 - self.keep_prob),
                                              batch_norm=batch_norm, training=training, reuse=(not training),
                                              name_scope=f'{domain_idx}_final_forward_mlp',
                                              domain_scope=f'domain_{domain_idx}', 
                                              ds_batch_norm=False) # [B, H_E]
            
            final_output = self.ds_mlp_module(final_output, hidden_layers=[1],
                                              act_func=None, dropout_prob=0.0,
                                              batch_norm=False, training=training, reuse=(not training),
                                              name_scope=f'{domain_idx}_final_forward_logits',
                                              domain_scope=f'domain_{domain_idx}', 
                                              ds_batch_norm=False) # [B, 1]
            
            final_output = tf.reshape(final_output, [-1]) # [B]
        return final_output

    def get_loss(self, train_y, label, _lambda, is_domain=False):
        all_sample_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=train_y, labels=label)
        loss = tf.reduce_mean(all_sample_loss, keep_dims=is_domain, name='loss')
        loss = loss + _lambda * tf.nn.l2_loss(self.embed_v)
        return loss
              
    def ds_mlp_module(self, nn_input, hidden_layers, act_func='relu', dropout_prob=0.1, 
                      batch_norm=False, inf_dropout=False, training=True, reuse=True, \
                      name_scope='mlp_module', domain_scope='domain_i', ds_batch_norm=False):

        with tf.variable_scope(name_scope, reuse=reuse):
            hidden_output = nn_input
            for i, layer in enumerate(hidden_layers):
                hidden_output = tf.layers.dense(hidden_output, layer, activation=act_func,
                                                use_bias=True, name=f'mlp_{i}', reuse=reuse,
                                                kernel_initializer=tf.random_uniform_initializer(
                                                        minval=-0.001, 
                                                        maxval=0.001
                                                    )   
                                                )
                print(f'========= dense layer : {hidden_output.name} ==========')
                print(f'hidden_output shape: {hidden_output.shape}')
                if batch_norm:
                    hidden_output = tf.layers.batch_normalization(
                        hidden_output, training=training, reuse=reuse, 
                        name=(f'bn_{i}_{domain_scope}' if ds_batch_norm else f'bn_{i}')
                    )
                    print(f'========= batch_norm : {hidden_output.name} ==========')
                    
                if training or inf_dropout:
                    # dropout at inference time only if inf_dropout is True
                    print(f"""apply dropout in *{'training' if training else 'inference'}* time,
                          inf_dropout: {inf_dropout}""")
                    hidden_output = tf.nn.dropout(hidden_output, rate=dropout_prob)
                    
            return hidden_output
        
        
    def construct_embedding(self, wt_hldr, id_hldr, merge_multi_hot=None, is_training=True, is_save=False):
        # construct the embedding layer
        print(f'construct_embedding use_sfps: {self.use_sfps}')
        if not self.use_sfps:
            if merge_multi_hot and self.num_multihot > 0:
                if self.merge_method == 'sum':
                    vx_embed = self.sum_merge(wt_hldr, id_hldr)
                elif self.merge_method == 'din':
                    vx_embed = self.din_merge(self.target_feat_idx, wt_hldr, id_hldr, is_training=is_training) 
                else:
                    raise ValueError(f'merge_method [{self.merge_method}] is invalid.')
            else:
                mask = tf.expand_dims(wt_hldr, 2)
                vx_embed = tf.multiply(tf.gather(self.embed_v, id_hldr), mask)
            
            return vx_embed
        else:
            return super().construct_embedding(wt_hldr, id_hldr,
                                               merge_multi_hot, is_training, is_save)

    def sum_merge(self, wt_hldr, id_hldr):
        mask = tf.expand_dims(wt_hldr, 2) # [B, M] -> [B, M, 1]
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

        one_hot_v, multi_hot_v = split_param(self.embed_v, id_hldr, self.multi_hot_flags)
        if self.config.POSITION_EMBEDDING:
            d_position_embed = tf.gather(self.posi_embed, d_position_value)
            multi_hot_v = multi_hot_v + tf.expand_dims(d_position_embed, 0)

            # fm part (reduce multi-hot vector's length to k*1)
        if self.config.DYNAMIC_LENGTH:
            multi_hot_vx = sum_multi_hot(multi_hot_v, multi_hot_mask, self.multi_hot_variable_len)
        else:
            multi_hot_vx = sum_multi_hot(multi_hot_v, multi_hot_mask, self.num_multihot)

        one_hot_vx = tf.multiply(one_hot_v, one_hot_mask)
        vx_embed = tf.concat([one_hot_vx, multi_hot_vx], axis=1)
        vx_embed = tf.reshape(vx_embed, [-1, (self.embedding_dim - self.dense_num)]) # [B, M * D]
        return vx_embed

    def din_merge(self, target_feat_idx, wt_hldr, id_hldr, is_training=True):
        
        one_hot_id, multi_hot_id = split_one_and_multi_hot_feat(id_hldr, self.multi_hot_flags)
        one_hot_wt, multi_hot_wt = split_one_and_multi_hot_feat(wt_hldr, self.multi_hot_flags)
        
        # the shape of group_id_list {[[B, l1], ..., [B, l1]], ..., [[B, L], ..., [B, L]]}
        group_wt_list = split_group_feat(multi_hot_wt, self.multi_hot_variable_len, self.seq_group_size_list) 
        group_mask_list = [] # {[B, l1], ..., [B, L]}
        for group_wt in group_wt_list:
            group_pos = tf.expand_dims(tf.range(group_wt[0].shape[-1]), axis=0) # [1, L]
            group_len = tf.cast(tf.reduce_sum(group_wt[0], axis=-1, keep_dims=True), 
                                dtype=group_pos.dtype) # [B, 1]
            no_padding_mask = group_pos < group_len
            group_mask = tf.where(no_padding_mask, tf.ones_like(no_padding_mask), tf.zeros_like(no_padding_mask))
            group_mask_list.append(group_mask)
        group_wt_list = [tf.stack(g, axis=1) for g in group_wt_list] # {[B, m1, l1], ..., [B, M, L]}          
            
        # the shape of group_id_list {[[B, l1], ..., [B, l1]], ..., [[B, L], ..., [B, L]]}
        group_id_list = split_group_feat(multi_hot_id, self.multi_hot_variable_len, self.seq_group_size_list) 
        group_id_list = [tf.stack(g, axis=1) for g in group_id_list] # {[B, m1, l1], ..., [B, M, L]}
        
        target_wt = tf.gather(wt_hldr, indices=target_feat_idx, axis=1)
        target_id = tf.gather(id_hldr, indices=target_feat_idx, axis=1)
        target_emb = get_item_target_emb(self.embed_v, target_id, target_wt) # [B, M * D]
        
        merge_embs = []
        for i, _ in enumerate(group_wt_list):
            item_his_emb = get_item_his_emb(self.embed_v, group_id_list[i], group_wt_list[i]) # [B, L, M * D]
            merge_embs.append(
                din_attention(
                    Din_attention_params(
                    target_emb, item_his_emb, None, group_mask_list[i],
                    hidden_layers=self.din_hidden_layers, act_func=self.din_act_func,
                    stag=f'seq_{i}', softmax_stag=self.din_softmax_norm, 
                    merge_method=self.seq_group_merge_method_list[i],
                    scope=f'din_merge_{i}', reuse=(not is_training))
                )
            ) # [B, M * D]

        one_hot_v = tf.gather(self.embed_v, one_hot_id) # [B, M_s, D]
        one_hot_vx = tf.multiply(one_hot_v, tf.expand_dims(one_hot_wt, axis=-1)) # [B, M_s, D]
        one_hot_vx = tf.reshape(one_hot_vx, shape=[-1, sum(self.one_hot_flags) * self.embedding_size]) # [B, M_s * D]
        
        vx_embed = tf.concat([one_hot_vx, tf.concat(merge_embs, axis=1)], axis=1)
        
        return vx_embed
        
    
