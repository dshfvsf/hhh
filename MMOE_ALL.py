from __future__ import print_function

import pickle
import json
import operator
from functools import reduce
from train.models.tf_util import init_var_map, get_params_count, split_mask, split_param, sum_multi_hot
from train.models.utils.grad_optm import mgda_alg, \
    pcgrad_retrieve_grad, pcgrad_flatten_grad, pcgrad_project_conflicting, pcgrad_unflatten_grad, \
    cagrad_algo
from train.layer.layers import CapeLayer, ResConvBlock
import tensorflow as tf
import numpy as np
try:
    import horovod.tensorflow as hvd
    distribute_tmp = True
except Exception:
    print("have no horovod package")
    distribute_tmp = False
from models.MMOE import MMOE


class MMOE_ALL(MMOE):

    def __init__(self, config, dataset_argv, embedding_size, expert_argv, tower_argv,
                 init_argv, ptmzr_argv, reg_argv, dense_flags, regression_task=False,
                 trans_argv=None, u_init_argv=None, loss_mode='full', merge_multi_hot=False, batch_norm=True):
        self.cross_w = []
        self.cross_b = []
        self.out_w = []
        self.out_b = []
        self.h_w = []
        self.h_b = []
        self.embed_v = None
        self.lbl_hldr = None
        self.keep_prob = None
        self._lambda = None
        self.l1_lambda = None
        self.all_tower_deep_layers = None
        self.fields_num = None
        
        self.experts = None
        self.gates = []
        self.num_tasks = None
        self.has_task_mask = None
        self.num_experts = None
        self.uncertainties = []
        self.uwl_task_num = None
        
        self.num_multihot = None
        self.embedding_dim = None
        self.num_cross_layer = None
        self.tower_act_func = None
        self.features_size = None
        self.multi_hot_flags = None
        self.expert_hidden_layers = None
        self.gate_hidden_layers = None 
        self.lbl_values = None
        self.lbl_masks = None 
        self.cagrad_w = None
        self.log = None
        self.num_experts_task = None 
        self.num_experts_shared = None 
        self.expert_act_func = None
        self.tasks_weight = None
        
        self.input_gate_method = None
        self.refine_stop_gradient = None
        self.hidden_factor = None
        self.mask_embed = None

        self.diversity_loss = None

        self.batch_norm = batch_norm
        # sample weight配置项
        self.sample_weight = None
        # bid weight配置项
        self.bid_weight = None
        # transmitter参数
        self.g = None
        self.h1 = None
        self.h2 = None
        self.h3 = None
        self.selector = None
        # stem配置项
        self.num_specific_experts = None
        self.use_stem = None
        self.is_reduce = True
        # capedin配置项
        self.multi_hot_variable_len = None
        self.use_cape = False
        self.emb_reduce_method = "sum"
        self.cape_bn = False
        self.cape_keep_prob_rate = None
        self.cape_len = []
        self.cape_mlp = None
        self.cape_att_type = None
        self.cape_act_func = None
        self.cape_use_softmax = True
        self.inverse = False
        self.use_cnn = False
        self.filter_width = None
        self.stride = None
        self.conv_use_bn = False
        self.mid_channels = None
        self.sequence_fields = []
        self.cape_param = None
        self.cape_item_fields = []
        self.cape_seq_fields = []
        # 二维数组，前面配置item特征list;后面配置item_seq序列特征list
        self.cape_seq_item_dic = dict()
        self.sequence_field_index_list = []
        self.cape_add_field = None
        self.cape_din_layer_dic = {}
        self.cnn_layer_dic = {}
        self.embed_list = []
        self.parameter_init(config, dataset_argv, embedding_size,
                            reg_argv, expert_argv,
                            tower_argv,
                            trans_argv,
                            merge_multi_hot)
        # capedin_layer
        if self.use_cape:
            self.is_reduce = False
            for index, (seq_field_id, _) in enumerate(self.cape_seq_item_dic.items()):
                self.cape_din_layer = CapeLayer(scope_name=f"field_{seq_field_id}_cape_din",
                                                dropout_rate=self.cape_keep_prob_rate,
                                                batch_norm=self.cape_bn,
                                                emb_dim=self.embedding_size,
                                                mlp=self.cape_mlp, use_cape=self.use_cape,
                                                cape_len=self.cape_len[index],
                                                cape_att_type=self.cape_att_type, cape_act_func=self.cape_act_func,
                                                cape_c=self.cape_param, cape_use_softmax=self.cape_use_softmax,
                                                inverse=self.inverse)
                self.cape_din_layer_dic[seq_field_id] = self.cape_din_layer
        # cnn layer
        if self.use_cnn:
            self.is_reduce = False
            for _, field_id in enumerate(self.sequence_field_index_list):
                self.conv1d_layer = ResConvBlock(in_channels=self.embedding_size, out_channels=self.embedding_size,
                                                 mid_channels=self.mid_channels, stride=self.stride,
                                                 filter_width=self.filter_width, layer_name=f"field_{field_id}_conv1d",
                                                 conv_use_bn=self.conv_use_bn)
                self.cnn_layer_dic[field_id] = self.conv1d_layer
        gate_act_func = 'softmax'

        self.init_transmitter_layer()

        self.init_aitm_layer()

        self.init_input_layer(embedding_size, init_argv, regression_task)

        dense_flag = reduce(operator.or_,
                            dense_flags)
        
        self.init_placeholder(dense_flag)
        
        # construct input embedding layer
        vx_embed = self.construct_embedding(self.wt_hldr, self.id_hldr, self.embed_v, merge_multi_hot,
                                            is_reduce=self.is_reduce)
        vx_embed = self.build_embeddings(vx_embed, is_training=True)
        nn_input = tf.reshape(vx_embed, [-1, self.embedding_dim])

        if self.weight_method == 'gradnorm' and self.gradnorm_add_share:
            nn_input = self.sharebottom_layer(nn_input, training=True,
                                              reuse=False, name_scope='sharebottom_module')

        if self.use_stem:
            self.bottom_outputs = self.calculate_expert_gate_stem(nn_input, batch_norm, gate_act_func,
                                                                  self.num_experts, self.num_tasks,
                                                                  training=True)
        else:
            self.bottom_outputs = self.calculate_expert_gate(nn_input, batch_norm, gate_act_func,
                                                             self.num_experts, self.num_tasks,
                                                             training=True)

        if self.scenario_use_transmitter and self.use_aux_loss:
            x_stacks, aux_stacks = self.tower_layer(self.bottom_outputs, self.dense_hldr if dense_flag else None,
                                                    dense_flags, batch_norm, training=True)
        else:
            x_stacks = self.tower_layer(self.bottom_outputs, self.dense_hldr if dense_flag else None,
                                        dense_flags, batch_norm, training=True)
            aux_stacks = None
        final_layer_y = self.final_layer(x_stacks, init_argv, batch_norm=False, training=True)
        if aux_stacks is not None:
            final_layer_aux = self.final_layer(aux_stacks, init_argv, batch_norm=False, training=True)
        else:
            final_layer_aux = None

        self.train_preds(final_layer_y)

        if self.scenario_use_transmitter and final_layer_aux is not None:
            transmitter_loss = self.extract_unique_numbers(self.transmitter_tasks)
            if self.transmitter_unloss is not None:
                transmitter_unloss = self.transmitter_unloss
            else:
                transmitter_unloss = transmitter_loss[0]
            transmitter_loss_list = [item for item in transmitter_loss if item not in transmitter_unloss]

            print("========transmitter_loss_list is=========:", transmitter_loss_list)
        else:
            transmitter_loss_list = None

        self.loss_part(final_layer_y, ptmzr_argv, final_layer_aux, transmitter_loss_list)

        # construct input embedding layer
        eval_vx_embed = self.construct_embedding(self.eval_wt_hldr, self.eval_id_hldr,
                                                 self.embed_v, merge_multi_hot, is_reduce=self.is_reduce)
        eval_vx_embed = self.build_embeddings(eval_vx_embed, is_training=False)

        eval_nn_input = tf.reshape(eval_vx_embed, [-1, self.embedding_dim])
        
        if self.weight_method == 'gradnorm' and self.gradnorm_add_share: 
            eval_nn_input = self.sharebottom_layer(eval_nn_input,
                                              training=False, reuse=True,
                                              name_scope='sharebottom_module')

        if self.use_stem:
            self.eval_bottom_outputs = self.calculate_expert_gate_stem(eval_nn_input,
                                                                       batch_norm, gate_act_func,
                                                                       self.num_experts, self.num_tasks,
                                                                       training=False)
        else:
            self.eval_bottom_outputs = self.calculate_expert_gate(eval_nn_input,
                                                                  batch_norm, gate_act_func,
                                                                  self.num_experts, self.num_tasks,
                                                                  training=False)

        eval_x_stacks = self.tower_layer(self.eval_bottom_outputs, self.eval_dense_hldr if dense_flag else None,
                                         dense_flags, batch_norm,
                                         training=False)

        eval_final_layer_y = self.final_layer(eval_x_stacks, 
                                              init_argv,
                                              batch_norm=False, training=False)

        self.eval_preds(eval_final_layer_y)

        
        if self.weight_method == 'cagrad':
            self.update_cagrad_optimizer(ptmzr_argv)
        elif self.weight_method == 'cagrad_sgd':
            self.update_cagrad_sgd_optimizer(ptmzr_argv)
        elif self.weight_method == 'pcgrad':
            self.update_pcgrad_optimizer(ptmzr_argv)    
        else:
            self.update_optimizer(ptmzr_argv)

    # 修正原版bug
    def construct_embedding(self, wt_hldr, id_hldr,
                            embed_v, merge_multi_hot=False,
                            is_reduce=True):
        mask = tf.expand_dims(wt_hldr, 2)
        if merge_multi_hot and self.num_multihot > 0:
            # *_hot_mask is weight(values that follow the ids in the dataset,
            # different from weight of param) that used
            one_hot_mask, multi_hot_mask = split_mask(
                mask, self.multi_hot_flags, self.multi_hot_variable_len)

            one_hot_v, multi_hot_v = split_param(
                embed_v, id_hldr, self.multi_hot_flags)

            # fm part (reduce multi-hot vector's length to k*1)
            multi_hot_vx = sum_multi_hot(
                multi_hot_v, multi_hot_mask, self.multi_hot_variable_len, is_reduce=is_reduce)
            one_hot_vx = tf.multiply(one_hot_v, one_hot_mask)
            vx_embed = tf.concat([one_hot_vx, multi_hot_vx], axis=1)
        else:
            vx_embed = tf.multiply(tf.gather(embed_v, id_hldr), mask)
        return vx_embed
    
    @staticmethod
    def reduce_vx_emb(vx_embed, reduce_method):
        field_vx = []
        reduce_method_dic = {"sum": tf.reduce_sum,
                             "mean": tf.reduce_mean}
        for _, emb in enumerate(vx_embed):
            field_vx.append(reduce_method_dic.get(reduce_method, "sum")(emb, axis=1))
        return tf.concat(field_vx, axis=-1)
    
    def build_embeddings(self, vx_embed, is_training=True):
        field_embeddings_cnn = []
        field_embeddings_cape = []
        cape_output_list = []
        if self.is_reduce:
            return vx_embed
        # 从外面传进特征的配置对应训练数据,上面构建embedding时候是单值在前，多值在后
        self.embed_list = tf.split(vx_embed, num_or_size_splits=self.config.field_len_list, axis=1)
        # 序列特征使用cnn
        if self.use_cnn:
            for index, field_vx in enumerate(self.embed_list):
                # 序列特征加上conv1d信息进行替换
                if index in self.cnn_layer_dic.keys():
                    conv1d_layer = self.cnn_layer_dic.get(index)
                    sequence_field_emb = self.embed_list[index]
                    conv1d_out = conv1d_layer(sequence_field_emb, is_training=is_training)
                    field_embeddings_cnn.append(conv1d_out)
                else:
                    field_embeddings_cnn.append(field_vx)
            self.embed_list = field_embeddings_cnn
        # 增加cape计算
        if self.use_cape:
            if self.cape_add_field:
                for seq_field_id, item_field_id in self.cape_seq_item_dic.items():
                    cape_layer = self.cape_din_layer_dic.get(seq_field_id)
                    target_list = [self.embed_list[item_field_id]]
                    seq_list = [self.embed_list[seq_field_id]]
                    cape_out = cape_layer((target_list, seq_list), is_training=is_training)
                    cape_output_list.append(cape_out)
                for _, field_vx in enumerate(self.embed_list):
                    field_embeddings_cape.append(field_vx)
                field_embeddings_cape.extend(cape_output_list)
            else:
                for index, field_vx in enumerate(self.embed_list):
                    # 序列特征加上capedin信息进行替换
                    if index in self.cape_seq_item_dic.keys():
                        cape_layer = self.cape_din_layer_dic.get(index)
                        target_list = [self.embed_list[self.cape_seq_item_dic.get(index, None)]]
                        seq_list = [self.embed_list[index]]
                        cape_out = cape_layer((target_list, seq_list), is_training=is_training)
                        field_embeddings_cape.append(cape_out)
                    else:
                        field_embeddings_cape.append(field_vx)
            self.embed_list = field_embeddings_cape
        return self.reduce_vx_emb(self.embed_list, self.emb_reduce_method)
    
    def extract_unique_numbers(self, input_list):
        all_numbers = []
        for sublist in input_list:
            # 检查子元素是否为列表，如果是，则展开；否则，直接添加
            if isinstance(sublist, list):
                all_numbers.extend(sublist)
            else:
                all_numbers.append(sublist)
        # 使用set去重后转回列表
        unique_numbers = list(set(all_numbers))
        return unique_numbers

    def init_transmitter_layer(self):
        if self.scenario_use_transmitter:
            self.transmitter_loss_info = self.init_aitm_loss(self.transmitter_tasks)
            print("======transmitter loss info is:======", self.transmitter_loss_info)
            if (self.transmitter_tasks is None) or isinstance(self.transmitter_tasks[-1], list):
                num_tasks = self.num_tasks  # int
            else:
                num_tasks = len(self.transmitter_tasks)  # int
            norm_dim = self.transmitter_hidden_layers[-1]
            print("norm_dim size is : ", norm_dim)
            print("transmitter num_tasks is : ", num_tasks)
            print("transmitter_tasks is : ", self.transmitter_tasks)
            print("transmitter head num is :", self.transmitter_head_num)
            initializer = tf.random_normal_initializer(0.0, 0.001)
            self.g = [tf.layers.Dense(norm_dim, kernel_initializer=tf.random_normal_initializer(0.0, 0.001),
                                      bias_initializer=tf.zeros_initializer(),
                                      name="g_layer_{}".format(i)) for i in range(num_tasks - 1)]
            self.h1 = [tf.layers.Dense(norm_dim * self.transmitter_head_num, kernel_initializer=initializer,
                                       bias_initializer=tf.zeros_initializer(),
                                       name=f"h1_layer_{i}")
                       for i in range(num_tasks - 1)]

            self.h2 = [tf.layers.Dense(norm_dim * self.transmitter_head_num, kernel_initializer=initializer,
                                       bias_initializer=tf.zeros_initializer(),
                                       name=f"h2_layer_{i}")
                       for i in range(num_tasks - 1)]

            self.h3 = [tf.layers.Dense(norm_dim * self.transmitter_head_num, kernel_initializer=initializer,
                                       bias_initializer=tf.zeros_initializer(),
                                       name=f"h3_layer_{i}")
                       for i in range(num_tasks - 1)]

            self.selector = [tf.layers.Dense(
                norm_dim, kernel_initializer=initializer,
                bias_initializer=tf.zeros_initializer(), name="selector_layer_{}".format(i))
                for i in range(num_tasks - 1)]

    def init_aitm_loss(self, relate_tasks, first_transfer=False):
        aitm_loss = {i: None for i in range(self.num_tasks)}
        if (relate_tasks is None) or isinstance(relate_tasks[-1], int):
            for q, i in enumerate(relate_tasks[1:]):
                if first_transfer:
                    aitm_loss[i] = relate_tasks[0]
                else:
                    aitm_loss[i] = relate_tasks[q]
        else:
            for sub_task in relate_tasks:
                for q, i in enumerate(sub_task[1:]):
                    if first_transfer:
                        aitm_loss[i] = sub_task[0]
                    else:
                        aitm_loss[i] = sub_task[q]
        return aitm_loss

    def init_aitm_layer(self):
        if self.scenario_use_aitm:
            self.aitm_loss_info = self.init_aitm_loss(self.aitm_tasks, first_transfer=self.only_first_transfer)
            print("======aitm loss info is:======", self.aitm_loss_info)
            if (self.aitm_tasks is None) or isinstance(self.aitm_tasks[-1], int):
                # [1,3,4,5]如果是一条线路就是一套g和h
                num_tasks = 1  # int
            else:
                # [[1,2],[3,4]] 多条线路 多套g和h
                num_tasks = len(self.aitm_tasks)  # list
            norm_dim = self.aitm_hidden_layers[-1]
            initializer = tf.random_normal_initializer(0.0, 0.001)
            if num_tasks == 1:
                if self.only_first_transfer and self.only_one_g:
                    self.g = [[tf.layers.Dense(norm_dim, kernel_initializer=initializer,
                                               bias_initializer=tf.zeros_initializer(),
                                               name="g_layer")]]
                else:
                    self.g = [[tf.layers.Dense(norm_dim, kernel_initializer=initializer,
                                               bias_initializer=tf.zeros_initializer(),
                                               name="g_layer_{}".format(i)) for i in range(len(self.aitm_tasks) - 1)]]
            else:
                # 大于1层
                self.g = []
                if self.only_first_transfer and self.only_one_g:
                    for i in range(len(self.aitm_tasks)):
                        layer = tf.layers.Dense(
                            norm_dim,
                            kernel_initializer=initializer,
                            bias_initializer=tf.zeros_initializer(),
                            name="g_layer_{}".format(i)
                        )
                        self.g.append([layer])
                else:
                    for q, task in enumerate(self.aitm_tasks):
                        layers_for_task = []
                        num_layers = len(task) - 1
                        for i in range(num_layers):
                            layer = tf.layers.Dense(
                                norm_dim,
                                kernel_initializer=initializer,
                                bias_initializer=tf.zeros_initializer(),
                                name="g_layer_{}_{}".format(q, i)
                            )
                            layers_for_task.append(layer)
                        self.g.append(layers_for_task)
            print("g layer num is :", len(self.g), self.g)
            self.h1 = [tf.layers.Dense(norm_dim, kernel_initializer=initializer,
                                       bias_initializer=tf.zeros_initializer(),
                                       name=f"h1_layer_{i}")
                       for i in range(num_tasks)]

            self.h2 = [tf.layers.Dense(norm_dim, kernel_initializer=initializer,
                                       bias_initializer=tf.zeros_initializer(),
                                       name=f"h2_layer_{i}")
                       for i in range(num_tasks)]

            self.h3 = [tf.layers.Dense(norm_dim, kernel_initializer=initializer,
                                       bias_initializer=tf.zeros_initializer(),
                                       name=f"h3_layer_{i}")
                       for i in range(num_tasks)]


    def Multi_link_transmitter_net(self, bottom_outputs_dict, transmitter_tasks, detach=False, is_training=False):
        print("Multi_link_transmitter_tasks is :", transmitter_tasks)
        # ctr->cvr信息传递改进，引入cvr之间信息传递
        g_inputs = {}
        if self.use_other_cvr_info:
            g_inputs = self.fusion_ctr_and_cvr(bottom_outputs_dict, is_training)
        for i in transmitter_tasks:
            aux_stacks = []
            # 例如 i=[0,3,5]
            for q, j in enumerate(i):
                if q == 0:
                    continue
                print("Multi_link transmitter 数据迁移方向_第{}次:".format(q), [i[q - 1]], "_to_",
                      [i[q]])
                if self.use_other_cvr_info:
                    intermidate_state = self.aait(g_inputs.get(i[q], None),
                                                  bottom_outputs_dict[i[q]], j - 1, detach=detach,
                                                  is_training=is_training)
                else:
                    # 此处的j代表按照task索引取aait网络的层进行计算 aait网络相关都要按照task索引取拿
                    intermidate_state = self.aait(bottom_outputs_dict[i[q - 1]],
                                                  bottom_outputs_dict[i[q]], j - 1, detach=detach,
                                                  is_training=is_training)
                select_value = tf.sigmoid(self.selector[j - 1](
                    tf.concat([intermidate_state, bottom_outputs_dict[i[q]]], axis=1)))
                hidden_state = bottom_outputs_dict[i[q]] + select_value * intermidate_state
                # 最终的aux_stacks包含迁移信息之后的[0,3,5]这几个task
                aux_stacks.append(hidden_state)
            # 更新bottom_outputs_dict 使用aux_stacks
            for q, item in enumerate(i[1:]):  # [3,5]
                bottom_outputs_dict[item] = aux_stacks[q]
        return list(bottom_outputs_dict.values())

    def Multi_link_aitm_net(self, bottom_outputs_dict, transmitter_tasks, is_training=False):
        print("Multi_link_aitm_tasks is :", transmitter_tasks)
        for h, i in enumerate(transmitter_tasks):
            aux_stacks = []
            # 例如 i=[0,3,5]
            for q, _ in enumerate(i):
                if q == 0:
                    continue
                if self.only_first_transfer and self.only_one_g:
                    print("Multi_link aitm 数据迁移方向_第{}次(仅头传递一个g网络):".format(q), [i[0]], "_to_",
                          [i[q]])
                    ait = self.aitm(bottom_outputs_dict[i[0]],
                                    bottom_outputs_dict[i[q]], 0, g_num=h, h_num=h)
                elif self.only_first_transfer:
                    print("Multi_link aitm 数据迁移方向_第{}次(仅头传递不同g网络):".format(q), [i[0]], "_to_",
                          [i[q]])
                    ait = self.aitm(bottom_outputs_dict[i[0]],
                                    bottom_outputs_dict[i[q]], q - 1, g_num=h, h_num=h)
                else:
                    # 此处的j代表按照task索引取aait网络的层进行计算 aait网络相关都要按照task索引取拿
                    print("Multi_link transmitter 数据迁移方向_第{}次(依次传递):".format(q), [i[q - 1]], "_to_",
                          [i[q]])
                    ait = self.aitm(bottom_outputs_dict[i[q - 1]],
                                    bottom_outputs_dict[i[q]], q - 1, g_num=h, h_num=h)  # 此处的j代表按照task索引取aait网络的层进行计算
                # 最终的aux_stacks包含迁移信息之后的[0,3,5]这几个task
                aux_stacks.append(ait)
            # 更新bottom_outputs_dict 使用aux_stacks
            for q, item in enumerate(i[1:]):  # [3,5]
                bottom_outputs_dict[item] = aux_stacks[q]
        return list(bottom_outputs_dict.values())

    def scenario_transmitter_net(self, bottom_outputs, detach=False, is_training=False):
        # 对task任务加上序列号 方便计算完进行归位处理
        bottom_outputs_dict = {i: bottom_outputs[i] for i in range(self.num_tasks)}
        # 通过外部传参 例如信息迁移顺序[0,1],[0,2],[0,3,5]
        if self.transmitter_tasks is None:
            transmitter_tasks = list(bottom_outputs_dict.keys())
        else:
            transmitter_tasks = self.transmitter_tasks
        if isinstance(transmitter_tasks[-1], list):
            return self.Multi_link_transmitter_net(bottom_outputs_dict, transmitter_tasks, detach, is_training)
        # aux_stacks装填经过transmitter的task
        aux_stacks = []
        for i in range(1, len(transmitter_tasks)):
            print("transmitter 数据迁移方向_第{}次:".format(i), transmitter_tasks[i - 1], "_to_",
                  transmitter_tasks[i])
            intermidate_state = self.aait(bottom_outputs_dict[transmitter_tasks[i - 1]],
                                          bottom_outputs_dict[transmitter_tasks[i]], i - 1, detach=detach,
                                          is_training=is_training)
            select_value = tf.sigmoid(
                self.selector[i - 1](
                    tf.concat([intermidate_state, bottom_outputs_dict[transmitter_tasks[i]]], axis=1)))
            hidden_state = bottom_outputs_dict[transmitter_tasks[i]] + select_value * intermidate_state
            aux_stacks.append(hidden_state)
        # 更新bottom_outputs_dict 使用aux_stacks
        for i, item in enumerate(transmitter_tasks[1:]):
            bottom_outputs_dict[item] = aux_stacks[i]
        return list(bottom_outputs_dict.values())

    def fusion_ctr_and_cvr(self, bottom_outputs_dict, is_training=False):
        g_inputs = {}
        # 三类不同的方法 使得cvr进行transmitter时可以获得其他cvr的信息内容
        for i in range(1, self.num_tasks):  # 1-7
            total_tensors = [bottom_outputs_dict[0]]
            # cvr做向量聚合之前都需要进行stop_gradient
            cvr_tensors = [tf.stop_gradient(bottom_outputs_dict[j]) for j in range(1, self.num_tasks) if i != j]
            total_tensors.extend(cvr_tensors)
            # 将这些矩阵堆叠成一个形状为 (B, 7, emb_size) 的张量
            total_stacked_matrices = tf.stack(total_tensors, axis=1)
            print("total_tensor shape is :", total_stacked_matrices.shape)
            if self.trans_cvr_version == "v1":
                # 方法一直接聚合
                fusionInformation = tf.reduce_mean(total_stacked_matrices, axis=1)
                print(f"{i} method one fusionInformation is:{fusionInformation.shape}")
            elif self.trans_cvr_version == "v2":
                # 方法二 引入不同的gate网络计算
                cvr_stacked_matrices = tf.stack(cvr_tensors, axis=1)
                # ctr转换成权重
                gate_output = self.gate_forward_commonmodule(
                    scope=f"transmitter_gate_{i}_net",
                    nn_input=bottom_outputs_dict[0],
                    act_func="softmax",
                    batch_norm=self.batch_norm,
                    training=is_training,
                    gate_hidden_layers=self.trans_cvr_gate_hidden)
                # b * 1024
                fusionInformation = tf.reduce_sum(tf.expand_dims(gate_output, axis=-1) * cvr_stacked_matrices, axis=1)
                print(f"method two fusionInformation is:{fusionInformation.shape}")
                print(f"gete shape:{gate_output.shape}, cvr_stacked_matrices shape is:,{cvr_stacked_matrices.shape}")
            else:
                # 方法三 引入不同的w矩阵进行向量的聚合操作
                with tf.variable_scope(f"trans_w_{i}", reuse=tf.AUTO_REUSE):
                    weights = tf.get_variable(name=f'trans_w_{i}', shape=(len(total_tensors)),
                                              initializer=tf.initializers.random_normal())
                reshaped_weights = tf.expand_dims(tf.expand_dims(weights, 0), 2)
                result = tf.reduce_sum(total_stacked_matrices * reshaped_weights, axis=1)
                fusionInformation = tf.nn.relu(result)
                print(f"method three fusionInformation is:{fusionInformation.shape}")
            g_inputs[i] = fusionInformation
        return g_inputs

    def scenario_aitm_net(self, bottom_outputs, is_training=False):
        # 对task任务加上序列号 方便计算完进行归位处理
        bottom_outputs_dict = {i: bottom_outputs[i] for i in range(self.num_tasks)}
        # 通过外部传参 例如信息迁移顺序[0,1],[0,2],[0,3,5]
        transmitter_tasks = self.aitm_tasks  # 如果单条线路的话
        if isinstance(transmitter_tasks[-1], list):
            return self.Multi_link_aitm_net(bottom_outputs_dict, transmitter_tasks, is_training)
        # 单条线路
        aux_stacks = []
        for i in range(1, len(transmitter_tasks)):

            if self.only_first_transfer and self.only_one_g:
                print("aitm 数据迁移方向_第{}次(仅头传递一个g网络):".format(i), transmitter_tasks[0], "_to_",
                      transmitter_tasks[i])
                ait = self.aitm(bottom_outputs_dict[transmitter_tasks[0]],
                                bottom_outputs_dict[transmitter_tasks[i]], 0)
            elif self.only_first_transfer:
                print("aitm 数据迁移方向_第{}次(仅头传递使用不同的g网络):".format(i), transmitter_tasks[0], "_to_",
                      transmitter_tasks[i])
                ait = self.aitm(bottom_outputs_dict[transmitter_tasks[0]],
                                bottom_outputs_dict[transmitter_tasks[i]], i - 1)
            else:
                print("aitm 数据迁移方向_第{}次(依次传递):".format(i), transmitter_tasks[i - 1], "_to_",
                      transmitter_tasks[i])
                ait = self.aitm(bottom_outputs_dict[transmitter_tasks[i - 1]],
                                bottom_outputs_dict[transmitter_tasks[i]], i - 1)
                print("ait tensor shape is :", ait.shape)

            aux_stacks.append(ait)
        # 更新bottom_outputs_dict 使用aux_stacks
        for i, item in enumerate(transmitter_tasks[1:]):
            bottom_outputs_dict[item] = aux_stacks[i]
        return list(bottom_outputs_dict.values())

    def aait(self, last_out, task_fea, q, detach=False, is_training=False):
        print("use aait:", q)
        if detach:
            _p = tf.stop_gradient(self.g[q](last_out))
            _p = tf.expand_dims(_p, axis=1)
            _q = tf.stop_gradient(task_fea)
            _q = tf.expand_dims(_q, axis=1)
        else:
            _p = tf.expand_dims(self.g[q](last_out), axis=1)
            _q = tf.expand_dims(task_fea, axis=1)

        V_tensors = tf.reshape(self.h1[q](_p), [-1, self.transmitter_head_num, self.transmitter_hidden_layers[-1]])
        K_tensors = tf.reshape(self.h2[q](_p), [-1, self.transmitter_head_num, self.transmitter_hidden_layers[-1]])
        Q_tensors = tf.reshape(self.h3[q](_q), [-1, self.transmitter_head_num, self.transmitter_hidden_layers[-1]])

        head_attentions = self.scaled_dot_product_attention(Q_tensors, K_tensors, V_tensors)

        hidden_input = tf.reshape(head_attentions,
                                  (-1, head_attentions.shape[1] * head_attentions.shape[2]))

        hidden_state = self.mlp_module(hidden_input,
                                       hidden_layers=self.transmitter_hidden_layers,
                                       act_func='relu',
                                       dropout_prob=(1.0 - self.keep_prob),
                                       batch_norm=self.batch_norm,
                                       training=is_training,
                                       reuse=tf.AUTO_REUSE,
                                       name_scope=f'aait_out_layer_{q}_mlp')

        return hidden_state

    def aitm(self, last_out, task_fea, q, g_num=0, h_num=0):
        print("use aitm g h:", q, g_num, h_num)
        _q = tf.expand_dims(task_fea, axis=1)
        _p = tf.expand_dims(self.g[g_num][q](last_out), axis=1)
        inputs = tf.concat([_q, _p], axis=1)
        # (N,L,K)*(K,K)->(N,L,K)
        V = self.h1[h_num](inputs)
        K = self.h2[h_num](inputs)
        Q = self.h3[h_num](inputs)
        a = tf.reduce_sum(tf.multiply(Q, K), axis=-1) / \
            tf.sqrt(tf.cast(inputs.shape[-1], tf.float32))
        a = tf.nn.softmax(a, axis=1)
        outputs = tf.multiply(a[:, :, None], V)
        return tf.reduce_sum(outputs, axis=1)  # (N, K)

    def scaled_dot_product_attention(self, query, key, value, attn_mask=None,
                                     dropout_p=0.0, is_causal=False, scale=None):
        import math
        L, S = query.shape[-2], key.shape[-2]
        scale_factor = 1 / math.sqrt(query.shape[-1].value) if scale is None else scale
        attn_bias = tf.zeros((L, S), dtype=query.dtype)

        if is_causal and attn_mask is None:
            temp_mask = tf.linalg.band_part(tf.ones((L, S), dtype=tf.bool), -1, 0)
            attn_bias = tf.where(temp_mask, 0.0, float("-inf"))
            attn_bias = tf.cast(attn_bias, dtype=query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == tf.bool:
                attn_bias = tf.where(attn_mask, 0.0, float("-inf"))
            else:
                attn_bias += attn_mask

        attn_weight = tf.matmul(query, tf.transpose(key, perm=[0, 2, 1])) * scale_factor
        attn_weight += attn_bias
        attn_weight = tf.nn.softmax(attn_weight, axis=-1)
        attn_weight = tf.nn.dropout(attn_weight, rate=dropout_p)

        return tf.matmul(attn_weight, value)

    def parameter_init(self, config, dataset_argv, embedding_size,
                       reg_argv, expert_argv, tower_argv, trans_argv,
                       merge_multi_hot=False):
        self.config = config 
        (self.features_size,
         self.fields_num,
         self.dense_num,
         self.multi_hot_flags, multi_hot_len) = dataset_argv
        self.embedding_size = embedding_size

        one_hot_flags = [not flag for flag in self.multi_hot_flags]

        self.argv_init(expert_argv, tower_argv, reg_argv)
        
        self.num_onehot = int(sum(one_hot_flags))

        self.num_multihot = len(self.config.NUM_MULTI_HOT_FIELD)
        self.multi_hot_variable_len = self.config.NUM_MULTI_HOT_FIELD
        if merge_multi_hot:
            self.embedding_dim = int((self.num_multihot + self.num_onehot) * embedding_size)
        else:
            self.embedding_dim = int(self.fields_num * embedding_size)

        self.weight_method = self.config.WEIGHT_METHOD 
        self.gradnorm_alpha = self.config.GRADNORM_ALPHA
        self.gradnorm_add_share = self.config.GRADNORM_ADD_SHARE
        self.ca_alpha = self.config.CA_ALPHA
        self.ca_rescale = self.config.CA_RESCALE
        self.mgda_only_embed = self.config.MGDA_ONLY_EMBED
        self.mean_all_samples = self.config.MEAN_ALL_SAMPLES
        self.not_div_zero = self.config.NOT_DIV_ZERO
        self.input_gate_method = getattr(self.config, 'INPUT_GATE_METHOD', None)
        self.refine_stop_gradient = getattr(self.config, 'REFINE_STOP_GRADIENT', False)
        self.hidden_factor = getattr(self.config, 'HIDDEN_FACTOR', 0.25)
        self.uwl_task_num = getattr(self.config, 'UWL_TASK_NUM', 3)
        self.use_cape = getattr(self.config, 'USE_CAPE', False)
        self.emb_reduce_method = getattr(self.config, 'EMB_REDUCE_METHOD', "sum")
        self.cape_bn = getattr(self.config, 'CAPE_BN', False)
        self.cape_keep_prob_rate = getattr(self.config, 'CAPE_KEEP_PROB_RATE', 0.9)
        self.cape_len = getattr(self.config, 'CAPE_LEN', 7)
        self.cape_mlp = getattr(self.config, 'CAPE_MLP', [32])
        self.cape_att_type = getattr(self.config, 'CAPE_ATT_TYPE', "dot")
        self.cape_act_func = getattr(self.config, 'CAPE_ATC_FUNC', "dice")
        self.cape_use_softmax = getattr(self.config, 'CAPE_USE_SOFTMAX', True)
        self.inverse = getattr(self.config, 'INVERSE', False)
        self.use_cnn = getattr(self.config, 'USE_CNN', False)
        self.filter_width = getattr(self.config, 'FILTER_WIDTH', 5)
        self.stride = getattr(self.config, 'STRIDE', 1)
        self.conv_use_bn = getattr(self.config, 'CONV_USE_BN', False)
        self.mid_channels = getattr(self.config, 'MID_CHANNELS', 15)
        self.sequence_fields = getattr(self.config, 'SEQUENCE_FIELDS', [])
        self.cape_param = getattr(self.config, 'CAPE_PARAM', 30)
        self.cape_add_field = getattr(self.config, 'CAPE_ADD_FIELD', False)
        self.cape_item_fields = getattr(self.config, 'CAPE_ITEM_FIELDS', [])
        self.cape_seq_fields = getattr(self.config, 'CAPE_SEQ_FIELDS', [])
        # 特征索引,序列特征在前，item特征在后
        self.cape_seq_item_dic = dict(zip([self.config.field_name_list.index(j) for j in self.cape_seq_fields],
                                          [self.config.field_name_list.index(i) for i in self.cape_item_fields]
                                          ))
        # 序列特征索引
        self.sequence_field_index_list = [self.config.field_name_list.index(i) for i in self.sequence_fields]
        # capedin生成一个新的特征参数在最后需要增加embedding_dim
        if self.cape_add_field:
            self.embedding_dim += self.embedding_size * len(self.cape_seq_item_dic)
        # transmitter
        self.scenario_use_transmitter = getattr(self.config, "SCENARIO_USE_TRANSMITTER", False)
        if self.scenario_use_transmitter:
            self.use_aux_loss = getattr(self.config, "USE_AUX_LOSS", False)
            self.transmitter_detach = getattr(self.config, "TRANSMITTER_DETACH", False)
            self.transmitter_head_num = getattr(self.config, 'TRANSMITTER_HEAD_NUM', 4)
            self.transmitter_weight = getattr(self.config, 'TRANSMITTER_WEIGHT', 0.005)
            self.transmitter_tasks = getattr(self.config, 'TRANSMITTER_TASKS', None)
            # 不参与计算loss的头部信息任务序号
            self.transmitter_unloss = getattr(self.config, 'TRANSMITTER_UNLOSS', None)
            # expert_hidden_layers 网络出来之后的shape 最后的维度需要对齐
            self.transmitter_hidden_layers = self.config.TRANSMITTER_HIDDEN_LAYERS
            # transmitter 改进 ctr->cvr 之外 加入cvr之间的信息迁移部分代码
            self.use_other_cvr_info = getattr(self.config, "USE_OTHER_CVR_INFO", False)
            self.trans_cvr_version = getattr(self.config, "TRANS_CVR_VERSION", 'v1')
            self.trans_cvr_gate_hidden = getattr(self.config, "TRANS_CVR_GATE_HIDDEN", [128, 6])
            print("transmitter_params is :", self.scenario_use_transmitter, self.transmitter_detach,
                  self.transmitter_head_num, self.transmitter_hidden_layers, type(self.transmitter_hidden_layers))

        # sample weight
        self.label_with_sample = getattr(self.config, "LABEL_WITH_SAMPLE", False)
        self.use_sample_weight = getattr(self.config, "USE_SAMPLE_WEIGHT", False)
        self.ctr_use_sample_weight = getattr(self.config, "CTR_USE_SAMPLE_WEIGHT", False)

        # aitm
        self.scenario_use_aitm = getattr(self.config, "SCENARIO_USE_AITM", False)
        if self.scenario_use_aitm:
            self.use_aitm_loss = getattr(self.config, "USE_AITM_LOSS", False)
            self.only_first_transfer = getattr(self.config, "ONLY_FIRST_TRANSFER", False)
            self.only_one_g = getattr(self.config, "ONLY_ONE_G", False)
            self.aitm_loss_weight = getattr(self.config, 'AITM_LOSS_WEIGHT', 0.005)
            self.aitm_tasks = getattr(self.config, 'AITM_TASKS', None)
            self.aitm_hidden_layers = self.config.AITM_HIDDEN_LAYERS
        # stem 配置
        self.num_specific_experts = getattr(self.config, "NUM_SPECIFIC_EXPERTS", 1)
        self.use_stem = getattr(self.config, "USE_STEM", False)

        # bid weight
        self.label_with_bid = getattr(self.config, "LABEL_WITH_BID", False)
        self.use_bid_weight = getattr(self.config, "USE_BID_WEIGHT", False)
        self.ctr_use_bid_weight = getattr(self.config, "CTR_USE_BID_WEIGHT", False)

        log_dict = locals()
        log_dict['weight_method'] = self.weight_method       
        log_dict['gradnorm_alpha'] = self.gradnorm_alpha
        log_dict['gradnorm_add_share'] = self.gradnorm_add_share
        log_dict['mgda_only_embed'] = self.mgda_only_embed 
        log_dict['ca_alpha'] = self.ca_alpha
        log_dict['ca_rescale'] = self.ca_rescale
        log_dict['mean_all_samples'] = self.mean_all_samples    
        log_dict['not_div_zero'] = self.not_div_zero
        log_dict['input_gate_method'] = self.input_gate_method
        log_dict['refine_stop_gradient'] = self.refine_stop_gradient
        log_dict['hidden_factor'] = self.hidden_factor
        log_dict['use_div_loss'] = getattr(self.config, 'USE_DIV_LOSS', False)
        log_dict['div_loss_reg'] = getattr(self.config, 'DIV_LOSS_REG', 0.0)
        self.log = json.dumps(log_dict, default=str, sort_keys=True, indent=4)
        
    def argv_init(self, expert_argv, tower_argv, reg_argv):
        # num_tower_units: dcn deep-layers []
        self.num_tasks, self.has_task_mask, \
        self.expert_hidden_layers, self.gate_hidden_layers, \
            self.num_experts, self.expert_act_func,\
            self.tasks_weight = expert_argv
        tower_deep_layers, self.num_cross_layer, self.tower_act_func = tower_argv
        self.keep_prob, self._lambda, self.l1_lambda = reg_argv
        
        self.all_tower_deep_layers = [self.expert_hidden_layers[-1]] + tower_deep_layers
        self.dense_deep_layers = [self.expert_hidden_layers[-1] + self.dense_num] + tower_deep_layers

    def init_input_layer(self, embedding_size,
                         init_argv, regression_task=False):
        # init input layer
        init_acts = [('embed', [self.features_size, embedding_size], 'random'),
                     ('cross_w', [self.num_cross_layer, self.expert_hidden_layers[-1]], 'random'),
                     ('cross_b', [self.num_cross_layer, self.expert_hidden_layers[-1]], 'random')]

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
        # task mask embedding
        if self.input_gate_method == 'task_mask_gate':
            init_acts.append(('mask_embed', [self.num_tasks * 2, embedding_size], 'random'))

        var_map, log = init_var_map(init_argv, init_acts)

        self.log += json.dumps(locals(), default=str, sort_keys=True, indent=4)
        self.log += log
        self.input_variable(var_map)
        
    def input_variable(self, var_map):
        super().input_variable(var_map)
        if self.input_gate_method == 'task_mask_gate':
            self.mask_embed = tf.Variable(var_map['mask_embed'])

    def label_hldl_placeholder(self):
        if self.has_task_mask:
            if self.label_with_bid:
                self.lbl_hldr = tf.placeholder(tf.float32, shape=[None, self.num_tasks * 3 - 1])
                self.lbl_values, self.lbl_masks, self.bid_weight = tf.split(
                    self.lbl_hldr, [self.num_tasks, self.num_tasks, self.num_tasks - 1], axis=-1)

            elif self.label_with_sample:
                self.lbl_hldr = tf.placeholder(tf.float32, shape=[None, self.num_tasks * 2 + 1])
                self.lbl_values, self.lbl_masks, self.sample_weight = \
                    tf.split(self.lbl_hldr, [self.num_tasks, self.num_tasks, 1], axis=-1)

            else:
                self.lbl_hldr = tf.placeholder(tf.float32, shape=[None, self.num_tasks * 2])
                self.lbl_values, self.lbl_masks = tf.split(self.lbl_hldr, [self.num_tasks, self.num_tasks], axis=-1)
        else:
            self.lbl_hldr = tf.placeholder(tf.float32, shape=[None, self.num_tasks])
            self.lbl_values = self.lbl_hldr
            
    def task_refinement_network(self, in_embs, hidden_factor=0.25, output_size=None, scope="task_refine_net"):
        """
        Forward function of task refinement network.

        Args:
            in_embs (tensor): the input of the refinement network, 
                tensor of shape [batch_size, dim]
            hidden_factor (float): hidden_layer = hidden_factor * in_embs.shape[-1]. Default 0.25. 
            output_size (int, None): the size of the gate value. Default None, meaning same size of in_embs. 
            
        Returns:
            tensor: task-specific gate value, tensor of shape [batch_size, embedding_dim]
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            print('refine_in', in_embs)
            if output_size is None: 
                output_size = in_embs.get_shape().as_list()[-1]
            hidden_output = tf.layers.dense(
                in_embs, int(in_embs.get_shape().as_list()[-1] * hidden_factor), activation="relu",
                use_bias=True, name=f'mlp_0', reuse=tf.AUTO_REUSE, 
                kernel_initializer=tf.random_uniform_initializer(minval=-0.001, maxval=0.001)
            )
            refine = tf.layers.dense(hidden_output, output_size, activation="sigmoid",
                                     use_bias=True, name=f'mlp_1', reuse=tf.AUTO_REUSE,
                                     kernel_initializer=tf.random_uniform_initializer(minval=-0.001, maxval=0.001))
            # [batch_size, (num_feat+num_behaviour*num_feat_per_item) x embedding_size]
            print('refine_out', refine)
        return 2 * refine 


    def sharebottom_layer(self, nn_input, training=True, reuse=None, name_scope='sharebottom_module'):
        with tf.variable_scope(name_scope, reuse=reuse):
            output = tf.layers.dense(nn_input, self.embedding_dim, activation=None,
                                            use_bias=True, name=f'sharebottom_dense', reuse=reuse,
                                            kernel_initializer=tf.random_uniform_initializer(minval=-0.001, 
                                                                                             maxval=0.001))
            print(f'========= dense layer : {output.name} ==========')              
        return output
    
    def expert_forward(self, nn_input, num_experts, batch_norm, training, scope="MMOE_expert"):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            shared_expert_output_list = []
            for i in range(num_experts):
                shared_expert_output_list.append(
                    self.mlp_module(nn_input, 
                                    hidden_layers=self.expert_hidden_layers,
                                    act_func='relu', 
                                    dropout_prob=(1.0 - self.keep_prob),
                                    batch_norm=batch_norm, 
                                    training=training, 
                                    reuse=tf.AUTO_REUSE,
                                    name_scope=f'expert_{i}_mlp')) # [B, H_E]
            shared_expert_output = tf.stack(shared_expert_output_list, axis=-2) # [B, N_E, H_E]
        return shared_expert_output
    

    def gate_forward(self, scope, nn_input, act_func, batch_norm, training):
        if len(self.gate_hidden_layers) >= 2:
            nn_input = self.mlp_module(
                nn_input=nn_input, 
                hidden_layers=self.gate_hidden_layers[:-1],
                act_func='relu', 
                dropout_prob=(1.0 - self.keep_prob),
                batch_norm=batch_norm, 
                training=training, 
                reuse=tf.AUTO_REUSE,
                name_scope=f"{scope}_1"
            )
    
        gate_output = self.mlp_module(
            nn_input=nn_input, 
            hidden_layers=self.gate_hidden_layers[-1:],
            act_func=act_func, 
            dropout_prob=0.0,
            batch_norm=False, 
            training=training, 
            reuse=tf.AUTO_REUSE,
            name_scope=f"{scope}_2"
        )
        return gate_output

    def mask_refinement_component(self, nn_input):
        mask_offset = tf.convert_to_tensor(np.cumsum([2] * self.num_tasks), dtype=tf.int64) # [N_E]
        mask_emb = tf.gather(self.mask_embed, tf.cast(self.lbl_masks, tf.int64) + mask_offset - 2) # [B, N_E, D]
        mask_emb = tf.reshape(mask_emb, 
                              [-1, mask_emb.shape.as_list()[-1] * mask_emb.shape.as_list()[-2]]) # [B, N_E * D]
        refine_output = self.task_refinement_network(
            tf.concat(
                [tf.stop_gradient(nn_input) if self.refine_stop_gradient else nn_input, 
                 mask_emb], 
                axis=-1
            ),
            hidden_factor=self.hidden_factor,
            output_size=nn_input.shape.as_list()[-1],
            scope='mask_refine_net'
        )
        return tf.multiply(nn_input, refine_output)
    

    def refine_nn_input(self, nn_input, num_tasks):
        if self.input_gate_method == 'trm':
            print('using trm gate')
            nn_input_list = []
            for i in range(num_tasks):
                refine_output = self.task_refinement_network(nn_input, scope=f"task_{i}_refine_net")
                nn_input_list.append(tf.multiply(nn_input, refine_output))
            nn_input = tf.stack(nn_input_list, axis=1)  # [B, num_tasks, D]\
        elif self.input_gate_method == 'task_mask_gate':
            print('using task mask gate')
            nn_input = self.mask_refinement_component(nn_input)
        elif self.input_gate_method == 'single_gate':
            print('using single gate')
            refine_output = self.task_refinement_network(
                in_embs=nn_input, 
                hidden_factor=self.hidden_factor,
                scope=f'single_task_refine_net'
            )
            nn_input = tf.multiply(nn_input, refine_output)
        else:
            nn_input = nn_input
        return nn_input


    def cal_expert_diversity_loss(self, expert_output):
        norm_expert_out = tf.norm(expert_output, axis=-1, keepdims=True)    # [B, N_E, 1]
        norm_expert_out_T = tf.transpose(norm_expert_out, [0, 2, 1])    # [B, 1, N_E]
        norm_dot = tf.matmul(norm_expert_out, norm_expert_out_T)    # [B, N_E, N_E]

        expert_output_T = tf.transpose(expert_output, [0, 2, 1])   # [B, H_E, N_E] 
        dot_product = tf.matmul(expert_output, expert_output_T)   # [B, N_E, N_E]
        cosine = dot_product / (norm_dot + 1e-10)
        
        diversity_loss = tf.reduce_mean(cosine)
        return diversity_loss
        
    
    def cal_gate_output(self, nn_input, num_tasks, gate_act_func, batch_norm, training):
        gate_outputs = []
        if self.weight_method == 'dr' or self.weight_method == 'drvb':
            num_task_gates = 2 * num_tasks - 1
        else:
            num_task_gates = num_tasks
        for i in range(num_task_gates):
            if self.input_gate_method == 'trm':
                gate_in = nn_input[:, i, :]
            else:
                gate_in = nn_input

            gate_output = self.gate_forward(
                scope=f"gate_{i}_net",
                nn_input=gate_in,
                act_func=gate_act_func,
                batch_norm=batch_norm,
                training=training
            ) # [B, N_E]
            gate_outputs.append(gate_output) 
        gate_outputs = tf.stack(gate_outputs, axis=1) # [B, N_task, N_E]
        return gate_outputs
    

    def syn_expert_with_gate(self, expert_output, gate_output):
        if self.input_gate_method == 'trm':
            gate_output = tf.expand_dims(gate_output, axis=-1) # [B, N_task, N_E, 1]
            print("shared_expert_output:", expert_output)
            print("gate_outputs:", gate_output)
            bottom_outputs = tf.multiply(expert_output, gate_output)
            bottom_outputs = tf.reduce_sum(bottom_outputs, axis=-2)  # [B, N_task, H_E]
        else:
            bottom_outputs = tf.matmul(gate_output, expert_output)  # [B, N_task, H_E]
        
        bottom_outputs = tf.transpose(bottom_outputs, [1, 0, 2])    # [N_task, B, H_E]
        return bottom_outputs

    def calculate_expert_gate_stem(
            self,
            nn_input,
            batch_norm,
            gate_act_func,
            num_experts, num_tasks,
            training
    ):
        # refine input with gate
        nn_input = self.refine_nn_input(nn_input=nn_input, num_tasks=num_tasks)

        split_dim = int(nn_input.shape[1]) // self.embedding_size
        single_dim = self.embedding_size // (self.num_tasks + 1)
        nn_input = tf.reshape(nn_input, [-1, split_dim, self.embedding_size])
        print("nn_input shape is :", nn_input)
        # 切分成多个emb层
        feature_embs = tf.split(nn_input, num_or_size_splits=self.num_tasks + 1, axis=2)
        print("feature_embs infomation is:", feature_embs[1].shape, len(feature_embs))
        # 维度聚合
        stem_inputs = []
        for i in range(self.num_tasks + 1):
            stem_input = tf.reshape(feature_embs[i], [-1, split_dim * single_dim])
            stem_inputs.append(stem_input)

        print("stem_inputs infomation is:", stem_inputs[1].shape, len(stem_inputs))
        print("expert and gate info is:", self.gate_hidden_layers, self.expert_hidden_layers, num_experts)

        num_shared_experts = num_experts
        print("共享专家数:", num_shared_experts)
        shared_expert_outputs = self.expert_forward_stem(
            stem_inputs[-1],
            num_shared_experts,
            batch_norm,
            training,
            scope="stem_share_expert"
        )
        num_specific_experts = self.num_specific_experts
        print("任务特有专家数:", num_specific_experts)
        specific_expert_outputs = []
        for i in range(self.num_tasks):
            distinct_expert_outputs = self.expert_forward_stem(
                stem_inputs[i],
                num_specific_experts,
                batch_norm,
                training,
                scope="stem_specific_expert_{}".format(i)
            )
            specific_expert_outputs.append(distinct_expert_outputs)
        # gate
        stem_outputs = []
        if self.weight_method == 'dr' or self.weight_method == 'drvb':
            num_task_gates = 2 * self.num_tasks - 1
        else:
            num_task_gates = self.num_tasks
        for i in range(num_task_gates):
            gate_input = []
            for j in range(self.num_tasks):
                if j == i or (i > j and (i % self.num_tasks + 1) == j):
                    gate_input.extend(specific_expert_outputs[j])
                else:
                    specific_expert_outputs_j = specific_expert_outputs[j]
                    specific_expert_outputs_j = [tf.stop_gradient(out) for out in specific_expert_outputs_j]
                    gate_input.extend(specific_expert_outputs_j)
            gate_input.extend(shared_expert_outputs)
            gate_input = tf.stack(gate_input, axis=1)
            gate = tf.nn.softmax(self.gate_forward_commonmodule(
                scope=f"gate_{i}_net",
                nn_input=stem_inputs[i if i < self.num_tasks else (i % self.num_tasks + 1)] + stem_inputs[-1],
                act_func=gate_act_func,
                batch_norm=batch_norm,
                training=training))
            print("gate(v2) shape is:", gate.shape, self.gate_hidden_layers)
            stem_output = tf.reduce_sum(tf.expand_dims(gate, axis=-1) * gate_input, axis=1)
            print("stem_output shape is:", stem_output.shape)
            stem_outputs.append(stem_output)
        return stem_outputs

    def expert_forward_stem(self, nn_input, num_experts, batch_norm, training, scope="MMOE_expert"):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            shared_expert_output_list = []
            for i in range(num_experts):
                shared_expert_output_list.append(
                    self.mlp_module(nn_input,
                                    hidden_layers=self.expert_hidden_layers,
                                    act_func='relu',
                                    dropout_prob=(1.0 - self.keep_prob),
                                    batch_norm=batch_norm,
                                    training=training,
                                    reuse=tf.AUTO_REUSE,
                                    name_scope=f'expert_{i}_mlp'))
        return shared_expert_output_list

    def gate_forward_commonmodule(self, scope, nn_input, act_func, batch_norm, training, gate_hidden_layers=None):
        if gate_hidden_layers is None:
            gate_hidden_layers = self.gate_hidden_layers
        if len(gate_hidden_layers) >= 2:
            nn_input = self.mlp_module(
                nn_input=nn_input,
                hidden_layers=gate_hidden_layers[:-1],
                act_func='relu',
                dropout_prob=(1.0 - self.keep_prob),
                batch_norm=batch_norm,
                training=training,
                reuse=tf.AUTO_REUSE,
                name_scope=f"{scope}_1"
            )

        gate_output = self.mlp_module(
            nn_input=nn_input,
            hidden_layers=gate_hidden_layers[-1:],
            act_func=act_func,
            dropout_prob=0.0,
            batch_norm=False,
            training=training,
            reuse=tf.AUTO_REUSE,
            name_scope=f"{scope}_2"
        )
        return gate_output


    def calculate_expert_gate(
            self, 
            nn_input, 
            batch_norm,
            gate_act_func,
            num_experts, num_tasks,
            training
        ):
        # refine input with gate
        nn_input = self.refine_nn_input(nn_input=nn_input, num_tasks=num_tasks)

        # first forward: input -> experts
        shared_expert_output = self.expert_forward(
            nn_input,
            num_experts,
            batch_norm,
            training,
            scope="MMOE_expert"
        )   # [B, num_tasks, N_E, H_E] or [B, N_E, H_E]
        
        # calculate diversity of expert output
        if training and self.config.USE_DIV_LOSS:            
            self.diversity_loss = self.cal_expert_diversity_loss(shared_expert_output)
        
        # second forward: input -> gate value
        gate_outputs = self.cal_gate_output(
            nn_input=nn_input,
            num_tasks=num_tasks,
            gate_act_func=gate_act_func,
            batch_norm=batch_norm,
            training=training
        )   # [B, N_task, N_E]
        
        bottom_outputs = self.syn_expert_with_gate(
            expert_output=shared_expert_output, 
            gate_output=gate_outputs
        )    # [N_task, B, H_E]

        return bottom_outputs

    def tower_layer(self, bottom_outputs, dense_hldr, dense_flags, batch_norm, training):
        x_stacks = []
        aux_stacks = []
        # transmitter 和 aitm用到
        final_hl_list = []
        x_l_list = []
        if self.weight_method == 'dr' or self.weight_method == 'drvb':
            for i in range(2 * self.num_tasks - 1):
                print("forward 的 i = ", i)
                x_l, final_hl = self.forward(bottom_outputs[i],
                                             self.cross_w[i % self.num_tasks],  # % 8
                                             self.cross_b[i % self.num_tasks],
                                             self.h_w[i % self.num_tasks],
                                             self.h_b[i % self.num_tasks],
                                             dense_hldr,
                                             self.expert_hidden_layers[-1],
                                             self.tower_act_func,
                                             dense_flags[i % self.num_tasks],
                                             training=training,
                                             batch_norm=batch_norm,
                                             name_scope="tower%d" % (i + 1))
                final_hl_list.append(final_hl)
                x_l_list.append(x_l)
            if self.scenario_use_aitm:
                final_hl_trans = self.scenario_aitm_net(final_hl_list,
                                                        is_training=training)
                # 替换操作
                final_hl_list[:self.num_tasks] = final_hl_trans

                print("=====here is aitm logit length=====", len(final_hl_list[:self.num_tasks]),
                      len(final_hl_trans))
            if self.scenario_use_transmitter:
                final_hl_trans = self.scenario_transmitter_net(final_hl_list,
                                                               detach=self.transmitter_detach,
                                                               is_training=training)
                if self.use_aux_loss and training:
                    for i in range(2 * self.num_tasks - 1):
                        if self.use_aux_loss and training:
                            aux_stack = final_hl_list[i] if dense_flags[i % self.num_tasks] else tf.concat(
                                [x_l_list[i], final_hl_list[i]], 1)
                            aux_stacks.append(aux_stack)
                # 替换操作
                final_hl_list[:self.num_tasks] = final_hl_trans

            for i in range(2 * self.num_tasks - 1):
                x_stack = final_hl_list[i] if dense_flags[i % self.num_tasks] else tf.concat(
                    [x_l_list[i], final_hl_list[i]], 1)
                x_stacks.append(x_stack)
        else:
            for i in range(self.num_tasks):
                x_l, final_hl = self.forward(bottom_outputs[i],
                                             self.cross_w[i],
                                             self.cross_b[i],
                                             self.h_w[i],
                                             self.h_b[i],
                                             dense_hldr,
                                             self.expert_hidden_layers[-1],
                                             self.tower_act_func,
                                             dense_flags[i],
                                             training=training,
                                             batch_norm=batch_norm,
                                             name_scope="tower%d" % (i + 1))
                final_hl_list.append(final_hl)
                x_l_list.append(x_l)
            if self.scenario_use_aitm:
                final_hl_trans = self.scenario_aitm_net(final_hl_list,
                                                        is_training=training)
                # 替换操作
                final_hl_list[:self.num_tasks] = final_hl_trans
                print("====aitm logit length====", len(final_hl_list[:self.num_tasks]), len(final_hl_trans))
            if self.scenario_use_transmitter:
                final_hl_trans = self.scenario_transmitter_net(final_hl_list,
                                                               detach=self.transmitter_detach,
                                                               is_training=training)
                if self.use_aux_loss and training:
                    for i in range(self.num_tasks):
                        if self.use_aux_loss and training:
                            aux_stack = final_hl_list[i] if dense_flags[i] else tf.concat(
                                [x_l_list[i], final_hl_list[i]], 1)
                            aux_stacks.append(aux_stack)
                # 替换操作
                final_hl_list[:self.num_tasks] = final_hl_trans

            for i in range(self.num_tasks):
                x_stack = final_hl_list[i] if dense_flags[i] else tf.concat(
                    [x_l_list[i], final_hl_list[i]], 1)
                x_stacks.append(x_stack)
        if self.scenario_use_transmitter and (self.use_aux_loss and training):
            return [x_stacks, aux_stacks]
        else:
            return x_stacks

    def final_layer(self, x_stacks, init_argv, batch_norm, training):
        if training:
            self.final_variable(x_stacks, init_argv)
        final_layer_y = []

        if self.weight_method == 'dr' or self.weight_method == 'drvb':
            for i in range(2 * self.num_tasks - 1):
                y = self.final_forward(x_stacks[i],
                                       self.out_w[i % self.num_tasks],
                                       self.out_b[i % self.num_tasks],
                                       training=training,
                                       batch_norm=batch_norm,
                                       name_scope="task%d" % (i + 1))
                final_layer_y.append(y)
        else:
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
        self.train_preds_cvr_list = []
        for i, final_y in enumerate(final_layer_y):
            self.train_preds_cvr_list.append(tf.sigmoid(final_y, name=f'predictions_cvr{i}'))

    def eval_preds(self, eval_final_layer_y):
        self.eval_preds_cvr_list = []
        for i, final_y in enumerate(eval_final_layer_y):
            self.eval_preds_cvr_list.append(tf.sigmoid(final_y))
            self.eval_preds_cvr_list[i] = tf.identity(self.eval_preds_cvr_list[i], name=f'cvr_prediction_node{i}')

    def loss_weight_variable(self):
        if self.weight_method == 'gradnorm': 
            self.loss_weights = tf.Variable([1.0 for _ in range(self.num_tasks)])
            self.init_losses = tf.Variable([-1.0 for _ in range(self.num_tasks)], trainable=False, name='init_losses')
        elif self.weight_method == 'mgda':
            self.loss_weights = tf.convert_to_tensor([1.0 for _ in range(self.num_tasks)])
        elif self.weight_method == "uwl":
            for i in range(1, self.uwl_task_num + 1):
                self.uncertainties.append(tf.get_variable(
                    'uncertainties_loss_w_' + str(i),
                    dtype=tf.float32, shape=(1,), initializer=tf.constant_initializer(0.),
                    trainable=True
                ))

    
    def loss_part(self, final_layer_y, ptmzr_argv, final_layer_aux=None, transmitter_loss_list=None):
        self.loss_weight_variable()
        
        if self.weight_method == 'gradnorm': 
            
            if self.gradnorm_add_share:
                weights_shared = []
                for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                    if 'sharebottom' in var.name:
                        weights_shared.append(var)
            else:
                weights_shared = self.embed_v
            
            self.loss = 0
            self.task_losses = []
            for i, final_y in enumerate(final_layer_y):
                loss_i = tf.nn.sigmoid_cross_entropy_with_logits(logits=final_y,
                                                                 labels=self.lbl_values[:, i]) # [B]
                if final_layer_aux is not None:
                    if transmitter_loss_list is not None and (i in transmitter_loss_list):
                        print("gradnorm====transmitter_{} loss, weight:{}===="
                              .format(i, self.transmitter_weight))
                        aux_loss_i = tf.nn.sigmoid_cross_entropy_with_logits(logits=final_layer_aux[i],
                                                                             labels=self.lbl_values[:, i])
                        loss_diff = aux_loss_i - loss_i
                        aux_loss_i = tf.maximum(loss_diff,
                                                tf.zeros_like(loss_diff)) * self.transmitter_weight
                        loss_i += aux_loss_i

                if self.has_task_mask:
                    loss_i = tf.multiply(loss_i, 
                                         self.lbl_masks[:, i])
                
                if self.mean_all_samples:
                    loss_i = tf.reduce_mean(loss_i)
                else:
                    num_samples = tf.reduce_sum(self.lbl_masks[:, i])
                    if self.not_div_zero: 
                        num_samples = tf.maximum(num_samples, 
                                                 tf.constant(1.0))
                    loss_i = tf.reduce_sum(loss_i) / num_samples
                
                self.task_losses.append(loss_i)
                self.loss += loss_i * self.loss_weights[i]
            
            # set initial losses    
            multi_task_losses = tf.convert_to_tensor(self.task_losses)
                        
            def assign_init_losses():
                with tf.control_dependencies([tf.assign(self.init_losses, multi_task_losses)]):
                    return tf.identity(self.init_losses)
            
            tmp_init_losses = tf.cond(
                tf.reduce_sum(self.init_losses) < 0,
                assign_init_losses,
                lambda: tf.identity(self.init_losses)
            )
        
            # calculate L2-norm of gradients of each task 
            grad_norm_list = []
            for i, task_loss in enumerate(self.task_losses):
                grad_norm = tf.norm(tf.gradients(task_loss, weights_shared)[0], ord=2)
                grad_norm_list.append(self.loss_weights[i] * grad_norm)
            grad_norm_mean = tf.math.divide(tf.math.add_n(grad_norm_list), self.num_tasks * 1.0)
            
            # calculate inverse training rate 
            inv_rates = multi_task_losses / tmp_init_losses
            inv_rate_mean = tf.reduce_mean(inv_rates)
            rel_inv_rates = inv_rates / inv_rate_mean
       
            # calculate constant grad norm target 
            gn_targets = tf.multiply(
                tf.convert_to_tensor([grad_norm_mean]),
                tf.pow(rel_inv_rates, self.gradnorm_alpha)
            )
            gn_targets = tf.stop_gradient(gn_targets)
            
            # calculate GradNorm loss 
            self.loss_gradnorm = tf.norm((tf.convert_to_tensor(grad_norm_list) - gn_targets), ord=1)
            
            self.update_gradnorm_optimizer(ptmzr_argv=ptmzr_argv)
            
            # Renormalize weights
            with tf.control_dependencies([self.gradnorm_ptmzr]):
                coef = tf.div(self.num_tasks * 1.0, tf.reduce_sum(self.loss_weights))
                # assign operation doesn't generate gradients
                self.update_loss_weights_op = self.loss_weights.assign(tf.multiply(self.loss_weights, coef)) 
        
        elif self.weight_method == 'mgda':
            
            weights_shared = [self.embed_v]
            if not self.mgda_only_embed:
                for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                    if 'expert' in var.name:
                        weights_shared.append(var)
            
            self.task_losses = []
            for i, final_y in enumerate(final_layer_y):
                loss_i = tf.nn.sigmoid_cross_entropy_with_logits(logits=final_y, 
                                                                 labels=self.lbl_values[:, i]) # [B]
                if final_layer_aux is not None:
                    if transmitter_loss_list is not None and (i in transmitter_loss_list):
                        print("mgda====transmitter_{} loss, weight:{}===="
                              .format(i, self.transmitter_weight))
                        aux_loss_i = tf.nn.sigmoid_cross_entropy_with_logits(logits=final_layer_aux[i],
                                                                             labels=self.lbl_values[:, i])
                        loss_diff = aux_loss_i - loss_i
                        aux_loss_i = tf.maximum(loss_diff,
                                                tf.zeros_like(loss_diff)) * self.transmitter_weight
                        loss_i += aux_loss_i

                if self.has_task_mask:
                    loss_i = tf.multiply(loss_i, 
                                         self.lbl_masks[:, i])
                
                if self.mean_all_samples:
                    loss_i = tf.reduce_mean(loss_i)
                else:
                    num_samples = tf.reduce_sum(self.lbl_masks[:, i])
                    if self.not_div_zero: 
                        num_samples = tf.maximum(num_samples, tf.constant(1.0))
                    loss_i = tf.reduce_sum(loss_i) / num_samples
                
                self.task_losses.append(loss_i)
                
            task_grads = []
            for loss in self.task_losses:
                _grads = []
                for _, g in enumerate(tf.gradients(loss, weights_shared)):
                    _grads.append(tf.reshape(g, [-1]))
                task_grads.append(tf.concat(_grads, axis=-1))

            self.loss_weights = tf.py_func(mgda_alg, task_grads, Tout=tf.float32)
            self.loss = tf.multiply(tf.convert_to_tensor(self.task_losses), self.loss_weights)
        elif self.weight_method == 'pcgrad' or self.weight_method == 'cagrad' or self.weight_method == 'cagrad_sgd':
            self.task_losses = []
            for i, final_y in enumerate(final_layer_y):
                loss_i = tf.nn.sigmoid_cross_entropy_with_logits(logits=final_y,
                                                                 labels=self.lbl_values[:, i]) # [B]
                if final_layer_aux is not None:
                    if transmitter_loss_list is not None and (i in transmitter_loss_list):
                        print("pcgrad or cagrad====transmitter_{} loss , weight:{}===="
                              .format(i, self.transmitter_weight))
                        aux_loss_i = tf.nn.sigmoid_cross_entropy_with_logits(logits=final_layer_aux[i],
                                                                             labels=self.lbl_values[:, i])
                        loss_diff = aux_loss_i - loss_i
                        aux_loss_i = tf.maximum(loss_diff,
                                                tf.zeros_like(loss_diff)) * self.transmitter_weight
                        loss_i += aux_loss_i

                if self.has_task_mask:
                    loss_i = tf.multiply(loss_i, 
                                         self.lbl_masks[:, i])
                
                if self.mean_all_samples:
                    loss_i = tf.reduce_mean(loss_i)
                else:
                    num_samples = tf.reduce_sum(self.lbl_masks[:, i])
                    if self.not_div_zero: 
                        num_samples = tf.maximum(num_samples, 
                                                 tf.constant(1.0))
                    loss_i = tf.reduce_sum(loss_i) / num_samples
                
                self.task_losses.append(loss_i) 
        else:
            self.loss = 0
            for i, final_y in enumerate(final_layer_y):
                
                loss_i = tf.nn.sigmoid_cross_entropy_with_logits(logits=final_y, labels=self.lbl_values[:, i]) # [B]

                if final_layer_aux is not None:
                    if transmitter_loss_list is not None and (i in transmitter_loss_list):
                        print("else====transmitter_{} loss , weight:{}===="
                              .format(i, self.transmitter_weight))
                        aux_loss_i = tf.nn.sigmoid_cross_entropy_with_logits(logits=final_layer_aux[i],
                                                                             labels=self.lbl_values[:, i])
                        loss_diff = aux_loss_i - loss_i
                        aux_loss_i = tf.maximum(loss_diff,
                                                tf.zeros_like(loss_diff)) * self.transmitter_weight
                        loss_i += aux_loss_i

                if self.has_task_mask:
                    loss_i = tf.multiply(loss_i, self.lbl_masks[:, i])
                if self.mean_all_samples:
                    loss_i = tf.reduce_mean(loss_i)
                else:
                    num_samples = tf.reduce_sum(self.lbl_masks[:, i])
                    num_samples = tf.maximum(num_samples, tf.constant(1.0))
                    loss_i = tf.reduce_sum(loss_i) / num_samples

                if self.tasks_weight is not None:
                    loss_i = tf.multiply(loss_i, self.tasks_weight[i])
                self.loss += loss_i
            
            if (getattr(self, 'diversity_loss', None) is not None) and self.config.USE_DIV_LOSS:
                print("Add diversity loss!")
                self.loss += self.config.DIV_LOSS_REG * self.diversity_loss
          
        if self.weight_method != 'pcgrad' and self.weight_method != 'cagrad' and self.weight_method != 'cagrad_sgd':      
            self.loss += tf.contrib.layers.l1_regularizer(self.l1_lambda)(self.cross_w[0]) \
                        + tf.contrib.layers.l1_regularizer(self.l1_lambda)(self.cross_b[0]) \
                        + tf.contrib.layers.l1_regularizer(self.l1_lambda)(self.cross_w[1]) \
                        + tf.contrib.layers.l1_regularizer(self.l1_lambda)(self.cross_b[1]) \
                        + self._lambda * tf.nn.l2_loss(self.embed_v)
    
    def update_gradnorm_optimizer(self, ptmzr_argv):
        with tf.control_dependencies([self.loss_gradnorm]):
        
            gradnorm_learning_rate = \
                tf.train.exponential_decay(learning_rate=ptmzr_argv[1], global_step=self.global_step,
                                           decay_rate=ptmzr_argv[3], decay_steps=ptmzr_argv[4],
                                           staircase=False)
            if distribute_tmp:
                gradnorm_opt = \
                    tf.train.AdamOptimizer(learning_rate=gradnorm_learning_rate * hvd.size(), epsilon=ptmzr_argv[2])
                gradnorm_opt = \
                    hvd.DistributedOptimizer(gradnorm_opt, compression=hvd.Compression.fp16, sparse_as_dense=True)
            else:
                gradnorm_opt = tf.train.AdamOptimizer(learning_rate=gradnorm_learning_rate, epsilon=ptmzr_argv[2])
            self.gradnorm_gradients = gradnorm_opt.compute_gradients(self.loss_gradnorm, var_list=[self.loss_weights])
            for i, (g, v) in enumerate(self.gradnorm_gradients):
                if g is not None:
                    self.gradnorm_gradients[i] = (tf.clip_by_value(g, -1, 1), v)
            self.gradnorm_ptmzr = gradnorm_opt.apply_gradients(self.gradnorm_gradients)
        log = 'gradnorm optimizer: %s, learning rate: %g, epsilon: %g\n' % (ptmzr_argv[0], ptmzr_argv[1], ptmzr_argv[2])
        self.log += log
        
    def update_optimizer(self, ptmzr_argv):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            learning_rate, opt = self.get_lr_and_opt(ptmzr_argv)
            
            vars_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            if self.weight_method == 'gradnorm': 
                vars_list.remove(self.loss_weights.experimental_ref().deref())
            self.gradients = opt.compute_gradients(self.loss, var_list=vars_list)
            for i, (g, v) in enumerate(self.gradients):
                if g is not None:
                    self.gradients[i] = (tf.clip_by_value(g, -1, 1), v)
            self.ptmzr = opt.apply_gradients(self.gradients)
            log = 'optimizer: %s, learning rate: %g, epsilon: %g\n' % (ptmzr_argv[0], ptmzr_argv[1], ptmzr_argv[2])
        self.log += log
        params_count = get_params_count()
        self.log += 'total count of trainable variables is: %d' % params_count
        
    def update_pcgrad_optimizer(self, ptmzr_argv):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            learning_rate, opt = self.get_lr_and_opt(ptmzr_argv)
                
            grads, shapes, has_grads = [], [], []
            pc_var = None
            for i, task_loss in enumerate(self.task_losses):
                task_gradients = opt.compute_gradients(task_loss)
                if i == 0:
                    pc_var = [v for _, v in task_gradients]
                grad, shape, has_grad = pcgrad_retrieve_grad(task_gradients)
                grads.append(pcgrad_flatten_grad(grad, shape))
                has_grads.append(pcgrad_flatten_grad(has_grad, shape))
                shapes.append(shape)
            pc_grad = pcgrad_project_conflicting(grads, has_grads)
            pc_grad = pcgrad_unflatten_grad(pc_grad, shapes[0])
            
            pc_gradients = list(zip(pc_grad, pc_var))
            for i, (g, v) in enumerate(pc_gradients):
                if g is not None:
                    pc_gradients[i] = (tf.clip_by_value(g, -1, 1), v)
            self.ptmzr = opt.apply_gradients(pc_gradients)
            
            log = 'pcgrad optimizer: %s, learning rate: %g, epsilon: %g\n' \
                % (ptmzr_argv[0], ptmzr_argv[1], ptmzr_argv[2])
        self.log += log
        params_count = get_params_count()
        self.log += 'total count of trainable variables is: %d' % params_count

    def update_cagrad_optimizer(self, ptmzr_argv):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            learning_rate, opt = self.get_lr_and_opt(ptmzr_argv)
                
            grads, shapes = [], []
            ca_var = None
            for i, task_loss in enumerate(self.task_losses):
                task_gradients = opt.compute_gradients(task_loss)
                if i == 0:
                    ca_var = [v for _, v in task_gradients]
                grad, shape, _ = pcgrad_retrieve_grad(task_gradients)
                grads.append(pcgrad_flatten_grad(grad, shape))
                shapes.append(shape)
            grads_stacked = tf.stack(grads, axis=0) # [N_t, M]
            GG = tf.matmul(grads_stacked, tf.transpose(grads_stacked)) # [N_t, N_t]
            ca_grad = tf.py_func(cagrad_algo, 
                                [grads_stacked, GG, tf.convert_to_tensor(self.num_tasks), 
                                 tf.convert_to_tensor(self.ca_alpha), 
                                 tf.convert_to_tensor(self.ca_rescale)], 
                                Tout=tf.float32)
            ca_grad = pcgrad_unflatten_grad(ca_grad, shapes[0])
            
            ca_gradients = list(zip(ca_grad, ca_var))
            for i, (g, v) in enumerate(ca_gradients):
                if g is not None:
                    ca_gradients[i] = (tf.clip_by_value(g, -1, 1), 
                                       v)
            self.ptmzr = opt.apply_gradients(ca_gradients)
            
            log = 'cagrad optimizer: %s, learning rate: %g, epsilon: %g\n' \
                % (ptmzr_argv[0], ptmzr_argv[1], ptmzr_argv[2])
        self.log += log
        params_count = get_params_count()
        self.log += 'total count of trainable variables is: %d' % params_count    
        
    def update_cagrad_sgd_optimizer(self, ptmzr_argv):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            learning_rate, opt = self.get_lr_and_opt(ptmzr_argv)
                
            grads, shapes = [], []
            ca_var = None
            for i, task_loss in enumerate(self.task_losses):
                task_gradients = opt.compute_gradients(task_loss)
                if i == 0:
                    ca_var = [v for _, v in task_gradients]
                grad, shape, _ = pcgrad_retrieve_grad(task_gradients)
                grads.append(pcgrad_flatten_grad(grad, shape))
                shapes.append(shape)
            grads_stacked = tf.stack(grads, 
                                     axis=0) # [N_t, M]
            GG = tf.matmul(grads_stacked, 
                           tf.transpose(grads_stacked)) # [N_t, N_t]
            ca_grad = self.cagrad_sgd(grads_stacked, 
                                      GG)
            ca_grad = pcgrad_unflatten_grad(ca_grad,
                                            shapes[0])
            
            ca_gradients = list(zip(ca_grad, ca_var))
            for i, (g, v) in enumerate(ca_gradients):
                if g is not None:
                    ca_gradients[i] = (tf.clip_by_value(g, -1, 1), v)
            self.ptmzr = opt.apply_gradients(ca_gradients)
            
            log = 'cagrad optimizer: %s, learning rate: %g, epsilon: %g\n' \
                % (ptmzr_argv[0], ptmzr_argv[1], ptmzr_argv[2])
        self.log += log
        params_count = get_params_count()
        self.log += 'total count of trainable variables is: %d' % params_count
        
        
    def get_lr_and_opt(self, ptmzr_argv):
        learning_rate = tf.train.exponential_decay(learning_rate=ptmzr_argv[1], global_step=self.global_step,
                                                       decay_rate=ptmzr_argv[3], decay_steps=ptmzr_argv[4],
                                                       staircase=False)
        if distribute_tmp:
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate * hvd.size(), epsilon=ptmzr_argv[2])
            opt = hvd.DistributedOptimizer(opt, compression=hvd.Compression.fp16, sparse_as_dense=True)
        else:
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=ptmzr_argv[2])
        return learning_rate, opt
        
    
    def cagrad_sgd_init(self):
        self.cagrad_w = tf.Variable([[0.0] for _ in range(self.num_tasks)]) # [N_t, 1]
    
    def cagrad_sgd(self, grads, GG):
        self.cagrad_sgd_init()
        
        scale = tf.math.reduce_mean(tf.math.sqrt(tf.diag_part(GG) + 1e-4))
        GG = GG / tf.math.pow(scale, 2)
        Gg = tf.math.reduce_mean(GG, axis=1, keepdims=True)
        gg = tf.math.reduce_mean(Gg, axis=0, keepdims=True)    
        
        with tf.control_dependencies([tf.assign(self.cagrad_w, tf.zeros([self.num_tasks, 1]))]):
            
            w_opt = tf.train.MomentumOptimizer(learning_rate=10, momentum=0.5)
            
            c = tf.sqrt(gg + 1e-4) * self.ca_alpha

            w_best = tf.convert_to_tensor([[0.0] for _ in range(self.num_tasks)])
            obj_best = tf.convert_to_tensor(float('inf'))
            self.w_opt_ops = []
            for i in range(21):
                ww = tf.nn.softmax(self.cagrad_w, axis=0)
                obj = tf.transpose(ww) @ Gg + \
                    c * tf.sqrt((tf.transpose(ww) @ GG @ ww) + 1e-4)
                
                # update best
                obj_best = tf.where(tf.reshape(obj, []) < obj_best, tf.reshape(obj, []), obj_best)
                w_best = tf.where(tf.tile(tf.reshape(obj < obj_best, [1, 1]), [self.num_tasks, 1]), 
                                self.cagrad_w, w_best)
                if i < 20:
                    w_gradients = w_opt.compute_gradients(obj, var_list=self.cagrad_w)
                    opt_op = w_opt.apply_gradients(w_gradients)
                    self.w_opt_ops.append(opt_op)

            ww = tf.nn.softmax(w_best, axis=0) # [N_t, 1]
            gw_norm = tf.sqrt(tf.transpose(ww) @ GG @ ww + 1e-4)

            lmbda = c / (gw_norm + 1e-4)
            g = tf.reduce_sum((1 / self.num_tasks + ww * lmbda) * grads, axis=0) / (1 + self.ca_alpha**2)
        return g

    def mlp_module(self, nn_input, hidden_layers, act_func='relu', dropout_prob=0.1, 
                   batch_norm=False, inf_dropout=False, training=True, reuse=True, name_scope='mlp_module'):

        with tf.variable_scope(name_scope, reuse=reuse):
            hidden_output = nn_input
            for i, layer in enumerate(hidden_layers):
                hidden_output = tf.layers.dense(hidden_output, layer, activation=act_func,
                                                use_bias=True, name=f'mlp_{i}', reuse=reuse,
                                                kernel_initializer=tf.random_uniform_initializer(minval=-0.001, 
                                                    maxval=0.001))
                print(f'========= dense layer : {hidden_output.name} ==========')
                if batch_norm:
                    in_shape = hidden_output.get_shape().as_list()
                    if self.input_gate_method == 'trm' and len(in_shape) == 3:
                        # [B, num_tasks, D] -> [B, num_tasksxD]
                        hidden_output = tf.reshape(hidden_output, [-1, in_shape[1]*in_shape[2]])
                        hidden_output = tf.layers.batch_normalization(
                            hidden_output, training=training, reuse=reuse, name=f'bn_{i}'
                        )
                        hidden_output = tf.reshape(hidden_output, [-1, in_shape[1], in_shape[2]])
                    else:
                        hidden_output = tf.layers.batch_normalization(
                            hidden_output, training=training, reuse=reuse, name=f'bn_{i}'
                        )
                    print(f'========= batch_norm : {hidden_output.name} ==========')
                    
                if training or inf_dropout:
                    # dropout at inference time only if inf_dropout is True
                    print(f"""apply dropout in *{'training' if training else 'inference'}* time,
                          inf_dropout: {inf_dropout}""")
                    hidden_output = tf.nn.dropout(hidden_output, rate=dropout_prob)
                    
            return hidden_output