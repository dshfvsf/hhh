from __future__ import print_function
import tensorflow as tf
import numpy as np

try:
    import horovod.tensorflow as hvd

    distribute_tmp = True
except Exception:
    print("have no horovod package")
    distribute_tmp = False

import os
import pickle
import json
import operator
from functools import reduce
import pandas as pd
from train.models.tf_util import build_optimizer, init_var_map, \
    get_field_index, get_field_num, split_mask, split_param, sum_multi_hot, \
    activate, get_params_count
from train.models.utils.grad_optm import mgda_alg, \
    pcgrad_retrieve_grad, pcgrad_flatten_grad, \
    pcgrad_project_conflicting, pcgrad_unflatten_grad, \
    cagrad_algo
from models.ESCM2_ALL_Mixup import ESCM2_ALL_Mixup


class STEM_Mixup(ESCM2_ALL_Mixup):

    def __init__(self, config, dataset_argv, embedding_size, expert_argv, tower_argv,
                 init_argv, ptmzr_argv, reg_argv, dense_flags, regression_task=False,
                 trans_argv=None, u_init_argv=None, loss_mode='full', merge_multi_hot=False, batch_norm=True):
        self.config = None
        self.lbl_values = None
        self.lbl_masks = None
        self.has_task_mask = None
        self.id_hldr = None
        self.diversity_loss = None

        super().__init__(
            config, dataset_argv, embedding_size, expert_argv, tower_argv,
            init_argv, ptmzr_argv, reg_argv, dense_flags, regression_task,
            trans_argv, u_init_argv, loss_mode, merge_multi_hot, batch_norm)

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

        if training:
            num_synth_sample = self.num_synth_sample
            alpha = self.alpha
            nn_input = self.get_synth_embs_and_lbs(nn_input, self.lbl_values, self.lbl_masks, self.id_hldr, alpha,
                                                   num_synth_sample)

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