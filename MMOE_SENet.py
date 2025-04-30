from __future__ import print_function
import json

import tensorflow as tf
try:
    import horovod.tensorflow as hvd
    distribute_tmp = True
except Exception:
    print("have no horovod package")
    distribute_tmp = False
from models.MMOE_ALL import MMOE_ALL


class MMOE_SENet(MMOE_ALL):
    """
    Add SENets before MMOE Experts. The model is different with the task refine module in MMOE, the major difference
    lie in that: the SENet of this model is not task-specific, which is designed for experts. We assume that various 
    experts are good at dealing with various views of features, therefore assigning different weights on original 
    features.

    There are two strategies in the model:
        - each expert a SENet: one SENet is set for each expert, assign `SHARE_SENET=False`
        - sharing SENets for all experts: all experts share several SENets, two arguments: 
          `SHARE_SENET=True`, `NUM_SENET=5`

    """
    def __init__(self, config, dataset_argv, embedding_size, expert_argv, tower_argv,
                 init_argv, ptmzr_argv, reg_argv, dense_flags, regression_task=False,
                 trans_argv=None, u_init_argv=None, loss_mode='full', merge_multi_hot=False, batch_norm=True):
        self.share_senet = None
        self.num_senet = None
        super().__init__(config, dataset_argv, embedding_size, expert_argv, tower_argv, 
                         init_argv, ptmzr_argv, reg_argv, dense_flags, regression_task, 
                         trans_argv, u_init_argv, loss_mode, merge_multi_hot, batch_norm)

    def parameter_init(self, config, dataset_argv, embedding_size,
                       reg_argv, expert_argv, tower_argv, trans_argv,
                       merge_multi_hot=False):
        super().parameter_init(
            config, dataset_argv, embedding_size,
            reg_argv, expert_argv, tower_argv, trans_argv,
            merge_multi_hot
        )
        self.share_senet = self.config.SHARE_SENET
        self.num_senet = self.config.NUM_SENET
        log_dict = {
            'share_senet': self.share_senet,
            'num_senet': self.num_senet
        }
        self.log += json.dumps(log_dict, default=str, sort_keys=True, indent=4)


    def expert_in_gate(self, senet_outs, scope="MMOE_expert_in_gate"):
        """
        Gates for multiple shared SENets for all experts.

        Args:
            senet_outs (Tensor): output of shared SENets, tensor of shape (batch size, number of SENets, hidden size)
            scope (str): the scope name

        Returns:
            Tensor: the weighted output for each expert, tensor of shape (number of experts, batch size, hidden_size)
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            gate_out = tf.layers.dense(
                senet_outs, # [B, N, H]
                self.num_experts,
                activation=None,
                use_bias=False,
                name=f'mlp'
            )
            weight = tf.transpose(gate_out, perm=[0, 2, 1]) # [B, N_expert, N_senet]
            weight = tf.nn.softmax(weight, axis=-1)    # [B, N_expert, N_senet]
            out = tf.matmul(weight, senet_outs)  # [B, N_expert, H]
            return tf.transpose(out, perm=[1, 0, 2])


    def share_senets(self, nn_input, scope="share_senets"):
        """
        Shared SENets.

        Args:
            nn_inpput (Tensor): the input feature representation, tensor of shape (batch size, hidden size)
            scope (str): the scope name

        Returns:
            Tensor: outputs of multiple SENets, tensor of shape (batch size, number of SENets, hidden_size)
        """
        out = []
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            for i in range(self.num_senet):
                gate_weight = self.task_refinement_network(
                    nn_input,
                    hidden_factor=0.25,
                    scope=f"SENet_{i}"
                )
                out.append(tf.multiply(gate_weight, nn_input))
        out = tf.stack(out, axis=1) # [B, N, H]
        return out


    def expert_forward(self, nn_input, num_experts, batch_norm, training, scope="MMOE_expert"):

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            if self.share_senet:
                # strategy 2: sharing SENets for all experts
                share_senet_outs = self.share_senets(nn_input)  # [B, N_SENet, H]
                expert_inputs = self.expert_in_gate(share_senet_outs)   # [N_expert, B, H]

            shared_expert_output_list = []
            for i in range(num_experts):
                if self.share_senet:
                    # strategy 2: sharing SENets for all experts
                    nn_input_senet = expert_inputs[i]
                else:
                    # strategy 1: each expert a SENet
                    gate_weight = self.task_refinement_network(
                        nn_input,
                        hidden_factor=0.25,
                        scope=f"SENet_expert_{i}"
                    )
                    nn_input_senet = tf.multiply(nn_input, gate_weight)
                shared_expert_output_list.append(
                    self.mlp_module(nn_input_senet, 
                                    hidden_layers=self.expert_hidden_layers,
                                    act_func='relu', 
                                    dropout_prob=(1.0 - self.keep_prob),
                                    batch_norm=batch_norm, 
                                    training=training, 
                                    reuse=tf.AUTO_REUSE,
                                    name_scope=f'expert_{i}_mlp')) # [B, H_E]
            shared_expert_output = tf.stack(shared_expert_output_list, axis=-2) # [B, N_E, H_E]
        return shared_expert_output
