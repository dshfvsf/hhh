from __future__ import print_function
import tensorflow as tf
import tensorflow_probability as tfp
from train.models.tf_util import activate, sum_multi_hot, split_param, split_mask, hidden_forward

tfd = tfp.distributions


class MMOE_GWD(MMOE):
    def __init__(self, dataset_argv, embedding_size, expert_argv, tower_argv,
                 init_argv, ptmzr_argv, reg_argv, dense_flags,
                 regression_task=True,  # 主任务为回归
                 trans_argv=None,
                 u_init_argv=None,
                 loss_mode='full',
                 merge_multi_hot=False,
                 batch_norm=True):

        # 添加鲸鱼用户参数
        self.whale_threshold = tf.Variable(100.0, trainable=True, dtype=tf.float32, name='whale_threshold')
        self.whale_p = tf.Variable(0.5, trainable=True, dtype=tf.float32, name="whale_p")

        # 修改任务参数：第一个回归，第二个分类
        self.num_tasks = 2
        self.regression_task = [True, False]

        super().__init__(dataset_argv, embedding_size, expert_argv, tower_argv,
                         init_argv, ptmzr_argv, reg_argv, dense_flags,
                         regression_task, trans_argv, u_init_argv,
                         loss_mode, merge_multi_hot, batch_norm)

    def init_input_layer(self, embedding_size, init_argv, u_init_argv, regression_task=False):
        # 扩展初始化参数
        init_acts = [
            ('embed', [self.features_size, embedding_size], 'random'),
            ('experts', [self.embedding_dim, self.num_expert_units, self.num_experts], 'random'),
            ('cross_w', [self.num_cross_layer, self.num_expert_units], 'random'),
            ('cross_b', [self.num_cross_layer, self.num_expert_units], 'random')
        ]

        # 分类任务门控
        for i in range(self.num_tasks):
            init_acts.append(('gate%d' % (i + 1), [self.embedding_dim, self.num_experts], 'random'))

        # 修改塔层初始化：第一个任务回归，第二个分类
        # 回归塔
        for i in range(len(self.dense_deep_layers) - 1):
            init_acts.extend([
                ('h%d_w_regression' % (i + 1), self.dense_deep_layers[i:i + 2], 'random'),
                ('h%d_b_regression' % (i + 1), [self.dense_deep_layers[i + 1]], 'random')
            ])

        # 分类塔
        for i in range(len(self.all_tower_deep_layers) - 1):
            init_acts.extend([
                ('h%d_w_class' % (i + 1), self.all_tower_deep_layers[i:i + 2], 'random'),
                ('h%d_b_class' % (i + 1), [self.all_tower_deep_layers[i + 1]], 'random')
            ])

        var_map, log = init_var_map(init_argv, init_acts)
        self.log += log
        self.input_variable(var_map)
        self.expert_gate_variable(var_map)

    def tower_layer(self, bottom_outputs, dense_hldr, dense_flags, batch_norm, training):
        x_stacks = []
        for i in range(self.num_tasks):
            # 任务1：回归任务
            if i == 0:
                x_l, final_hl = self.regression_forward(
                    bottom_outputs[i],
                    self.cross_w[i],
                    self.cross_b[i],
                    self.h_w[i],
                    self.h_b[i],
                    dense_hldr,
                    self.num_expert_units,
                    training
                )
            # 任务2：分类任务
            else:
                x_l, final_hl = self.classification_forward(
                    bottom_outputs[i],
                    self.cross_w[i],
                    self.cross_b[i],
                    self.h_w[i],
                    self.h_b[i],
                    dense_hldr,
                    self.num_expert_units,
                    training
                )

            x_stack = final_hl if dense_flags[i] else tf.concat([x_l, final_hl], 1)
            x_stacks.append(x_stack)
        return x_stacks

    def regression_forward(self, vx_embed, cross_w, cross_b, h_w, h_b,
                           dense_feature, num_expert_units, training):
        # 回归任务前向传播
        x_0 = tf.reshape(vx_embed, [-1, num_expert_units])
        x_l = x_0
        # 交叉网络
        for i in range(self.num_cross_layer):
            xlw = tf.tensordot(x_l, cross_w[i], axes=1)
            x_l = x_0 * tf.expand_dims(xlw, -1) + cross_b[i] + x_l

        # 深度网络
        hidden_output = tf.reshape(vx_embed, [-1, num_expert_units])
        for i in range(len(h_w)):
            hidden_output = tf.tensordot(hidden_output, h_w[i], axes=1) + h_b[i]
            if i < len(h_w) - 1:
                hidden_output = activate('relu', hidden_output)
                if training:
                    hidden_output = tf.nn.dropout(hidden_output, self.keep_prob)

        return x_l, hidden_output

    def classification_forward(self, vx_embed, cross_w, cross_b, h_w, h_b,
                               dense_feature, num_expert_units, training):
        # 分类任务前向传播
        x_0 = tf.reshape(vx_embed, [-1, num_expert_units])
        x_l = x_0
        # 交叉网络
        for i in range(self.num_cross_layer):
            xlw = tf.tensordot(x_l, cross_w[i], axes=1)
            x_l = x_0 * tf.expand_dims(xlw, -1) + cross_b[i] + x_l

        # 深度网络
        hidden_output = tf.reshape(vx_embed, [-1, num_expert_units])
        for i in range(len(h_w)):
            hidden_output = tf.tensordot(hidden_output, h_w[i], axes=1) + h_b[i]
            if self.batch_norm:
                hidden_output = tf.layers.batch_normalization(
                    hidden_output, training=training, reuse=not training)
            if i < len(h_w) - 1:
                hidden_output = activate('relu', hidden_output)
                if training:
                    hidden_output = tf.nn.dropout(hidden_output, self.keep_prob)

        return x_l, hidden_output

    def loss_part(self, final_layer_y):
        # LTV回归损失（Gamma分布）
        ltv_logits = final_layer_y[0]
        alpha = tf.nn.softplus(ltv_logits[:, 0]) + 1e-8
        beta = tf.nn.softplus(ltv_logits[:, 1]) + 1e-8
        ltv_labels = self.lbl_hldr_1  # 假设第一个标签是LTV

        # Gamma负对数似然
        gamma_loss = -tf.reduce_mean(
            tfd.Gamma(concentration=alpha, rate=beta).log_prob(ltv_labels)

        # 鲸鱼用户分类损失
        whale_logits = final_layer_y[1]
        whale_labels = tf.cast(self.lbl_hldr_2 > self.whale_threshold, tf.float32)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=whale_labels, logits=whale_logits)

        # KL散度正则
        whale_probs = tf.sigmoid(whale_logits)
        kl_loss = tf.reduce_mean(
            whale_labels * tf.math.log(whale_labels / (whale_probs + 1e-8)) +
            (1 - whale_labels) * tf.math.log((1 - whale_labels) / (1 - whale_probs + 1e-8)))

        # 组合损失
        total_loss = gamma_loss + cross_entropy + 0.1 * kl_loss

        # 添加正则项
        if self.fields_num != 0:
            total_loss += tf.contrib.layers.l1_regularizer(self.l1_lambda)(self.cross_w)
        total_loss += tf.contrib.layers.l1_regularizer(self.l1_lambda)(self.cross_b)
        total_loss += self._lambda * tf.nn.l2_loss(self.embed_v)

        self.loss = total_loss

    def train_preds(self, final_layer_y):
        # LTV预测
        ltv_logits = final_layer_y[0]
        alpha = tf.nn.softplus(ltv_logits[:, 0])
        beta = tf.nn.softplus(ltv_logits[:, 1])
        ltv_pred = alpha / beta  # Gamma分布的期望

        # 鲸鱼用户概率
        whale_pred = tf.sigmoid(final_layer_y[1])

        # 组合预测
        adjusted_ltv = tf.where(
            whale_pred > tf.sigmoid(self.whale_p),
            ltv_pred * 1.2,  # 对鲸鱼用户加成
            ltv_pred
        )
        self.train_preds = tf.identity(adjusted_ltv, name='predictions')

    def eval_train_preds(self, eval_final_layer_y):
        # 评估模式预测
        ltv_logits = eval_final_layer_y[0]
        alpha = tf.nn.softplus(ltv_logits[:, 0])
        beta = tf.nn.softplus(ltv_logits[:, 1])
        ltv_eval = alpha / beta

        whale_eval = tf.sigmoid(eval_final_layer_y[1])

        final_eval = tf.where(
            whale_eval > tf.sigmoid(self.whale_p),
            ltv_eval * 1.2,
            ltv_eval
        )
        self.eval_preds = tf.identity(final_eval, name='predictionNode')