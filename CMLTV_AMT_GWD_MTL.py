from __future__ import print_function
from functools import reduce
import json
import operator
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from train.models.tf_util import init_var_map, split_mask, split_param, sum_multi_hot, activate, hidden_forward, \
    get_params_count

from train.models.CMLTV_AMT import CMLTV_AMT

tfd = tfp.distributions


class CMLTV_AMT_GWD_MTL(CMLTV_AMT):
    def __init__(self, config, dataset_argv, init_argv, checkpoint, sess):

        self.train_preds = None
        self.loss_reg = None
        self.loss_class = None
        self.loss = None
        self.eval_preds = None
        self.wt_hldr = None
        self.id_hldr = None
        self.eval_wt_hldr = None
        self.eval_id_hldr = None
        self.dense_hldr = None
        self.eval_dense_hldr = None
        
        self.lbl_hldr_1 = None
        self.lbl_hldr_2 = None
        self.soft_hldr = None
        self.train_preds_whale = None
        self.train_preds_ltv = None
        self.eval_preds_whale = None
        self.eval_preds_ltv = None
        self.loss_whale = None
        self.global_step = None
        self.distill = config.DISTILL
        self.whale_threshold = config.WHALE_THRESHOLD
        super().__init__(config, dataset_argv, init_argv, checkpoint, sess)

    def placeholder_part(self):
        if self.fields_num != 0:
            self.wt_hldr = tf.placeholder(tf.float32, shape=[None, self.fields_num])
            self.id_hldr = tf.placeholder(tf.int64, shape=[None, self.fields_num])
            self.eval_wt_hldr = tf.placeholder(tf.float32, [None, self.fields_num], name='wt')
            self.eval_id_hldr = tf.placeholder(tf.int64, [None, self.fields_num], name='id')

        if self.dense_num != 0:
            self.dense_hldr = tf.placeholder(tf.float32, shape=[None, self.dense_num])
            self.eval_dense_hldr = tf.placeholder(tf.float32, [None, self.dense_num], name='dense')
        self.lbl_hldr_1 = tf.placeholder(tf.float32)
        self.lbl_hldr_2 = tf.placeholder(tf.float32)

        if self.distill:
            self.soft_hldr = tf.placeholder(tf.float32)
        self.global_step = tf.placeholder(tf.int64, name='global_step')

    def labcvar(self, class_preds, label_class_onehot, label01):
        class_weights_list = []

        for w, num in self.class_weights:
            class_weights_list.extend([float(self.class_num) / float(w)] * num)

        class_weights_list_np = np.array(class_weights_list, dtype=np.float32)
        class_weights = tf.constant(class_weights_list_np / sum(class_weights_list_np), dtype=tf.float32)
        weight_map = tf.reduce_sum(tf.multiply(label_class_onehot, class_weights), axis=1)

        return tf.reduce_mean(label01 * tf.multiply(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=label_class_onehot, logits=class_preds), weight_map)) * \
            tf.cast(tf.reduce_sum(tf.ones_like(self.lbl_hldr_1), keepdims=True), tf.float32) / \
            (1e-3 + tf.reduce_sum(label01, keepdims=True))

    def output_train_pred(self, logits, classification_logits):

        self.train_preds_whale = tf.sigmoid(logits[:, 0])

        self.train_preds_ltv = tf.reduce_sum(
            tf.nn.softmax(classification_logits) *
            (tf.cast(tf.ones_like(classification_logits), tf.float32) *
             (tf.cast(2 ** tf.range(self.class_num), tf.float32) +
              tf.cast(2 ** tf.range(1, 1 + self.class_num), tf.float32) - 3) / 2),
            axis=-1
        )

        final_res = self.train_preds_ltv

        self.train_preds = tf.identity(final_res, name='predictions')

    def loss_part(self, logits, class_preds):
        label01 = tf.cast(self.lbl_hldr_1 > 0, tf.float32)
        label02 = tf.cast(self.lbl_hldr_2 > self.whale_threshold, tf.float32)
        safe_labels = label01 * self.lbl_hldr_1 + (1 - label01) * tf.ones_like(self.lbl_hldr_1)

        alpha = tf.math.maximum(tf.nn.softplus(logits[:, 1]), 1e-4)
        beta = tf.math.maximum(tf.nn.softplus(logits[:, 2]), 1e-4)

        self.loss_reg = -tf.reduce_mean(label01 * tfd.Gamma(concentration=alpha, rate=beta).log_prob(safe_labels))

        label_class = tf.math.minimum(tf.math.floor(tf.math.log(1 + tf.math.maximum(
            self.lbl_hldr_1, 0.)) / tf.math.log(2.)), self.class_num - 1)
        label_class_onehot = tf.one_hot(indices=tf.cast(label_class, dtype=tf.int32),
                                        depth=self.class_num, axis=-1, dtype=tf.float32)

        self.loss_class = self.labcvar(class_preds, label_class_onehot, label01)

        reg_loss = 0
        if self.fields_num != 0:
            reg_loss += tf.contrib.layers.l1_regularizer(self.l1_lambda)(self.cross_w)
            reg_loss += tf.contrib.layers.l1_regularizer(self.l1_lambda)(self.cross_b)
            reg_loss += self._lambda * tf.nn.l2_loss(self.embed_v)

        self.loss_whale = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=label02,
                logits=logits[:, 0]
            )
        )

        loss = self.loss_w1 * self.loss_reg + self.loss_w2 * self.loss_class + self.loss_w3 * self.loss_whale

        if self.fields_num != 0:
            self.loss = loss + tf.contrib.layers.l1_regularizer(self.l1_lambda)(
                self.cross_w) + tf.contrib.layers.l1_regularizer(self.l1_lambda)(
                self.cross_b) + self._lambda * tf.nn.l2_loss(
                self.embed_v)
        else:
            self.loss = loss

    def output_eval_pred(self, eval_x_stack):
        eval_classification_logits = hidden_forward(eval_x_stack, self.out_w_c, self.out_b_c)

        pay_amt = tf.reduce_sum(tf.nn.softmax(eval_classification_logits) * (tf.cast(tf.ones_like(
            eval_classification_logits), tf.float32) * (tf.cast(2 ** tf.range(self.class_num), tf.float32) +
                                                        tf.cast(2 ** tf.range(1, 1 + self.class_num), tf.float32) - 3) /
                                                                             2), axis=-1)

        eval_logits = hidden_forward(eval_x_stack, self.out_w, self.out_b)
        self.eval_preds_whale = tf.sigmoid(eval_logits[:, 0], name='whale_prediction_node')

        self.eval_preds_ltv = tf.identity(pay_amt, name='predictionNode')