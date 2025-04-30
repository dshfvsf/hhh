from __future__ import print_function
from __future__ import division

import pickle
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

from train.models.tf_loss import loss_choose

_ver = sys.version_info
is_py3 = (_ver[0] == 3)


def sim_gate_func(cos_sim, w, b):
    return tf.sigmoid(w * cos_sim - b) / tf.sigmoid(w - b)


def cosine_distance(tensor1, tensor2):
    tensor1_norm = tf.sqrt(tf.reduce_sum(tf.square(tensor1), axis=-1))
    tensor2_norm = tf.sqrt(tf.reduce_sum(tf.square(tensor2), axis=-1))

    tensor1_tensor2 = tf.reduce_sum(tf.multiply(tensor1, tensor2), axis=-1)
    return tensor1_tensor2 / (tensor1_norm * tensor2_norm)


def init_srn(init_argv,
             num_bins: int = 10,
             embedding_size: int = 90,
             sim_gate_w: float = 10.0,
             sim_gate_b: float = 9.0,
             sim_gate_trainable: bool = True,
             gru_units: int = None,
             binning_type: str = 'default',
             **kwargs
             ):
    init_acts = [('bins_embed', [num_bins, embedding_size], 'random')]
    var_map, log = init_var_map(init_argv, init_acts)
    sim_bins = tf.Variable(var_map['bins_embed'], validate_shape=False)
    interval = 2.0 / num_bins
    gru = tf.keras.layers.GRU(gru_units) if gru_units is not None else None

    if sim_gate_trainable:
        sim_gate_w = tf.Variable(sim_gate_w, validate_shape=False)
        sim_gate_b = tf.Variable(sim_gate_b, validate_shape=False)
    else:
        sim_gate_w = tf.constant(sim_gate_w)
        sim_gate_b = tf.constant(sim_gate_b)
    return [num_bins, embedding_size, sim_bins, interval, sim_gate_w, sim_gate_b, gru, binning_type]


def srn(inputs, mask_new, srn_var):
    num_bins, bins_bs, sim_bins, interval, sim_gate_w, sim_gate_b, gru, binning_type = srn_var

    target, sequence = inputs

    cos_sim = cosine_distance(target, sequence)
    if binning_type == 'normalize':
        if mask_new[1] is not None:
            mask_ = tf.squeeze(mask_new[1], axis=-1)
            if mask_.dtype != tf.bool:
                mask_ = tf.cast(mask_, tf.bool)

            cos_sim_mask = tf.where(mask_, cos_sim, tf.ones_like(cos_sim, cos_sim.dtype) * -1e9)
            max_val = tf.tile(tf.expand_dims(tf.reduce_max(cos_sim_mask, axis=-1), axis=1), [1, cos_sim.shape[-1]])

            cos_sim_mask = tf.where(mask_, cos_sim, tf.ones_like(cos_sim, cos_sim.dtype) * 1e9)
            min_val = tf.tile(tf.expand_dims(tf.reduce_min(cos_sim_mask, axis=-1), axis=1), [1, cos_sim.shape[-1]])
        else:
            max_val = tf.tile(tf.expand_dims(tf.reduce_max(cos_sim, axis=-1), axis=1), [1, cos_sim.shape[-1]])
            min_val = tf.tile(tf.expand_dims(tf.reduce_min(cos_sim, axis=-1), axis=1), [1, cos_sim.shape[-1]])

        cos_sim = 2.0 * (cos_sim - min_val) / (max_val - min_val + 1e-6) - 1.0

    cos_sim_idx = tf.cast((cos_sim + 1) / interval, tf.int64)
    cos_sim_idx = tf.clip_by_value(cos_sim_idx, 0, num_bins - 1)
    bin_embed = tf.gather(sim_bins, cos_sim_idx, axis=0)
    bin_embed = tf.reshape(bin_embed, [-1, cos_sim_idx.shape[1], bins_bs])

    if mask_new[1] is not None:
        combine_embed = tf.multiply(bin_embed, mask_new[1])
    else:
        combine_embed = bin_embed

    if sim_gate_w is not None and sim_gate_b is not None:
        sim_weight = sim_gate_func(cos_sim, sim_gate_w, sim_gate_b)
        sim_weight = tf.tile(tf.expand_dims(sim_weight, axis=2), [1, 1, bins_bs])
        combine_embed = tf.multiply(combine_embed, sim_weight)

    combine_embed = tf.reduce_mean(combine_embed, axis=1)

    if gru is not None:
        gru_out = gru(bin_embed)
        combine_embed = tf.concat([combine_embed, gru_out], axis=-1)
    return combine_embed


def local_split_embedding(embed_v, id_hldr, wt_hldr, multi_hot_flags, multi_hot_variable_len):
    mask_ = tf.expand_dims(wt_hldr, 2)
    if sum(multi_hot_flags) > 0:
        one_hot_v, multi_hot_v = split_param(embed_v, id_hldr, multi_hot_flags)
        one_hot_mask, multi_hot_mask = split_mask(mask_, multi_hot_flags, multi_hot_variable_len)
    else:
        one_hot_v = tf.gather(embed_v, id_hldr)
        one_hot_mask = mask_
        multi_hot_v, multi_hot_mask = None, None
    return [one_hot_v, one_hot_mask, multi_hot_v, multi_hot_mask]


def pretrained_embedding(init_argv,
                         max_index=None,
                         file_path=None,
                         feature_name=None,
                         feature_map_path=None,
                         dtype='float32',
                         field_sep="|",
                         value_sep=",",
                         output_dim=128,
                         dnn_units=None,
                         trainable=False,
                         **kwargs
                         ):
    feature_map = pd.read_csv(feature_map_path, delimiter='\t', header=None)
    feature_map.columns = ['feature_value', 'id']
    filter_df = feature_map[feature_map.feature_value.apply(lambda x: x.startswith(feature_name))]
    filter_df['feature_value'] = filter_df.feature_value.apply(lambda x: x.split(',')[1])
    feature_key = filter_df['feature_value'].to_numpy().tolist()
    #
    feature_value = filter_df['id'].to_numpy().tolist()
    new_feature_map = {k: int(v) for k, v in zip(feature_key, feature_value)}
    max_index = max_index or max(feature_value)
    # emb初始化方法
    if init_argv[0] == 'uniform' and dtype.startswith("float"):
        low, high = init_argv[1:3]
        embedding = np.random.uniform(low, high, size=(max_index + 1, output_dim))
    elif dtype.startswith("int"):
        embedding = np.zeros((max_index + 1, output_dim), dtype=dtype)
    else:  # use normal distribution
        loc, scale = init_argv[1:3]
        embedding = np.random.normal(loc, scale, size=(max_index + 1, output_dim))
    # 预训练emb的文件位置
    if file_path:
        cnt = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for idx, emb_line in enumerate(f):
                field_list = emb_line.rstrip("\n").split(field_sep)
                field_count = len(field_list)
                if field_list[0] not in new_feature_map:
                    continue
                if field_count == 2:
                    idx = new_feature_map[field_list[0]]
                    if value_sep == 'list':
                        emb_arr = np.array(eval(field_list[1]), dtype=dtype)[:output_dim]
                    else:
                        emb_arr = np.fromstring(field_list[1], dtype=dtype, sep=value_sep)[:output_dim]
                elif field_count == 1:
                    emb_arr = np.fromstring(field_list[0], dtype=dtype, sep=value_sep)[:output_dim]
                else:
                    raise ValueError(f"The length of embedding fields is neither 1 nor 2: {field_list}")
                embedding[idx] = emb_arr
                cnt += 1
        print(f"pretrained_embedding {feature_name} {cnt}/{len(new_feature_map)}")
    name = kwargs.get('name', feature_name)
    if trainable:
        embedding = tf.Variable(embedding, dtype=dtype, name=name + '_pretrain_emb')
    else:
        embedding = tf.constant(embedding, dtype=dtype, name=name + '_pretrain_emb')
    for units in dnn_units:
        embedding = tf.layers.dense(embedding, units)
    return embedding


def scaled_dot_product_attention(q, k, v, query_masks, key_masks,
                                 causality=False, dropout_rate=0.,
                                 training=True,
                                 scope="scaled_dot_product_attention"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d_k = q.get_shape().as_list()[-1]

        # dot product
        outputs = tf.matmul(q, tf.transpose(k, [0, 2, 1]))  # (hN, T_q, T_k)

        # scale
        outputs /= d_k ** 0.5

        # key masking
        if key_masks is not None:
            outputs = mask(outputs, key_masks=key_masks, type_="key")

        # causality or future blinding masking
        if causality:
            outputs = mask(outputs, type_="future")

        # softmax
        outputs = tf.nn.softmax(outputs)
        attention = tf.transpose(outputs, [0, 2, 1])

        # dropout
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)

        # weighted sum (context vectors)
        outputs = tf.matmul(outputs, v)

    return outputs


def ln(inputs, epsilon=1e-8, scope="ln", gamma_=None, beta_=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        try:
            normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        except ZeroDivisionError:
            print("You can't divide by 0!")
        if gamma_ is None:
            outputs = gamma * normalized + beta
        else:
            outputs = gamma_ * gamma * normalized + beta_ + beta

    return outputs


def mask(inputs, query_masks=None, key_masks=None, type_=None):
    padding_num = -2 ** 32 + 1
    if type_ in ("k", "key", "keys"):
        key_masks = tf.to_float(key_masks)
        key_masks = tf.tile(key_masks, [tf.shape(inputs)[0] // tf.shape(key_masks)[0], 1])  # (h*N, seqlen)
        key_masks = tf.expand_dims(key_masks, 1)  # (h*N, 1, seqlen)
        outputs = inputs + key_masks * padding_num
    elif type_ in ("q", "query", "queries"):
        query_masks = tf.cast(query_masks, tf.bool)
        num_heads = tf.shape(inputs)[0] // tf.shape(query_masks)[0]
        query_masks = tf.tile(query_masks, [num_heads, 1])
        query_masks = tf.expand_dims(query_masks, -1)
        query_masks = tf.tile(query_masks, [1, 1, tf.shape(key_masks)[1]])
        paddings = tf.ones_like(inputs) * padding_num
        outputs = tf.where(query_masks, inputs, paddings)
    elif type_ in ("f", "future", "right"):
        diag_vals = tf.ones_like(inputs[0, :, :])
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
        future_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])

        paddings = tf.ones_like(future_masks) * padding_num
        outputs = tf.where(tf.equal(future_masks, 0), paddings, inputs)
    else:
        print("Check if you entered type correctly!")

    return outputs


def mask_old(inputs, key_masks=None, type_1=None):
    padding_num = -2 ** 32 + 1
    if type_1 in ("k", "key", "keys"):
        key_masks = tf.to_float(key_masks)
        key_masks = tf.tile(key_masks, [tf.shape(inputs)[0] // tf.shape(key_masks)[0], 1])
        key_masks = tf.expand_dims(key_masks, 1)
        outputs = inputs + key_masks * padding_num
    elif type_1 in ("f", "future", "right"):
        diag_vals = tf.ones_like(inputs[0, :, :])
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
        future_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])

        paddings = tf.ones_like(future_masks) * padding_num
        outputs = tf.where(tf.equal(future_masks, 0), paddings, inputs)
    else:
        print("Check if you entered type correctly!")

    return outputs


def hstu_attention(queries, keys, values, gating_weights,
                   key_masks=None,
                   gamma_=None,
                   beta_=None,
                   num_heads=1,
                   linear_activation="silu",
                   dropout_rate=0,
                   training=True,
                   causality=False,
                   concat_ua=False,
                   normalization="rel_bias",
                   scope="hstu_attention"):
    d_model = queries.get_shape().as_list()[-1]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Linear projections)

        q = tf.layers.dense(queries, d_model, use_bias=True)  # (N, T_q, d_model)
        k = tf.layers.dense(keys, d_model, use_bias=True)  # (N, T_k, d_model)
        v = tf.layers.dense(values, d_model, use_bias=True)  # (N, T_k, d_model)
        u = tf.layers.dense(gating_weights, d_model, use_bias=True)  # (N, T_k, d_model)

        q = tf.nn.sigmoid(q) * q
        k = tf.nn.sigmoid(k) * k
        v = tf.nn.sigmoid(v) * v
        u = tf.nn.sigmoid(u) * u

        q_ = tf.concat(tf.split(q, num_heads, axis=2), axis=0)  # (h*N, T_q, d_model/h)
        k_ = tf.concat(tf.split(k, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)
        v_ = tf.concat(tf.split(v, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)
        u_ = tf.concat(tf.split(u, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)

        print('---------------------------test-------------------------  ')
        print('self.q                    :', q_)
        print('self.k                    :', k_)
        print('self.v                    :', v_)
        print('self.u                    :', u_)
        print('---------------------------test-------------------------  ')

        qk_attn = tf.einsum('bnd,bmd->bnm', q_, k_)  # (h*N, T_q, T_k)
        print('---------------------------test-------------------------  ')
        print('self.qk_attn[idx]               :', qk_attn)
        print('---------------------------test-------------------------  ')
        # relative attention bias

        if normalization == "rel_bias":
            # 将上方代码的silu替换为 tf.nn.sigmoid（x） * x
            qk_attn = tf.nn.sigmoid(qk_attn) * qk_attn / d_model
        elif normalization == "softmax_rel_bias":
            qk_attn = tf.nn.softmax(qk_attn / np.sqrt(d_model), axis=-1)
        print('---------------------------test-------------------------  ')
        print('self.qk_attn[idx]               :', qk_attn)
        print('---------------------------test-------------------------  ')
        attn_output = tf.matmul(qk_attn, v_)  # tf.einsum('bnm,bmd->bnd', qk_attn, v)
        print('---------------------------test-------------------------  ')
        print('self.attn_output[idx]               :', attn_output)
        print('---------------------------test-------------------------  ')
        # FFN + dropout & add
        if concat_ua:
            a = tf.contrib.layers.layer_norm(attn_output)
            o_input = tf.concat([u_, a, u_ * a], axis=-1)
        else:
            o_input = u_ * tf.contrib.layers.layer_norm(attn_output)

        # Restore shape
        o_input = tf.concat(tf.split(o_input, num_heads, axis=0), axis=2)

        outputs = tf.layers.dense(tf.layers.dropout(o_input, dropout_rate, training), d_model) + queries

        # Normalize
        if gamma_ is None:
            outputs = ln(outputs)
        else:
            outputs = ln(outputs, scope=scope + '_conditional_ln', gamma_=gamma_, beta_=beta_)

    return outputs


def multihead_attention(queries, keys, values, queries_length, keys_length,
                        key_masks=None,
                        gamma_=None,
                        beta_=None,
                        num_heads=8,
                        dropout_rate=0,
                        training=True,
                        causality=False,
                        scope="multihead_attention"):
    d_model = queries.get_shape().as_list()[-1]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Linear projections
        q = tf.layers.dense(queries, d_model, use_bias=True)  # (N, T_q, d_model)
        k = tf.layers.dense(keys, d_model, use_bias=True)  # (N, T_k, d_model)
        v = tf.layers.dense(values, d_model, use_bias=True)  # (N, T_k, d_model)

        # Split and concat
        q_ = tf.concat(tf.split(q, num_heads, axis=2), axis=0)  # (h*N, T_q, d_model/h)
        k_ = tf.concat(tf.split(k, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)
        v_ = tf.concat(tf.split(v, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)

        # Attention
        if key_masks is None:
            outputs = scaled_dot_product_attention(q_, k_, v_, None, None, causality, dropout_rate, training)
        else:
            outputs = scaled_dot_product_attention(q_, k_, v_, None, key_masks, causality, dropout_rate, training)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)

        # Residual connection
        outputs += queries

        # Normalize
        if gamma_ is None:
            outputs = ln(outputs)
        else:
            outputs = ln(outputs, scope=scope + '_conditional_ln', gamma_=gamma_, beta_=beta_)

    return outputs


def ff(inputs, num_units, scope="positionwise_feedforward", gamma_=None, beta_=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        print('ff', inputs)
        # Inner layer
        outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)
        # Outer layer
        outputs = tf.layers.dense(outputs, num_units[1])
        # Residual connection
        outputs += inputs
        # Normalize
        if gamma_ is None:
            outputs = ln(outputs)
        else:
            outputs = ln(outputs, scope=scope + '_conditional_ln', gamma_=gamma_, beta_=beta_)

    return outputs


def positional_encoding(inputs,
                        maxlen,
                        masking=False,
                        scope="positional_encoding"):
    e = inputs.get_shape().as_list()[-1]
    n, t = tf.shape(inputs)[0], tf.shape(inputs)[1]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        position_ind = tf.tile(tf.expand_dims(tf.range(t), 0), [n, 1])
        try:
            position_enc = np.array([
                [pos / np.power(10000, (i - i % 2) / e) for i in range(e)]
                for pos in range(maxlen)])
        except Exception:
            print('zero')

        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])
        position_enc = tf.convert_to_tensor(position_enc, tf.float32)

        # lookup
        outputs = tf.nn.embedding_lookup(position_enc, position_ind)
        zeros = tf.zeros_like(outputs, dtype=tf.float32)

        # masks
        if masking:
            outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

        return tf.to_float(outputs)


def positional_encoding_learn(inputs,
                              maxlen,
                              masking=False,
                              scope="positional_encoding_learn"):
    e = inputs.get_shape().as_list()[-1]
    n, t = tf.shape(inputs)[0], tf.shape(inputs)[1]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        position_ind = tf.tile(tf.expand_dims(tf.range(t), 0), [n, 1])
        position_enc = tf.get_variable("embedding_position_learn", [maxlen, e],
                                       initializer=tf.contrib.layers.xavier_initializer())

        outputs = tf.nn.embedding_lookup(position_enc, position_ind)
        if masking:
            outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

        return tf.to_float(outputs)


def get_params_count():
    return np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])


def normalize(norm, x_value, num_inputs):
    if norm:
        with tf.name_scope('norm'):
            try:
                ret_ = x_value / np.sqrt(num_inputs)
            except ZeroDivisionError:
                print("You can't divide by 0!")
            return ret_
    else:
        return x_value


def build_optimizer(_ptmzr_argv, loss, var_list=None):
    _ptmzr = _ptmzr_argv[0]
    if _ptmzr == 'adam':
        # is adam
        _learning_rate, _epsilon = _ptmzr_argv[1:3]
        ptmzr = tf.train.AdamOptimizer(learning_rate=_learning_rate,
                                       epsilon=_epsilon).minimize(loss)
        # ptmzr done
        log = 'optimizer: %s, learning rate: %g, epsilon: %g' % \
              (_ptmzr, _learning_rate, _epsilon)
    elif _ptmzr == 'adagrad':
        # is adagrad
        _learning_rate, _initial_accumulator_value = _ptmzr_argv[1:3]
        ptmzr = tf.train.AdagradOptimizer(
            learning_rate=_learning_rate,
            initial_accumulator_value=_initial_accumulator_value).minimize(loss)
        # ptmzr final done
        log = 'optimizer: %s, learning rate: %g, init_accumulator_value: %g' % (
            _ptmzr, _learning_rate, _initial_accumulator_value)
    elif _ptmzr == 'ftrl':
        # is ftrl
        _learning_rate, learning_rate_power, init_accum, lambda_1, lambda_2 = _ptmzr_argv[1:6]
        ptmzr = tf.train.FtrlOptimizer(
            learning_rate=_learning_rate,
            learning_rate_power=learning_rate_power,
            initial_accumulator_value=init_accum,
            l1_regularization_strength=lambda_1,
            l2_regularization_strength=lambda_2).minimize(loss)
        # l1,l2 norm done
        log = ('optimizer: %s, learning rate: %g, initial accumulator: %g, '
               'l1_regularization: %g, l2_regularization: %g' %
               (_ptmzr, _learning_rate, init_accum, lambda_1, lambda_2))
    else:
        # the other
        _learning_rate = _ptmzr_argv[1]
        ptmzr = tf.train.GradientDescentOptimizer(
            learning_rate=_learning_rate).minimize(loss)
        log = 'optimizer: %s, learning rate: %g' % (_ptmzr, _learning_rate)
    return ptmzr, log


def init_params(params):
    if params["init_method"][0] == "uniform":
        params_init = tf.random_uniform_initializer(minval=params["init_method"][1],
                                                    maxval=params["init_method"][2])
    elif params["init_method"][0] == "normal":
        params_init = tf.random_normal_initializer()
    elif params["init_method"][0] == "xavier":
        params_init = tf.glorot_uniform_initializer()
    else:
        raise Exception("init method %s not found" % (params["init_method"]))
    return params_init


def activate(act_func, x_value):
    if act_func == 'tanh':
        return tf.tanh(x_value)
    elif act_func == 'relu':
        return tf.nn.relu(x_value)
    elif act_func == 'softmax':
        return tf.nn.softmax(x_value)
    else:
        return tf.sigmoid(x_value)


def _normal_init_var_map(_init_argv, init_vars, var_map, log):
    _mean, _stddev, _seeds = _init_argv[1:-1]
    log += 'init method: %s(mean=%g, stddev=%g), seeds: %s\n' % (
        _init_argv[0], _mean, _stddev, str(_seeds))
    _j = 0
    for _i in range(len(init_vars)):
        key, shape, action = init_vars[_i]
        # iterator key
        if key not in var_map.keys():
            if action == 'random' or action == "normal":
                # action is random or normal
                print('%s normal(mean=%g, stddev=%g) random init' %
                      (key, _mean, _stddev))
                var_map[key] = tf.random_normal(shape, _mean, _stddev,
                                                seed=_seeds[_j % 10])
                # var map and random done
                _j += 1
            elif action == 'zero':
                var_map[key] = tf.zeros(shape)
            elif action == 'one':
                var_map[key] = tf.ones(shape)
            else:
                # the other 
                var_map[key] = tf.zeros(shape)
        else:
            print('%s already set' % key)
    return var_map, log


def _uniform_init_var_map(_init_argv, init_vars, var_map, log):
    _min_val, _max_val, _seeds = _init_argv[1:-1]
    log += 'init method: %s(minval=%g, maxval=%g), seeds: %s\n' % (
        _init_argv[0], _min_val, _max_val, str(_seeds))
    _j = 0
    for _i in range(len(init_vars)):
        key, shape, action = init_vars[_i]
        if key not in var_map.keys():
            if action == 'random' or action == 'uniform':
                # the action is random or uniform
                print('%s the uniform random init, ' % key,
                      "(minval=%g, maxval=%g)\nseeds: %s" % (
                          _min_val, _max_val, str(_seeds)))
                var_map[key] = tf.random_uniform(
                    shape, _min_val, _max_val,
                    seed=_seeds[_j % len(_seeds)])
                # map done
                _j += 1
            elif action == 'zero':
                var_map[key] = tf.zeros(shape)
                _j += 1
            elif action == 'one':
                var_map[key] = tf.ones(shape)
                _j += 1
            else:
                # the other 
                var_map[key] = tf.zeros(shape)
                _j += 1
        else:
            print('%s the already set' % key)
    return var_map, log


def init_var_map(_init_argv, init_vars):
    _init_path = _init_argv[-1]
    if _init_path:
        var_map_f_ = pickle.load(open(_init_path, 'rb'))
        log_f = 'init model from: %s, ' % _init_path
    else:
        var_map_f = {}
        log_f = 'random init, '
        _init_method = _init_argv[0]

        if _init_method == 'normal':
            # the normal
            var_map_f_, log_f = _normal_init_var_map(_init_argv, init_vars, var_map_f, log_f)
        else:
            # the other
            var_map_f_, log_f = _uniform_init_var_map(_init_argv, init_vars, var_map_f, log_f)

    return var_map_f_, log_f


def layer_normalization(_input_tensor, gain, biase, epsilon=1e-5):
    layer_mean, layer_variance = tf.nn.moments(_input_tensor, [1], keep_dims=True)
    try:
        layer_norm_input = (_input_tensor - layer_mean) / tf.sqrt(layer_variance + epsilon)
    except ZeroDivisionError:
        print("You can't divide by 0!")
    return layer_norm_input * gain + biase


def split_mask(mask_, multi_hot_flags, num_multihot):
    multi_hot_mask = tf.transpose(
        tf.boolean_mask(tf.transpose(mask_, [1, 0, 2]),
                        multi_hot_flags),
        [1, 0, 2])
    # done tmp
    mul_mask_list = tf.split(multi_hot_mask, num_multihot, axis=1)
    mul_mask_list_proc = []
    for mul_mask_tmp in mul_mask_list:
        sum_mul_mask = tf.reduce_sum(mul_mask_tmp, 1, keep_dims=True)
        sum_mul_mask = tf.maximum(sum_mul_mask, tf.ones_like(sum_mul_mask))
        mul_mask_tmp /= sum_mul_mask
        mul_mask_list_proc.append(mul_mask_tmp)
    # loop done
    multi_hot_mask = tf.concat(mul_mask_list_proc, axis=1)
    multi_hot_mask.set_shape((None, sum(multi_hot_flags), None))

    one_hot_flags = [not flag for flag in multi_hot_flags]
    one_hot_mask = tf.transpose(
        tf.boolean_mask(tf.transpose(mask_, [1, 0, 2]),
                        one_hot_flags),
        [1, 0, 2])
    one_hot_mask.set_shape((None, sum(one_hot_flags), None))
    return one_hot_mask, multi_hot_mask


def split_param(model_param, id_hldr, multi_hot_flags, hash_tables=None):
    one_hot_flags = [not flag for flag in multi_hot_flags]
    if hash_tables is not None:
        batch_param = tf.gather(model_param, hash_tables.lookup(id_hldr))
    else:
        batch_param = tf.gather(model_param, id_hldr)
    batch_param = tf.transpose(batch_param, [1, 0, 2])
    print('batch_param & shape', batch_param, tf.shape(batch_param))
    batch_one_hot_param = tf.transpose(
        tf.boolean_mask(batch_param, one_hot_flags),
        [1, 0, 2])
    batch_one_hot_param.set_shape((None, sum(one_hot_flags), None))
    # batch one hot done
    batch_multi_hot_param = tf.transpose(
        tf.boolean_mask(batch_param, multi_hot_flags),
        [1, 0, 2])
    batch_multi_hot_param.set_shape((None, sum(multi_hot_flags), None))
    return batch_one_hot_param, batch_multi_hot_param


def split_param_4d(model_param, id_hldr, multi_hot_flags):
    one_hot_flags = [not flag for flag in multi_hot_flags]
    batch_param = tf.gather(model_param, id_hldr)
    batch_param = tf.transpose(batch_param, [1, 0, 2, 3])
    batch_one_hot_param = tf.transpose(
        tf.boolean_mask(batch_param, one_hot_flags),
        [1, 0, 2, 3])
    batch_one_hot_param.set_shape((None, sum(one_hot_flags), None, None))
    batch_multi_hot_param = tf.transpose(
        tf.boolean_mask(batch_param, multi_hot_flags),
        [1, 0, 2, 3])
    batch_multi_hot_param.set_shape((None, sum(multi_hot_flags), None, None))
    return batch_one_hot_param, batch_multi_hot_param


def sum_multi_hot(batch_multi_hot_param, multi_hot_mask, num_multihot, is_reduce=True):
    # param_masked
    param_masked = tf.multiply(batch_multi_hot_param, multi_hot_mask)
    # param_list
    param_list = tf.split(param_masked, num_multihot, axis=1)
    # param_reduced_list
    param_reduced_list = []
    # for param_list
    for param_ in param_list:
        if is_reduce:
            param_reduced = tf.reduce_sum(param_, axis=1, keep_dims=True)
        else:
            param_reduced = param_
        param_reduced_list.append(param_reduced)
    return tf.concat(param_reduced_list, axis=1)


def sum_pooling(multi_hot_field_list):
    param_reduced_list = []
    for param_ in multi_hot_field_list:
        param_reduced = tf.reduce_sum(param_, axis=1, keep_dims=True)
        param_reduced_list.append(param_reduced)
    return param_reduced_list


def sum_pooling_multi_hot(multi_hot_vx, num_multihot):
    param_list = tf.split(multi_hot_vx, num_multihot, axis=1)
    param_reduced_list = []
    for param_ in param_list:
        param_reduced = tf.reduce_sum(param_, axis=1, keep_dims=True)
        param_reduced_list.append(param_reduced)
    return tf.concat(param_reduced_list, axis=1)


def short_multi_hot(batch_multi_hot_param, multi_hot_mask, num_multihot):
    param_masked = tf.multiply(batch_multi_hot_param, multi_hot_mask)
    param_list = tf.split(param_masked, num_multihot, axis=1)
    param_reduced_list = []
    for param_ in param_list:
        param_reduced = param_[:, 0:1, :]
        param_reduced_list.append(param_reduced)
    return tf.concat(param_reduced_list, axis=1)


def get_field_index(multi_hot_flags):
    cur_field_index = 0
    field_indices = []
    for i, flag in enumerate(multi_hot_flags):
        field_indices.append(cur_field_index)
        flat_two = flag and (i + 1 < len(multi_hot_flags)) and not multi_hot_flags[i + 1]
        if not flag or flat_two:
            print("cur_field_index ++")
            cur_field_index += 1
    return field_indices


def get_field_num(multi_hot_flags, multi_hot_len):
    # one_hot_flags
    one_hot_flags = [not flag for flag in multi_hot_flags]
    # one_hot_field_num
    one_hot_field_num = sum(one_hot_flags)
    if sum(multi_hot_flags) % multi_hot_len != 0:
        raise ValueError("cannot infer field number. please check input!")
    # sum done
    multi_hot_field_num = sum(multi_hot_flags) // multi_hot_len
    field_num = one_hot_field_num + multi_hot_field_num
    return field_num


def split_embedding(embed_v, id_hldr, wt_hldr, multi_hot_flags, multi_hot_variable_len):
    one_hot_vx = None
    multi_hot_vx = None
    mask_ = tf.expand_dims(wt_hldr, 2)
    if sum(multi_hot_flags) > 0:
        one_hot_v, multi_hot_v = split_param(embed_v, id_hldr, multi_hot_flags)
        one_hot_mask, multi_hot_mask = split_mask(mask_, multi_hot_flags, multi_hot_variable_len)
        one_hot_vx = tf.multiply(one_hot_v, one_hot_mask)
        multi_hot_vx = tf.multiply(multi_hot_v, multi_hot_mask)
    else:
        one_hot_vx = tf.multiply(tf.gather(embed_v, id_hldr), mask_)

    return one_hot_vx, multi_hot_vx


def get_domain_mask_noah(sub_domain_list, domain_col):
    domain_mask = None
    for sub_domain_val in sub_domain_list:
        sub_domain_id = tf.to_int64(tf.convert_to_tensor(sub_domain_val))
        if domain_mask is None:
            domain_mask = tf.equal(domain_col, sub_domain_id)
        else:
            domain_mask = tf.math.logical_or(domain_mask, tf.equal(domain_col, sub_domain_id))
    domain_mask.set_shape([None, ])
    return domain_mask


def get_domain_mask_light(sub_domain_list, domain_col):
    sub_domain_ids = tf.to_int64(tf.convert_to_tensor(sub_domain_list))
    domain_mask = tf.reduce_any(tf.equal(tf.expand_dims(domain_col, axis=1), sub_domain_ids), axis=1)
    domain_mask.set_shape([None, ])
    return domain_mask


def get_domain_loss(config, preds, labels, id_hldr):
    if not config.add_domain_loss_weight:
        raise Exception("please setting 'add_domain_loss_weight' in config.")
    domain_col = tf.gather(id_hldr, config.domain_col_idx, axis=1)
    wt_df = config.wt_df
    max_id = max(wt_df['old_id'])
    tmp = tf.fill((max_id + 1,), tf.constant(1, dtype=tf.int64))  # 0 for padding
    old2newid = dict(zip(wt_df['old_id'], wt_df['new_id']))
    old2newid = tf.tensor_scatter_nd_update(
        tmp,
        tf.reshape(tf.constant(list(old2newid.keys()), dtype=tf.int64), (-1, 1)),
        tf.constant(list(old2newid.values()), dtype=tf.int64),
        name='old2newid')
    newid2wt = dict(zip(wt_df['new_id'], wt_df['domain_wt']))
    newid2wt = tf.tensor_scatter_nd_update(
        tf.to_float(tmp),
        tf.reshape(tf.constant(list(newid2wt.keys()), dtype=tf.int64), (-1, 1)),
        tf.constant(list(newid2wt.values()), dtype=tf.float32),
        name='newid2wt')
    domain_wt = tf.gather(newid2wt, tf.gather(old2newid, domain_col))
    # ------
    loss = tf.reduce_mean(loss_choose(config, preds, labels), name='loss')
    loss = tf.multiply(loss, domain_wt)
    return loss


class STARMLP:
    def __init__(self, layers, act_func, bias=True, batch_norm=True, drop_rate=0.1, scope="mlp") -> None:
        self.layers = layers
        self.n_layers = len(layers) - 1
        self.act_func = getattr(tf.nn, act_func)
        self.bias = bias
        self.batch_norm = batch_norm
        self.drop_rate = drop_rate
        self.scope = scope
        self.var_dict = self._init_vars(layers)

    def _init_vars(self, layers):
        var_dict = {}
        for i, (n_in, n_out) in enumerate(zip(layers[:-1], layers[1:])):
            var_dict[f'{self.scope}_w_{i}'] = tf.Variable(tf.random.uniform(
                shape=(n_in, n_out),
                minval=-0.001,
                maxval=0.001,
                name=f'{self.scope}_w_{i}'
            ))
            if self.bias:
                var_dict[f'{self.scope}_b_{i}'] = tf.Variable(tf.random.uniform(
                    shape=(n_out,),
                    minval=-0.001,
                    maxval=0.001,
                    name=f'{self.scope}_b_{i}'
                ))
        return var_dict

    def forward(self, x, shared_fc_var=None, training=True):
        if shared_fc_var:
            var_dict = self.star(shared_fc_var)
        else:
            var_dict = self.var_dict
        x_ = x
        for i in range(self.n_layers):
            x_ = tf.matmul(x_, var_dict[f'{self.scope}_w_{i}'])
            if self.bias:
                x_ = x_ + var_dict[f'{self.scope}_b_{i}']
            if i != self.n_layers - 1:
                x_ = self.act_func(x_)
            if self.batch_norm:
                x_ = tf.layers.batch_normalization(
                    x_,
                    training=training,
                    reuse=tf.AUTO_REUSE,
                    name=f'{self.scope}_bn_{i}'
                )
            x_ = tf.nn.dropout(x_, rate=self.drop_rate)
        return x_

    def star(self, shared_fc_var):
        var_dict = {}
        for k, v in shared_fc_var.items():
            try:
                if "_w_" in k:
                    suffix = '_w_' + k.split('_w_')[1]
                    name = self.scope + suffix
                    var_dict[name] = tf.multiply(v, self.var_dict[name])
                else:
                    suffix = '_b_' + k.split('_b_')[1]
                    name = self.scope + suffix
                    var_dict[name] = tf.add(v, self.var_dict[name])
            except KeyError as e:
                print(f"Warning: there is no key in the var_dict, use shared parameters only. {e}")
                var_dict[name] = v
        return var_dict


def extract_embedding_from_pb(config):
    from tensorflow.python.platform import gfile
    from tensorflow.python.framework import tensor_util
    config_ = tf.ConfigProto()
    sess = tf.Session(config=config_)
    with gfile.FastGFile(config.PRE_MODEL_DIR, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        opnode = [tensor for tensor in tf.get_default_graph().as_graph_def().node if tensor.op == 'Const']
    # 获取输入tensor
    emb_pre = tensor_util.MakeNdarray(opnode[config.EMB_INDEX].attr['value'].tensor)
    emb_save_dir = os.path.join(config.DATA_DIR, 'pre_emb_save')
    np.save(emb_save_dir, emb_pre)
    sess.close()
    tf.reset_default_graph()


def get_filed_map(one_hot_flags, multi_hot_variable_len, multi_hot_flags, filed_names):
    num_onehot = sum(one_hot_flags)
    num_multihot = len(multi_hot_variable_len)
    pos = 0
    multi_pos = 0
    filed_range = {}
    filed_length = {}
    num_filed = 0
    for _ in range(num_onehot + num_multihot):
        if not multi_hot_flags[pos]:
            filed_range[filed_names[num_filed]] = [pos, pos + 1]  # 左闭右开
            filed_length[filed_names[num_filed]] = 1
            pos += 1
            num_filed += 1
        else:
            filed_range[filed_names[num_filed]] = [pos, pos + multi_hot_variable_len[multi_pos]]
            filed_length[filed_names[num_filed]] = multi_hot_variable_len[multi_pos]
            pos += multi_hot_variable_len[multi_pos]
            multi_pos += 1
            num_filed += 1
    return filed_range, filed_length


def get_pretrain_weight(init_argv, dtype, file_path, value_feature_id_map, field_sep, value_sep, pretrain_name,
                        trainable, dnn_units, max_feat_id):
    """

    :param init_argv:  初始化参数
    :param dtype: 嵌入类别
    :param file_path: 文件路径
    :param value_feature_id_map:  特征值对应feature id 的数组（可能多对一）
    :param field_sep:  pretrian 文件中value和vector 的分隔符
    :param value_sep:  pretrian 文件中vector 的分隔符
    :param pretrain_name: 当前pretrian的名字
    :param trainable: 是否训练
    :param dnn_units: dnn转化 [dim] 为空表示不进行dnn
    :param max_feat_id:  feature_map 的最大id值
    :return:
    """

    with open(file_path, "r", encoding="utf-8") as f:
        # 获取预训练编码信息
        value_num = sum(1 for _ in f)
        max_index = value_num

    with open(file_path, "r", encoding="utf-8") as f:
        first_line = f.readline().rstrip("\n")
        output_dim = len(first_line.split(field_sep)[1].split(value_sep))

    with open(file_path, "r", encoding="utf-8") as f:
        # 初始化编码矩阵
        if init_argv[0] == 'uniform' and dtype.startswith("float"):
            low, high = init_argv[1:3]
            embedding = np.random.uniform(low, high, size=(max_index + 1, output_dim))
        elif dtype.startswith("int"):
            embedding = np.zeros((max_index + 1, output_dim), dtype=dtype)
        else:  # use normal distribution
            loc, scale = init_argv[1:3]
            embedding = np.random.normal(loc, scale, size=(max_index + 1, output_dim))

        # 初始化featureid2pretrainedid映射
        feature_pretrained_id_map = np.zeros((max_feat_id + 1, 1), dtype='int32')

        # 将预训练向量填入初始化矩阵，从第1位开始，第0位是未匹配的默认值
        cnt = 0
        for idx, emb_line in enumerate(f):
            field_list = emb_line.rstrip("\n").split(field_sep)
            field_count = len(field_list)
            # emb初始化方法
            if field_list[0] in value_feature_id_map:
                for f_idx in value_feature_id_map[field_list[0]]:
                    feature_pretrained_id_map[f_idx] = idx + 1
            pre_idx = idx + 1
            if field_count == 2:
                if value_sep == 'list':
                    emb_arr = np.array(eval(field_list[1]), dtype=dtype)[:output_dim]
                else:
                    emb_arr = np.fromstring(field_list[1], dtype=dtype, sep=value_sep)[:output_dim]
            else:
                raise ValueError(f"The length of embedding fields is not 2: {field_list}")
            embedding[pre_idx] = emb_arr
            cnt += 1

    name = pretrain_name
    if trainable:
        embedding = tf.Variable(embedding, dtype=dtype, name=name + '_pretrain_emb')
    else:
        embedding = tf.constant(embedding, dtype=dtype, name=name + '_pretrain_emb')
    for units in dnn_units:
        embedding = tf.layers.dense(embedding, units)
    feature_pretrained_id_map = tf.constant(feature_pretrained_id_map, dtype='int32',
                                            name=name + '_feat_pretrain_id_map')

    return embedding, feature_pretrained_id_map, output_dim


def pretrained_embedding_v2(init_argv,
                            file_path_list=None,
                            pretrain_names=None,
                            feature_map_path=None,
                            dtype=None,
                            field_sep=None,
                            value_sep=None,
                            dnn_units=None,
                            trainable=False,
                            **kwargs
                            ):
    """

    :param init_argv:
    :param file_path_list: 所有预训练文件的地址
    :param pretrain_name:  所有预训练的名称
    :param feature_map_path:  特征工程featuremap地址
    :param dtype: 所有预训练矩阵的dtype
    :param field_sep: 预训练文件区分value和向量的符号
    :param value_sep:  预训练向量值的分割符
    :param dnn_units: [[d1,d2]] 所有预训练接dense的情况，[]表示不接
    :param trainable: 向量是否可以学习
    :param kwargs:
    :return: 三个字典，分别存放 pretrain_name 对应的 embedding矩阵、featureid2preid的转化变量，以及dim
    """
    feature_map = pd.read_csv(feature_map_path, delimiter='\t', header=None)  # 特征工程输出的特征映射文件
    feature_map.columns = ['feature_value', 'id']

    # 所有的值对应id map 可能是一对多
    feature_map['feature_value'] = feature_map.feature_value.apply(lambda x: x.split(',')[1])
    feature_key = feature_map['feature_value'].to_numpy().tolist()

    feature_value = feature_map['id'].to_numpy().tolist()
    value_feature_id_map = {}
    max_feat_id = 0
    for i, feat_value in enumerate(feature_value):
        if feature_key[i] not in value_feature_id_map:
            value_feature_id_map[feature_key[i]] = [feat_value]
            max_feat_id = max(max_feat_id, feat_value)
        else:
            value_feature_id_map[feature_key[i]].append(feature_value[i])

    pretrained_embedding_dict = {}
    pretrained_embedding_map_dict = {}
    pretrained_embedding_dim_dict = {}
    for i, file_path in enumerate(file_path_list):
        (pretrained_embedding_dict[pretrain_names[i]],
         pretrained_embedding_map_dict[pretrain_names[i]],
         pretrained_embedding_dim_dict[pretrain_names[i]]) = get_pretrain_weight(init_argv, dtype[i], file_path,
                                                                                 value_feature_id_map, field_sep[i],
                                                                                 value_sep[i], pretrain_names[i],
                                                                                 trainable[i], dnn_units[i],
                                                                                 max_feat_id)
    return pretrained_embedding_dict, pretrained_embedding_map_dict, pretrained_embedding_dim_dict


def extend_param(model, sess, new_input_dim):
    ori_embed_v = sess.run(model.embed_v)
    print("ori_embed_v = \n%s\n ...\n%s\nshape = %s" % (ori_embed_v[0], ori_embed_v[-1], ori_embed_v.shape))
    ori_input_dim = ori_embed_v.shape[0]
    if new_input_dim < ori_input_dim:
        raise ValueError("new_input_dim %d is smaller than ori_input_dim %d" % (
            new_input_dim, ori_input_dim))
    if new_input_dim == ori_input_dim:
        return
    print("init params(for embed_v): ", model.init_argv)
    np.random.seed(model.init_argv[3])
    if model.init_argv[0] == 'uniform':
        low, high = model.init_argv[1:3]
        new_feature_init_embed_v = np.random.uniform(low, high,
                                                     size=(new_input_dim - ori_input_dim, model.embedding_size))
    # use normal distribution
    else:
        loc, scale = model.init_argv[1:3]
        new_feature_init_embed_v = np.random.normal(loc, scale,
                                                    size=(new_input_dim - ori_input_dim, model.embedding_size))
    print("new init feature embed shape: ", new_feature_init_embed_v.shape)
    new_embed_v = np.vstack((ori_embed_v, new_feature_init_embed_v))
    print("new_embed_v = \n%s\n ...\n%s\nshape = %s" % (new_embed_v[0], new_embed_v[-1], new_embed_v.shape))
    extend_fm_v = tf.assign(model.embed_v, new_embed_v, validate_shape=False)
    sess.run(extend_fm_v)


def deep_forward(vx_embed, h_w, h_b, act_func, keep_prob, training, batch_norm=False):
    hidden_output = vx_embed
    for i, _ in enumerate(h_w):
        hidden_output = tf.matmul(activate(act_func, hidden_output), h_w[i]) + h_b[i]
        if batch_norm:
            print("setting bn for training stage")
            hidden_output = tf.layers.batch_normalization(
                hidden_output, training=training, reuse=not training, name="bn_%d" % i
            )
        if training:
            hidden_output = tf.nn.dropout(hidden_output, keep_prob=keep_prob)

    return hidden_output


def hidden_forward(final_layer, out_w_h, out_b_h):
    hidden_output = tf.matmul(final_layer, out_w_h) + out_b_h
    return hidden_output


def compute_embedding_dim(model):
    model.num_onehot = sum(model.one_hot_flags)
    if model.config.DYNAMIC_LENGTH:
        model.num_multihot = len(model.multi_hot_variable_len)
    else:
        if is_py3:
            model.num_multihot = sum(model.multi_hot_flags) // model.multi_hot_len
        else:
            model.num_multihot = sum(model.multi_hot_flags) / model.multi_hot_len \
                if model.multi_hot_len != 0 else sum(model.multi_hot_flags)

    if model.merge_multi_hot:
        if is_py3:
            model.embedding_dim = int((model.num_multihot + model.num_onehot) * model.embedding_size)
        else:
            model.embedding_dim = (model.num_multihot + model.num_onehot) * model.embedding_size
    else:
        model.embedding_dim = model.fields_num * model.embedding_size


# construct the embedding layer
def construct_embedding(model, wt_hldr, id_hldr, merge_multi_hot=False):
    feat_mask = tf.expand_dims(wt_hldr, 2)
    if merge_multi_hot and model.num_multihot > 0:
        # *_hot_mask is weight(values that follow the ids in the dataset, different from weight of param) that used
        if model.config.DYNAMIC_LENGTH:
            one_hot_mask, multi_hot_mask = split_mask(feat_mask, model.multi_hot_flags, model.multi_hot_variable_len)
        else:
            one_hot_mask, multi_hot_mask = split_mask(feat_mask, model.multi_hot_flags, model.num_multihot)

        one_hot_v, multi_hot_v = split_param(model.embed_v, id_hldr, model.multi_hot_flags)

        # fm part (reduce multi-hot vector's length to k*1)
        if model.config.DYNAMIC_LENGTH:
            multi_hot_vx = sum_multi_hot(multi_hot_v, multi_hot_mask, model.multi_hot_variable_len)
        else:
            multi_hot_vx = sum_multi_hot(multi_hot_v, multi_hot_mask, model.num_multihot)

        one_hot_vx = tf.multiply(one_hot_v, one_hot_mask)
        vx_embed = tf.concat([one_hot_vx, multi_hot_vx], axis=1)
    else:
        vx_embed = tf.multiply(tf.gather(model.embed_v, id_hldr), feat_mask)
    return vx_embed