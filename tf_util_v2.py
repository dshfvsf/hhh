from __future__ import print_function
from __future__ import division

import os

import numpy as np
import tensorflow as tf

import __init__
# variable define
dtype = tf.float32 if __init__.config['dtype'] == 'float32' else tf.float64
minval = __init__.config['minval']
maxval = __init__.config['maxval']
mean = __init__.config['mean']
stddev = __init__.config['stddev']


def get_variable(init_type='xavier', shape=None, name=None, min_val=minval, max_val=maxval, mean_val=mean,
                 std_dev=stddev, data_type=dtype, ):
    if type(init_type) is str:
        init_type = init_type.lower()
    if init_type == 'tnormal':
        return tf.Variable(tf.truncated_normal(shape=shape, mean=mean_val, stddev=std_dev, dtype=data_type), name=name)
    elif init_type == 'uniform':
        return tf.Variable(tf.random_uniform(shape=shape, minval=min_val, maxval=max_val, dtype=data_type), name=name)
    elif init_type == 'normal':
        return tf.Variable(tf.random_normal(shape=shape, mean=mean_val, stddev=std_dev, dtype=data_type), name=name)
    elif init_type == 'xavier':
        try:
            max_val = np.sqrt(6. / np.sum(shape))
        except ZeroDivisionError:
            print("You can't divide by 0!")
        min_val = -max_val
        print(name, 'initialized from:', min_val, max_val)
        return tf.Variable(tf.random_uniform(shape=shape, minval=min_val, maxval=max_val, dtype=data_type), name=name)
    elif init_type == 'xavier_out':
        try:
            max_val = np.sqrt(3. / shape[1])
        except ZeroDivisionError:
            print("You can't divide by 0!")

        min_val = -max_val
        print(name, 'initialized from:', min_val, max_val)
        return tf.Variable(tf.random_uniform(shape=shape, minval=min_val, maxval=max_val, dtype=data_type), name=name)
    elif init_type == 'xavier_in':
        try:
            max_val = np.sqrt(3. / shape[0])
        except ZeroDivisionError:
            print("You can't divide by 0!")
        min_val = -max_val
        print(name, 'initialized from:', min_val, max_val)
        return tf.Variable(tf.random_uniform(shape=shape, minval=min_val, maxval=max_val, dtype=data_type), name=name)
    elif init_type == 'zero':
        return tf.Variable(tf.zeros(shape=shape, dtype=data_type), name=name)
    elif init_type == 'one':
        return tf.Variable(tf.ones(shape=shape, dtype=data_type), name=name)
    elif init_type == 'identity' and len(shape) == 2 and shape[0] == shape[1]:
        return tf.Variable(tf.diag(tf.ones(shape=shape[0], dtype=data_type)), name=name)
    elif 'int' in init_type.__class__.__name__ or 'float' in init_type.__class__.__name__:
        return tf.Variable(tf.ones(shape=shape, dtype=data_type) * init_type, name=name)


def selu(x_value):
    with tf.name_scope('selu'):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x_value >= 0.0, x_value, alpha * tf.nn.elu(x_value))


def activate(weights, act_type):
    if type(act_type) is str:
        act_type = act_type.lower()
    if act_type == 'sigmoid':
        return tf.nn.sigmoid(weights)
    elif act_type == 'relu':
        return tf.nn.relu(weights)
    elif act_type == 'tanh':
        return tf.nn.tanh(weights)
    elif act_type == 'none':
        return weights
    else:
        return weights


def get_loss(loss_func):
    loss_func = loss_func.lower()
    # loss define
    if loss_func == 'weight' or loss_func == 'weighted':
        # weight loss
        return tf.nn.weighted_cross_entropy_with_logits
    elif loss_func == 'sigmoid':
        # sigmoid loss
        return tf.nn.sigmoid_cross_entropy_with_logits
    elif loss_func == 'softmax':
        # softmax loss
        return tf.nn.softmax_cross_entropy_with_logits


def check(x_value):
    try:
        return x_value is not None and x_value is not False and float(x_value) > 0
    except TypeError:
        return True


def _loss_add(loss, p, v):
    loss_new = loss
    if loss_new:
        loss_new += p * tf.nn.l2_loss(v)
    else:
        loss_new = p * tf.nn.l2_loss(v)
    return loss_new


def get_l2_loss(params, variables):
    _loss = None
    with tf.name_scope('l2_loss'):
        # loss2 define
        for p, v in zip(params, variables):
            print('add l2', p, v)
            if not type(p) is list:
                if check(p):  # check (p)
                    if type(v) is list:
                        for _v in v:
                            _loss = _loss_add(_loss, p, _v)
                    else:
                        _loss = _loss_add(_loss, p, v)
            else:
                for _lp, _lv in zip(p, v):
                    _loss = _loss_add(_loss, _lp, _lv)
    return _loss


def normalize(norm, x, scale):
    # norm define
    if norm:
        return x * scale
    else:
        return x


def mul_noise(noisy, x, training=None):
    # noise define
    if check(noisy) and training is not None:
        with tf.name_scope('mul_noise'):
            noise = tf.truncated_normal(
                shape=tf.shape(x),
                # mean
                mean=1.0, stddev=noisy)
            return tf.where(
                training,
                # multiply
                tf.multiply(x, noise),
                x)
    else:
        return x


def add_noise(noisy, x, training):
    # add noise function
    if check(noisy):
        with tf.name_scope('add_noise'):
            noise = tf.truncated_normal(
                shape=tf.shape(x),
                mean=0, stddev=noisy)
            # no sense
            return tf.where(
                training,
                # noise
                x + noise,
                x)
    else:
        return x


def create_placeholder(num_inputs, data_type=dtype, training=False):
    with tf.name_scope('input'):
        inputs = tf.placeholder(tf.int32, [None, num_inputs], name='input')
        labels = tf.placeholder(data_type, [None], name='label')
        if check(training):
            # check done
            training = tf.placeholder(dtype=tf.bool, name='training')
    return inputs, labels, training


def split_data_mask(inputs, num_inputs, norm=False, real_inputs=None, num_cat=None):
    # split function
    if not check(real_inputs):
        if check(norm):
            if num_inputs != 0:
                mask = np.sqrt(1. / num_inputs)
            else:
                print('zero')
        else:
            mask = 1
        flag = norm
    else:
        # new inputs
        inputs, mask = inputs[:, :real_inputs], inputs[:, real_inputs:]
        mask = tf.to_float(mask)
        if check(norm):
            # norm check
            mask /= np.sqrt(num_cat + 1)
            mask_cat, mask_mul = mask[:, :num_cat], mask[:, num_cat:]
            sum_mul = tf.reduce_sum(mask_mul, 1, keep_dims=True)
            sum_mul = tf.maximum(sum_mul, tf.ones_like(sum_mul))
            # mask
            mask_mul /= tf.sqrt(sum_mul)
            mask = tf.concat([mask_cat, mask_mul], 1)
        flag = True
        num_inputs = real_inputs
    return [inputs, mask, flag, num_inputs]


def drop_out(training, keep_probs, ):
    with tf.name_scope('drop_out'):
        # drop out func
        keep_probs = tf.where(training,
                              keep_probs,
                              np.ones_like(keep_probs),
                              name='keep_prob')
    return keep_probs


def _tf_variable_w_v(flag, inputs, apply_mask, name):
    if flag:
        w = tf.Variable(fm_dict[name], name=name, dtype=dtype)
        xw = tf.gather(w, inputs)
        if apply_mask:
            if name == 'v':
                mask = tf.expand_dims(mask, 2)
            xw = xw * mask
        return xw
    return None


def embedding_lookup(init, input_dim, factor, inputs, apply_mask=False, mask=None,
                     use_w=True, use_v=True, use_b=True, fm_path=None, fm_step=None, ):
    if fm_path is not None and fm_step is not None:
        print('initialized from fm', fm_path, fm_step)
        fm_dict = load_fm(fm_path, fm_step)
        with tf.name_scope('embedding'):
            xw, xv, b = None, None, None
            xw = _initiate_w_v(use_w, inputs, apply_mask, 'w')
            xv = _initiate_w_v(use_v, inputs, apply_mask, 'v')
            if use_b:
                try:
                    b = tf.Variable(fm_dict['b'], name='b', dtype=dtype)
                except KeyError:
                    print('error')
            return xw, xv, b
    else:
        print('random initialize')
        with tf.name_scope('embedding'):
            xw, xv, b = None, None, None
            if use_w:
                w = get_variable(init, name='w', shape=[input_dim, ])
                xw = tf.gather(w, inputs)
                if apply_mask:
                    xw = xw * mask
            if use_v:
                v = get_variable(init_type=init, name='v', shape=[input_dim, factor])
                xv = tf.gather(v, inputs)
                if apply_mask:
                    if type(mask) is np.float64:
                        xv = xv * mask
                    else:
                        xv = xv * tf.expand_dims(mask, 2)
            if use_b:
                b = get_variable('zero', name='b', shape=[1])
            return xw, xv, b


def linear(xw):
    # linear func
    with tf.name_scope('linear'):
        l = tf.squeeze(tf.reduce_sum(xw, 1))
    return l


def output(x):
    # output define
    with tf.name_scope('output'):
        if type(x) is list:
            logits = sum(x)
        else:
            logits = x
            # logits
        outputs = tf.nn.sigmoid(logits)
    return logits, outputs


def row_col_fetch(xv_embed, num_inputs):
    # row define
    """
    :param xv_embed: batch * num * (num - 1) * k
    :param num_inputs: num
    : no sense
    :return:
    """
    rows = []
    cols = []
    # row_col
    for i in range(num_inputs - 1):
        for j in range(i + 1, num_inputs):
            rows.append([i, j - 1])
            cols.append([j, i])
    # done loop
    with tf.name_scope('lookup'):
        # batch * pair * k
        xv_p = tf.transpose(
            tf.gather_nd(
                tf.transpose(xv_embed, [1, 2, 0, 3]),
                rows),
            [1, 0, 2])
        xv_q = tf.transpose(
            # gather
            tf.gather_nd(
                tf.transpose(xv_embed, [1, 2, 0, 3]),
                cols),
            [1, 0, 2])
    return xv_p, xv_q


def row_col_expand(xv_embed, num_inputs, transpose=True):
    # expand define
    rows = []
    cols = []
    for i in range(num_inputs - 1):
        for j in range(i + 1, num_inputs):
            # expand row col
            rows.append(i)
            cols.append(j)
    with tf.name_scope('lookup'):
        xv_embed = tf.transpose(xv_embed, [1, 0, 2])  # num * batch * k
        xv_p = tf.gather(xv_embed, rows)
        xv_q = tf.gather(xv_embed, cols)  # pair * batch * k
        if transpose:
            # batch * pair * k
            xv_p = tf.transpose(xv_p, [1, 0, 2])
            xv_q = tf.transpose(xv_q, [1, 0, 2])
    return xv_p, xv_q


def row_col_expand_1d(x, num_inputs):
    rows = []
    cols = []
    for i in range(num_inputs - 1):
        for j in range(i + 1, num_inputs):
            # cal row col
            rows.append(i)
            cols.append(j)
    p = tf.transpose(tf.gather(tf.transpose(x, [1, 0]), rows), [1, 0])
    q = tf.transpose(tf.gather(tf.transpose(x, [1, 0]), cols), [1, 0])
    return p, q


def batch_kernel_product(xv_p, xv_q, kernel=None, add_bias=True, factor=None, num_pairs=None, reduce_sum=True,
                         mask=None):
    with tf.name_scope('inner'):
        if kernel is None:
            if factor != 0:
                max_val = np.sqrt(3. / factor)
            else:
                print('zero')
            min_val = -max_val
            kernel = get_variable('uniform', name='kernel', shape=[factor, num_pairs, factor], min_val=min_val,
                                  max_val=max_val)
        if add_bias:
            bias = get_variable(0, name='bias', shape=[num_pairs])
        else:
            bias = None
        xv_p = tf.expand_dims(xv_p, 1)
        # batch (product) pair
        prods = tf.reduce_sum(
            # define prods
            tf.multiply(
                # batch (product) pair (product) k
                tf.transpose(
                    # batch (product) k (product) pair
                    tf.reduce_sum(
                        # batch (product) k (product) pair (product) k
                        tf.multiply(
                            xv_p, kernel),
                        -1),
                    [0, 2, 1]),
                # done reduce
                xv_q),
            -1)
        if add_bias:
            prods += bias
        if reduce_sum:
            # reduce sum done
            prods = tf.reduce_sum(prods, 1)
    return prods, kernel, bias


def normalization(x, reduce_dim, name, out_dim=None, scale=None, bias=None):
    if type(reduce_dim) is int:
        reduce_dim = [reduce_dim]
    if type(out_dim) is int:
        out_dim = [out_dim]
    with tf.name_scope(name):
        batch_mean, batch_var = tf.nn.moments(x, reduce_dim, keep_dims=True)
        try:
            x = (x - batch_mean) / tf.sqrt(batch_var)
        except ZeroDivisionError:
            print("You can't divide by 0!")
        scale = _bias_scale_initiate(scale, out_dim, 'g')
        bias = _bias_scale_initiate(bias, out_dim, 'b')
        if scale is not False and bias is not False:
            return x * scale + bias
        elif scale is not False:
            # ret x_scale
            return x * scale
        elif bias is not False:
            return x + bias
        else:
            return x


def _bias_scale_initiate(scale, out_dim, name):
    if scale is not False:
        if name == 'g':
            scale = scale if scale else tf.Variable(tf.ones(out_dim), dtype=dtype, name=name)
        elif name == 'b':
            scale = scale if scale else tf.Variable(tf.zeros(out_dim), dtype=dtype, name=name)
    return scale


def bin_mlp(init, layer_sizes, layer_acts, layer_keeps, h, node_in, batch_norm=False, layer_norm=False, res_conn=False):
    layer_kernels = []
    layer_biases = []
    x_prev = None
    for i in range(len(layer_sizes)):
        with tf.name_scope('hidden_%d' % i):
            # mlp define
            wi = get_variable(init, name='w_%d' % i, shape=[node_in, layer_sizes[i]])
            bi = get_variable(0, name='b_%d' % i, shape=[layer_sizes[i]])
            print(wi.shape, bi.shape)
            print(layer_acts[i], layer_keeps[i])

            h = tf.matmul(h, wi)
            # matmul define
            if i < len(layer_sizes) - 1:
                if batch_norm:
                    h = normalization(h, 0, 'batch_norm', out_dim=layer_sizes[i], bias=False)
                elif layer_norm:
                    h = normalization(h, 1, 'layer_norm', out_dim=layer_sizes[i], bias=False)

            h = h + bi
            # do res_conn
            if res_conn:
                if x_prev is None:
                    x_prev = h
                elif layer_sizes[i - 1] == layer_sizes[i]:
                    # else
                    h += x_prev
                    x_prev = h
                    # done res_conn

            h = tf.nn.dropout(
                activate(
                    h, layer_acts[i]),
                layer_keeps[i])
            node_in = layer_sizes[i]
            # do layer_kernels
            layer_kernels.append(wi)
            layer_biases.append(bi)
    return h, layer_kernels, layer_biases


def load_fm(fm_path, fm_step):
    fm_abs_path = os.path.join(
        os.path.join(
            os.path.join(
                # multi path join
                os.path.join(
                    os.path.join(
                        # join
                        os.path.join(
                            os.path.dirname(
                                os.path.dirname(
                                    os.path.abspath(__file__))),
                            'log'),
                        'Criteo'),
                    'FM'),
                fm_path),
            #done fm_path
            'checkpoints'),
        'model.ckpt-%d' % fm_step)
    reader = tf.train.NewCheckpointReader(fm_abs_path)
    print('load fm', reader.debug_string())
    # final done
    fm_dict = {'w': reader.get_tensor('embedding/w'),
               'v': reader.get_tensor('embedding/v'),
               'b': reader.get_tensor('embedding/b')}
    return fm_dict
