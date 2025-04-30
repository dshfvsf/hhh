"""Loss functions for learning global objectives.

These functions have two return values: a Tensor with the value of
the loss, and a dictionary of internal quantities for customizability.
"""

# Dependency imports
import numpy
import tensorflow as tf

from common_util.util import get_specific_filed_feat_val2id_id2val


def loss_choose(config, logits, labels, special_field_weight_hldr=None, label_weight_hldr=None):
    if config.LOSS == 'CROSS':
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    elif config.LOSS == 'WEIGHTED_CROSS':
        loss_array = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        if config.positive_weight:
            loss_array = tf.multiply(label_weight_hldr, loss_array)  # 将正负样本权重的变量与每个样本的Loss相乘
        return tf.multiply(special_field_weight_hldr, loss_array)  # 将样本权重的变量与每个样本的Loss相乘
    elif config.LOSS == 'FOCAL':
        new_logits = tf.nn.sigmoid(logits)
        new_logits = tf.where(tf.equal(labels, 0), 1 - new_logits, new_logits)  # P_t
        base_new_logits = 1 - new_logits
        coefficient = tf.pow(base_new_logits, special_field_weight_hldr)  # 得到focal loss的系数项
        if config.positive_weight:
            coefficient = tf.multiply(label_weight_hldr, coefficient)  # 将正负样本权重的变量与每个样本的权重相乘
        loss_array = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        return tf.multiply(coefficient, loss_array)  # 将系数项与每个样本的Loss相乘
    elif config.LOSS == "PRAUC":
        return precision_recall_auc_loss(labels=labels, logits=logits, dual_rate_factor=config.DUAL_RATE_FACTOR)[0]
    elif config.LOSS == "ROCAUC":
        return roc_auc_loss(labels=labels, logits=logits)[0]
    else:
        return roc_auc_loss(labels=labels, logits=logits)[0]


def precision_recall_auc_loss(
        labels,
        logits,
        precision_range=(0.0, 1.0),
        num_anchors=20,
        weights=1.0,
        dual_rate_factor=0.1,
        label_priors=None,
        surrogate_type='xent',
        lambdas_initializer=tf.constant_initializer(1.0),
        reuse=None,
        variables_collections=None,
        trainable=True,
        scope=None):
    """the Computes precision-recall AUC loss.
    desc->
    The loss is based on a sum of losses for recall at a range of
    precision values (anchor points). This sum is a Riemann sum that
    approximates the area under the precision-recall curve.

    The per-example `weights` argument changes not only the coefficients of
    individual training examples, but how the examples are counted toward the
    constraint. If `label_priors` is given, it MUST take `weights` into account.
    That is,
        the -> label_priors = P / (P + N)
    where
        the -> P = sum_i (wt_i on positives)
        the -> N = sum_i (wt_i on negatives).

    Args desc:
      labels: the  A `Tensor` of shape [batch_size] or [batch_size, num_labels].
      logits: the A `Tensor` with the same shape as `labels`.
      precision_range: A length-two tuple, the range of precision values over
        which to compute AUC. The entries must be nonnegative, increasing, and
        less than or equal to 1.0.
      the num_anchors: The number of grid points used to approximate the Riemann sum.
      the weights: Coefficients for the loss. Must be a scalar or `Tensor` of shape
        [batch_size] or [batch_size, num_labels].
      the dual_rate_factor: A floating point value which controls the step size for
        the Lagrange multipliers.
      the label_priors: None, or a floating point `Tensor` of shape [num_labels]
        containing the prior probability of each label (i.e. the fraction of the
        training data consisting of positive examples). If None, the label
        priors are computed from `labels` with a moving average. See the notes
        above regarding the interaction with `weights` and do not set this unless
        you have a good reason to do so.
      the surrogate_type: Either 'xent' or 'hinge', specifying which upper bound
        should be used for indicator functions.
      the lambdas_initializer: An initializer for the Lagrange multipliers.
      reuse: Whether or not the layer and its variables should be reused. To be
        able to reuse the layer scope must be given.
      variables_collections: Optional list of collections for the variables.
      trainable: If `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`)..
      scope: Optional scope for `variable_scope`.

    Returns:
      loss:The A `Tensor` of the same shape as `logits` with the component-wise
        loss.
      other_outputs: A dictionary of useful internal quantities for debugging. For
        more details, see
        lambdas: desc -> A Tensor of shape [1, num_labels, num_anchors] consisting of the
          Lagrange multipliers.
        biases: A Tensor of shape [1, num_labels, num_anchors] consisting of the
          learned bias term for each.
        label_priors: A Tensor of shape [1, num_labels, 1] consisting of the prior
          probability of each label learned by the loss, if not provided.
        true_positives_lower_bound: Lower bound on the number of true positives
          given `labels` and `logits`. This is the same lower bound which is used
          in the loss expression to be optimized.
        false_positives_upper_bound: Upper bound on the number of false positives
          given `labels` and `logits`. This is the same upper bound which is used
          in the loss expression to be optimized.

    Raises desc:
      ValueError: If `surrogate_type` is not `xent` or `hinge`.
    """
    with tf.variable_scope(scope,
                           'precision_recall_auc',
                           [labels, logits, label_priors],
                           reuse=reuse):
        labels, logits, weights, original_shape = _prepare_labels_logits_weights(
            labels, logits, weights)
        num_labels = get_num_labels(logits)

        # Convert other inputs to tensors and standardize dtypes.
        dual_rate_factor = convert_and_cast(
            dual_rate_factor, 'dual_rate_factor', logits.dtype)

        # Create Tensor of anchor points and distance between anchors.
        precision_values, delta = _range_to_anchors_and_delta(
            precision_range, num_anchors, logits.dtype)
        # Create lambdas with shape [1, num_labels, num_anchors].
        lambdas, lambdas_variable = _create_dual_variable(
            'lambdas',
            shape=[1, num_labels, num_anchors],
            dtype=logits.dtype,
            initializer=lambdas_initializer,
            collections=variables_collections,
            trainable=trainable,
            dual_rate_factor=dual_rate_factor)
        # Create biases with shape [1, num_labels, num_anchors].
        biases = tf.contrib.framework.model_variable(
            name='biases',
            shape=[1, num_labels, num_anchors],
            dtype=logits.dtype,
            initializer=tf.zeros_initializer(),
            collections=variables_collections,
            trainable=trainable)
        # Maybe create label_priors.
        label_priors = maybe_create_label_priors(
            label_priors, labels, weights, variables_collections)
        label_priors = tf.reshape(label_priors, [1, num_labels, 1])

        # Expand logits, labels, and weights to shape [batch_size, num_labels, 1].
        logits = tf.expand_dims(logits, 2)
        labels = tf.expand_dims(labels, 2)
        weights = tf.expand_dims(weights, 2)

        # Calculate weighted loss and other outputs. The log(2.0) term corrects for
        # logloss not being an upper bound on the indicator function.
        loss = weights * weighted_surrogate_loss(
            labels,
            logits + biases,
            surrogate_type=surrogate_type,
            positive_weights=1.0 + lambdas * (1.0 - precision_values),
            negative_weights=lambdas * precision_values)
        maybe_log2 = tf.log(2.0) if surrogate_type == 'xent' else 1.0
        maybe_log2 = tf.cast(maybe_log2, logits.dtype.base_dtype)
        lambda_term = lambdas * (1.0 - precision_values) * label_priors * maybe_log2
        per_anchor_loss = loss - lambda_term
        per_label_loss = delta * tf.reduce_sum(per_anchor_loss, 2)
        # Normalize the AUC such that a perfect score function will have AUC 1.0.
        # Because precision_range is discretized into num_anchors + 1 intervals
        # but only num_anchors terms are included in the Riemann sum, the
        # effective length of the integration interval is `delta` less than the
        # length of precision_range.
        scaled_loss = tf.div(per_label_loss,
                             precision_range[1] - precision_range[0] - delta,
                             name='AUC_Normalize')
        scaled_loss = tf.reshape(scaled_loss, original_shape)

        other_outputs = {
            'lambdas': lambdas_variable,
            'biases': biases,
            'label_priors': label_priors,
            'true_positives_lower_bound': true_positives_lower_bound(
                labels, logits, weights, surrogate_type),
            'false_positives_upper_bound': false_positives_upper_bound(
                labels, logits, weights, surrogate_type)}

        return scaled_loss, other_outputs


def roc_auc_loss(
        labels,
        logits,
        weights=1.0,
        surrogate_type='xent',
        scope=None):
    """Computes ROC AUC loss.

    The area under the ROC curve is the probability p that a randomly chosen
    positive example will be scored higher than a randomly chosen negative
    example. This loss approximates 1-p by using a surrogate (either hinge loss or
    cross entropy) for the indicator function. Specifically, the loss is:

      sum_i sum_j w_i*w_j*loss(logit_i - logit_j)

    where i ranges over the positive datapoints, j ranges over the negative
    datapoints, logit_k denotes the logit (or score) of the k-th datapoint, and
    loss is either the hinge or log loss given a positive label.

    Args:
      labels: A `Tensor` of shape [batch_size] or [batch_size, num_labels].
      logits: A `Tensor` with the same shape and dtype as `labels`.
      weights: Coefficients for the loss. Must be a scalar or `Tensor` of shape
        [batch_size] or [batch_size, num_labels].
      surrogate_type: Either 'xent' or 'hinge', specifying which upper bound
        should be used for the indicator function.
      scope: Optional scope for `name_scope`.

    Returns:
      loss: A `Tensor` of the same shape as `logits` with the component-wise loss.
      other_outputs: An empty dictionary, for consistency.

    Raises:
      ValueError: If `surrogate_type` is not `xent` or `hinge`.
    """
    with tf.name_scope(scope, 'roc_auc', [labels, logits, weights]):
        # Convert inputs to tensors and standardize dtypes.
        labels, logits, weights, original_shape = _prepare_labels_logits_weights(
            labels, logits, weights)

        # Create tensors of pairwise differences for logits and labels, and
        # pairwise products of weights. These have shape
        # [batch_size, batch_size, num_labels].
        logits_difference = tf.expand_dims(logits, 0) - tf.expand_dims(logits, 1)
        labels_difference = tf.expand_dims(labels, 0) - tf.expand_dims(labels, 1)
        weights_product = tf.expand_dims(weights, 0) * tf.expand_dims(weights, 1)

        signed_logits_difference = labels_difference * logits_difference
        raw_loss = weighted_surrogate_loss(
            labels=tf.ones_like(signed_logits_difference),
            logits=signed_logits_difference,
            surrogate_type=surrogate_type)
        weighted_loss = weights_product * raw_loss

        # Zero out entries of the loss where labels_difference zero (so loss is only
        # computed on pairs with different labels).
        loss = tf.reduce_mean(tf.abs(labels_difference) * weighted_loss, 0) * 0.5
        loss = tf.reshape(loss, original_shape)
        return loss, {}


def _prepare_labels_logits_weights(labels, logits, weights):
    """Validates labels, logits, and weights.

    Converts inputs to tensors, checks shape compatibility, and casts dtype if
    necessary.

    Args:
      labels: A `Tensor` of shape [batch_size] or [batch_size, num_labels].
      logits: A `Tensor` with the same shape as `labels`.
      weights: Either `None` or a `Tensor` with shape broadcastable to `logits`.

    Returns:
      labels: Same as `labels` arg after possible conversion to tensor, cast, and
        reshape.
      logits: Same as `logits` arg after possible conversion to tensor and
        reshape.
      weights: Same as `weights` arg after possible conversion, cast, and reshape.
      original_shape: Shape of `labels` and `logits` before reshape.

    Raises:
      ValueError: If `labels` and `logits` do not have the same shape.
    """
    # Convert `labels` and `logits` to Tensors and standardize dtypes.
    logits = tf.convert_to_tensor(logits, name='logits')
    labels = convert_and_cast(labels, 'labels', logits.dtype.base_dtype)
    weights = convert_and_cast(weights, 'weights', logits.dtype.base_dtype)

    try:
        labels.get_shape().merge_with(logits.get_shape())
    except Exception:
        print('logits and labels must have the same shape (%s vs %s)' %
              (logits.get_shape(), labels.get_shape()))

    original_shape = logits.get_shape().as_list()
    if logits.get_shape().ndims > 0:
        original_shape[0] = -1
    if logits.get_shape().ndims <= 1:
        labels = tf.reshape(labels, [-1, 1])
        logits = tf.reshape(logits, [-1, 1])

    if weights.get_shape().ndims == 1:
        # Weights has shape [batch_size]. Reshape to [batch_size, 1].
        weights = tf.reshape(weights, [-1, 1])
    if weights.get_shape().ndims == 0:
        # Weights is a scalar. Change shape of weights to match logits.
        weights *= tf.ones_like(logits)

    return [labels, logits, weights, original_shape]


def _range_to_anchors_and_delta(precision_range, num_anchors, dtype):
    """Calculates anchor points from precision range.

    Args:
      precision_range: As required in precision_recall_auc_loss.
      num_anchors: int, number of equally spaced anchor points.
      dtype: Data type of returned tensors.

    Returns:
      precision_values: A `Tensor` of data type dtype with equally spaced values
        in the interval precision_range.
      delta: The spacing between the values in precision_values.

    Raises:
      ValueError: If precision_range is invalid.
    """
    # Validate precision_range.
    if not 0 <= precision_range[0] <= precision_range[-1] <= 1:
        raise ValueError('precision values must obey 0 <= %f <= %f <= 1' %
                         (precision_range[0], precision_range[-1]))
    if not 0 < len(precision_range) < 3:
        raise ValueError('length of precision_range (%d) must be 1 or 2' %
                         len(precision_range))

    # Sets precision_values uniformly between min_precision and max_precision.
    values = numpy.linspace(start=precision_range[0],
                            stop=precision_range[1],
                            num=num_anchors + 2)[1:-1]
    precision_values = convert_and_cast(
        values, 'precision_values', dtype)
    delta = convert_and_cast(
        values[0] - precision_range[0], 'delta', dtype)
    # Makes precision_values [1, 1, num_anchors].
    precision_values = expand_outer(precision_values, 3)
    return precision_values, delta


def _create_dual_variable(name, shape, dtype, initializer, collections,
                          trainable, dual_rate_factor):
    """Creates a new dual variable.

    Dual variables are required to be nonnegative. If trainable, their gradient
    is reversed so that they are maximized (rather than minimized) by the
    optimizer.

    Args:
      name: A string, the name for the new variable.
      shape: Shape of the new variable.
      dtype: Data type for the new variable.
      initializer: Initializer for the new variable.
      collections: List of graph collections keys. The new variable is added to
        these collections. Defaults to `[GraphKeys.GLOBAL_VARIABLES]`.
      trainable: If `True`, the default, also adds the variable to the graph
        collection `GraphKeys.TRAINABLE_VARIABLES`. This collection is used as
        the default list of variables to use by the `Optimizer` classes.
      dual_rate_factor: A floating point value or `Tensor`. The learning rate for
        the dual variable is scaled by this factor.

    Returns:
      dual_value: An op that computes the absolute value of the dual variable
        and reverses its gradient.
      dual_variable: The underlying variable itself.
    """
    # We disable partitioning while constructing dual variables because they will
    # be updated with assign, which is not available for partitioned variables.
    partitioner = tf.get_variable_scope().partitioner
    try:
        tf.get_variable_scope().set_partitioner(None)
        dual_variable = tf.contrib.framework.model_variable(
            name=name,
            shape=shape,
            dtype=dtype,
            initializer=initializer,
            collections=collections,
            trainable=trainable)
    finally:
        tf.get_variable_scope().set_partitioner(partitioner)
    # Using the absolute value enforces nonnegativity.
    dual_value = tf.abs(dual_variable)

    if trainable:
        # To reverse the gradient on the dual variable, multiply the gradient by
        # -dual_rate_factor
        dual_value = (tf.stop_gradient((1.0 + dual_rate_factor) * dual_value)
                      - dual_rate_factor * dual_value)
    return dual_value, dual_variable


def maybe_create_label_priors(label_priors,
                              labels,
                              weights,
                              variables_collections):
    """Creates moving average ops to track label priors, if necessary.

    Args:
      label_priors: As required in e.g. precision_recall_auc_loss.
      labels: A `Tensor` of shape [batch_size] or [batch_size, num_labels].
      weights: As required in e.g. precision_recall_auc_loss.
      variables_collections: Optional list of collections for the variables, if
        any must be created.

    Returns:
      label_priors: A Tensor of shape [num_labels] consisting of the
        weighted label priors, after updating with moving average ops if created.
    """
    if label_priors is not None:
        label_priors = convert_and_cast(
            label_priors, name='label_priors', dtype=labels.dtype.base_dtype)
        return tf.squeeze(label_priors)

    label_priors = build_label_priors(
        labels,
        weights,
        variables_collections=variables_collections)
    return label_priors


def true_positives_lower_bound(labels, logits, weights, surrogate_type):
    """Calculate a lower bound on the number of true positives.

    This lower bound on the number of true positives given `logits` and `labels`
    is the same one used in the global objectives loss functions.

    Args:
      labels: A `Tensor` of shape [batch_size] or [batch_size, num_labels].
      logits: A `Tensor` of shape [batch_size, num_labels] or
        [batch_size, num_labels, num_anchors]. If the third dimension is present,
        the lower bound is computed on each slice [:, :, k] independently.
      weights: Per-example loss coefficients, with shape broadcast-compatible with
          that of `labels`.
      surrogate_type: Either 'xent' or 'hinge', specifying which upper bound
        should be used for indicator functions.

    Returns:
      A `Tensor` of shape [num_labels] or [num_labels, num_anchors].
    """
    maybe_log2 = tf.log(2.0) if surrogate_type == 'xent' else 1.0
    maybe_log2 = tf.cast(maybe_log2, logits.dtype.base_dtype)
    if logits.get_shape().ndims == 3 and labels.get_shape().ndims < 3:
        labels = tf.expand_dims(labels, 2)
    if maybe_log2 != 0:
        loss_on_positives = weighted_surrogate_loss(
            labels, logits, surrogate_type, negative_weights=0.0) / maybe_log2
    else:
        print('zero')
    return tf.reduce_sum(weights * (labels - loss_on_positives), 0)


def false_positives_upper_bound(labels, logits, weights, surrogate_type):
    """Calculate an upper bound on the number of false positives.

    This upper bound on the number of false positives given `logits` and `labels`
    is the same one used in the global objectives loss functions.

    Args:
      labels: A `Tensor` of shape [batch_size, num_labels]
      logits: A `Tensor` of shape [batch_size, num_labels]  or
        [batch_size, num_labels, num_anchors]. If the third dimension is present,
        the lower bound is computed on each slice [:, :, k] independently.
      weights: Per-example loss coefficients, with shape broadcast-compatible with
          that of `labels`.
      surrogate_type: Either 'xent' or 'hinge', specifying which upper bound
        should be used for indicator functions.

    Returns:
      A `Tensor` of shape [num_labels] or [num_labels, num_anchors].
    """
    maybe_log2 = tf.log(2.0) if surrogate_type == 'xent' else 1.0
    maybe_log2 = tf.cast(maybe_log2, logits.dtype.base_dtype)
    if maybe_log2 != 0:
        loss_on_negatives = weighted_surrogate_loss(
            labels, logits, surrogate_type, positive_weights=0.0) / maybe_log2
    else:
        print('zero')

    return tf.reduce_sum(weights * loss_on_negatives, 0)


def weighted_sigmoid_cross_entropy_with_logits(labels,
                                               logits,
                                               positive_weights=1.0,
                                               negative_weights=1.0,
                                               name=None):
    logits_tmp = logits
    with tf.name_scope(
            name,
            'weighted_logistic_loss',
            [logits_tmp, labels, positive_weights, negative_weights]) as name:
        labels, logits_tmp, positive_weights, negative_weights = prepare_loss_args(
            labels, logits_tmp, positive_weights, negative_weights)
        # done label
        softplus_term = tf.add(tf.maximum(-logits_tmp, 0.0),
                               tf.log(1.0 + tf.exp(-tf.abs(logits_tmp))))
        weight_dependent_factor = (
                negative_weights + (positive_weights - negative_weights) * labels)
        return (negative_weights * (logits_tmp - labels * logits_tmp) +
                weight_dependent_factor * softplus_term)


def weighted_hinge_loss(labels,
                        logits,
                        positive_weights=1.00,
                        negative_weights=1.00,
                        name=None):
    logits_tmp = logits
    with tf.name_scope(
            name, 'weighted_hinge_loss',
            [logits_tmp, labels, positive_weights, negative_weights]) as name:
        labels, logits_tmp, positive_weights, negative_weights = prepare_loss_args(
            labels, logits_tmp, positive_weights, negative_weights)
        # done hinge loss
        positives_term = positive_weights * labels * tf.maximum(1.0 - logits_tmp, 0)
        negatives_term = (negative_weights * (1.0 - labels)
                          * tf.maximum(1.0 + logits_tmp, 0))
        return positives_term + negatives_term


def weighted_surrogate_loss(labels,
                            logits,
                            surrogate_type='xent',
                            positive_weights=1.00,
                            negative_weights=1.00,
                            name=None):
    logits_tmp = logits
    with tf.name_scope(
            name, 'weighted_loss',
            [logits_tmp, labels, surrogate_type, positive_weights,
             negative_weights]) as name:
        if surrogate_type == 'xent':
            # done xent
            return weighted_sigmoid_cross_entropy_with_logits(
                logits=logits_tmp,
                labels=labels,
                positive_weights=positive_weights,
                negative_weights=negative_weights,
                name=name)
        elif surrogate_type == 'hinge':
            # ret hinge
            return weighted_hinge_loss(
                logits=logits_tmp,
                labels=labels,
                positive_weights=positive_weights,
                negative_weights=negative_weights,
                name=name)
        # raise
        raise ValueError('surrogate_type %s not supported.' % surrogate_type)


def expand_outer(tensor, rank):
    if tensor.get_shape().ndims is None:
        raise ValueError('tensor dimension must be known.')
    rank_tmp = rank
    if len(tensor.get_shape()) > rank_tmp:
        # rais some
        raise ValueError(
            '`rank` must be at current tensor dimension: (%s vs %s).' %
            (rank_tmp, len(tensor.get_shape())))
    while len(tensor.get_shape()) < rank_tmp:
        tensor = tf.expand_dims(tensor, 0)
    return tensor


def build_label_priors(labels,
                       weights=None,
                       positive_pseudocount=1.00,
                       negative_pseudocount=1.00,
                       variables_collections=None):
    dtype = labels.dtype.base_dtype
    num_labels = get_num_labels(labels)
    positive_pseudocount_tmp = positive_pseudocount

    if weights is None:
        weights = tf.ones_like(labels)

    partitioner = tf.get_variable_scope().partitioner
    try:
        tf.get_variable_scope().set_partitioner(None)
        # get done
        weighted_label_counts = tf.contrib.framework.model_variable(
            name='weighted_label_counts_tmp',
            shape=[num_labels],
            dtype=dtype,
            initializer=tf.constant_initializer(
                [positive_pseudocount_tmp] * num_labels, dtype=dtype),
            collections=variables_collections,
            trainable=False)
        weighted_label_counts_update = weighted_label_counts.assign_add(
            tf.reduce_sum(weights * labels, 0))
        # define weight
        weight_sum = tf.contrib.framework.model_variable(
            name='weight_sum_tmp',
            shape=[num_labels],
            dtype=dtype,
            initializer=tf.constant_initializer(
                [positive_pseudocount_tmp + negative_pseudocount] * num_labels,
                dtype=dtype),
            collections=variables_collections,
            trainable=False)
        weight_sum_update = weight_sum.assign_add(tf.reduce_sum(weights, 0))
    #tmp_finnal
    finally:
        tf.get_variable_scope().set_partitioner(partitioner)
    # label beg
    label_priors = tf.div(
        weighted_label_counts_update,
        weight_sum_update)
    return label_priors


def convert_and_cast(value, name, dtype):
    return tf.cast(tf.convert_to_tensor(value, name=name), dtype=dtype)


def prepare_loss_args(labels, logits, positive_weights, negative_weights):
    positive_weights_tm = positive_weights
    logits = tf.convert_to_tensor(logits, name='logits')
    labels = convert_and_cast(labels, 'labels', logits.dtype)
    # labels convert done
    if len(labels.get_shape()) == 2 and len(logits.get_shape()) == 3:
        labels = tf.expand_dims(labels, [2])

    positive_weights_tm = convert_and_cast(positive_weights_tm, 'positive_weights_tm',
                                        logits.dtype)
    positive_weights_tm = expand_outer(positive_weights_tm, logits.get_shape().ndims)
    negative_weights = convert_and_cast(negative_weights, 'negative_weights',
                                        logits.dtype)
    negative_weights = expand_outer(negative_weights, logits.get_shape().ndims)
    return [labels, logits, positive_weights_tm, negative_weights]


def get_num_labels(labels_or_logits):
    """Returns the number of labels inferred from labels_or_logits."""
    if labels_or_logits.get_shape().ndims <= 1:
        return 1
    return labels_or_logits.get_shape()[1].value


def get_bid_weighted_loss(model, loss):
    final_bid_weight = tf.ones_like(model.lbl_hldr)
    bid_weight_plans = getattr(model.config, 'bid_weight_plans', [])

    for weight_plan in bid_weight_plans:
        # 缩放参数
        scale_min = weight_plan.get('scale_min', 1)
        scale_max = weight_plan.get('scale_max', 2)
        overall_min = weight_plan.get('overall_min', 0)
        overall_max = weight_plan.get('overall_max', 100000)

        # 使用何种缩放的相关配置
        use_log_scaling = weight_plan.get('use_log_scaling', False)
        use_overall_scaling = weight_plan.get('use_overall_scaling', False)
        use_batch_scaling = weight_plan.get('use_batch_scaling', False)
        use_weight_mask = weight_plan.get('use_weight_mask', False)
        use_positive_reweighting_only = weight_plan.get('use_positive_reweighting_only', False)

        bid_weight = model.bid_hldr
        bid_weight = tf.clip_by_value(bid_weight, overall_min, overall_max)

        if use_log_scaling:
            bid_weight = tf.math.log(bid_weight + 1)

        if use_overall_scaling:
            k = (scale_max - scale_min) / (overall_max - overall_min)
            bid_weight = tf.where(tf.equal(bid_weight, 1 if use_log_scaling else -1),
                                  tf.ones_like(bid_weight),
                                  scale_min + k * (bid_weight - overall_min))
        elif use_batch_scaling:
            k = (scale_max - scale_min) / (tf.reduce_max(bid_weight) - tf.reduce_min(bid_weight))
            bid_weight = tf.where(tf.equal(bid_weight, 1 if use_log_scaling else -1),
                                  tf.ones_like(bid_weight),
                                  scale_min + k * (bid_weight - tf.reduce_min(bid_weight)))

        if use_weight_mask:
            mask_idx = weight_plan.get('mask_idx')
            mask_tag = weight_plan.get('mask_tag')
            mask_content = weight_plan.get('mask_content')
            if mask_idx is None or mask_tag is None or mask_content is None:
                raise ValueError("When use_weight_mask is true, please give the feature index to mask_idx "
                                 "the feature name to mask_tag and give the feature value to mask_content")
            mask_val2id, _ = get_specific_filed_feat_val2id_id2val(model.config, mask_tag)
            mask_id = mask_val2id.get(mask_content, None)
            print('reweighting for %s-%s' % (mask_tag, mask_content))
            bid_weight = tf.where(tf.equal(model.id_hldr[:, mask_idx], mask_id), bid_weight,
                                    tf.ones_like(bid_weight))

        if use_positive_reweighting_only:
            print("reweighting positive samples only...")
            bid_weight = tf.where(tf.equal(model.lbl_hldr, 0.0), tf.ones_like(bid_weight), bid_weight)

        final_bid_weight = tf.multiply(final_bid_weight, bid_weight)

    loss_array = tf.multiply(final_bid_weight, loss)
    return loss_array
