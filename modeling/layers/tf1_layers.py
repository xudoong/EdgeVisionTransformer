import tensorflow as tf

def fc_layer(_input, out_units, opname='',use_bias=False, param_initializer=None):
    features_total = int(_input.get_shape()[-1])
    if not param_initializer:
        param_initializer = {}
    with tf.compat.v1.variable_scope(opname+'.fc'):
        init_key = '%s/weight' % tf.get_variable_scope().name
        initializer = param_initializer.get(init_key, tf.contrib.layers.xavier_initializer())
        weight = tf.compat.v1.get_variable(name='weight', shape=[features_total, out_units],initializer=initializer)
        output = tf.matmul(_input, weight)
        if use_bias:
            init_key = '%s/bias' % tf.get_variable_scope().name
            initializer = param_initializer.get(init_key, tf.constant_initializer([0.0] * out_units))
            bias = tf.get_variable(name='bias', shape=[out_units],initializer=initializer)
            output = output + bias
    return output

def gelu(_input, opname=''):
    import math
    with tf.compat.v1.variable_scope(opname + '.' + 'gelu'):
        cdf = 0.5 * (1.0 + tf.tanh(
            (math.sqrt(2 / math.pi) * (_input + 0.044715 * tf.pow(_input, 3)))))
        return _input * cdf

def ffn(_input, intermediate_size, opname=''):
    h = int(_input.get_shape()[-1])
    with tf.compat.v1.variable_scope(opname + '.' + 'ffn'):
        x = fc_layer(_input, intermediate_size, use_bias=True, opname='dense1')
        x = gelu(x)
        x = fc_layer(x, h, use_bias=True, opname='dense2')
        return x
