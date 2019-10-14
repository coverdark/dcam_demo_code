
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf
import numpy.matlib
import scipy.io
import scipy.sparse
import os

def convert_mat_to_one_hot_npz(read_file, write_file=""):
    data = scipy.io.loadmat(read_file)
    data_num, source_num = np.shape(data['L'])
    tmp_true_labels = data['true_labels']
    tmp_user_labels = data['L']
    
    # 'true_labels'
    tmp_min_category = np.min(tmp_true_labels)
    tmp_true_labels -= tmp_min_category
    category_num = np.max(tmp_true_labels) + 1
    true_labels = np.reshape(tmp_true_labels, (data_num, 1))
    
    # 'user_labels' & 'label_mask'
    user_labels = np.zeros((data_num, source_num*category_num))
    label_mask = np.zeros((data_num, source_num*category_num))
    for i in range(data_num):
        for j in range(source_num):
            tmp_label = tmp_user_labels[i, j]
            if tmp_label > 0:
                target_label = tmp_label - tmp_min_category
                user_labels[i, j*category_num+target_label] = 1
                label_mask[i, j*category_num:(j+1)*category_num] = 1
                
    # 'feature'
    feature = data['feature']
    
    if write_file == "":
        write_file = os.path.basename(read_file)
    np.savez(write_file, 
             true_labels=true_labels, 
             user_labels=user_labels, 
             label_mask=label_mask, 
             feature=feature, 
             category_num=category_num, 
             source_num=source_num)
    
def get_label_num(user_labels):
    '''
    return label_num_per_sample[n_samples], total_label_num
    '''
    label_num_per_sample = np.sum(user_labels, 1)
    total_label_num = np.sum(label_num_per_sample)
    return label_num_per_sample, total_label_num

def get_constant_y(batch_size, category_size):
    '''
    return constant_y as [category_size, batch_size, category_size]
    the last two dimensions construct a matrix of batch_size vetors
    with only one element being 1
    '''
    constant_y = np.zeros([category_size, batch_size, category_size], dtype=np.float32)
    for i in range(category_size):
        constant_y[i, :, i] = 1.0
    return constant_y

def get_majority_y(user_labels, source_num, category_num):
    if not scipy.sparse.issparse(user_labels):
        tmp = np.eye(category_num)
        template = np.matlib.repmat(tmp, source_num, 1)
        majority_y = np.matmul(user_labels, template)
        majority_y = np.divide(majority_y, np.sum(majority_y, 1, keepdims=True))
    else:
        user_labels = user_labels.todense()
        tmp = np.eye(category_num)
        template = np.matlib.repmat(tmp, source_num, 1)
        majority_y = np.matmul(user_labels, template)
        majority_y = np.divide(majority_y, np.sum(majority_y, 1, keepdims=True))
    return majority_y

def get_source_wise_template(input_size, category_size):
    source_wise_template = np.zeros((input_size, input_size), dtype=np.float32)
    for i in range(input_size):
        source_wise_template[i*category_size:(i+1)*category_size, i*category_size:(i+1)*category_size] = 1
    return source_wise_template

def get_uniform_prior(n_samples, category_num):
    uniform_prior = np.ones((n_samples, category_num), dtype=np.float32) / category_num
    return uniform_prior    

def gen_data(filename, data_num, source_num, category_num):
    data_label_vectors = np.zeros((data_num, source_num*category_num))
    _tmp = np.random.multinomial(1, [1./category_num]*category_num, size=data_num*source_num)
    for i in range(data_num):
        for j in range(source_num):
            data_label_vectors[i, category_num*j:category_num*(j+1)] = _tmp[i*source_num+j, :]
    data_y_labels = np.argmax(np.random.multinomial(1, [1./category_num]*category_num, size=data_num), axis=1)
    np.savez(filename, data=data_label_vectors, labels=np.reshape(data_y_labels, (data_num, 1)))

def gen_rand_sample_index(n_samples, batch_size):
    rand_sample_index = np.random.permutation(n_samples)
    sample_index = []
    current_index = 0
    while current_index+batch_size <= n_samples:
        rand_sample_index[current_index:current_index+batch_size]
        sample_index.append(rand_sample_index[current_index:current_index+batch_size])
        current_index = current_index + batch_size
    if current_index < n_samples:
        sample_index.append(rand_sample_index[current_index:])
    return sample_index
    
#================= basic connection type in networks =====================
def full_connect(inputs, weights_shape):
    weights = tf.get_variable('weights', dtype=tf.float32,
        initializer=tf.truncated_normal(weights_shape, mean=0.0, stddev=.01))
    biases = tf.get_variable('biases', dtype=tf.float32,
        initializer=tf.zeros(weights_shape[1], dtype=tf.float32))
    return tf.matmul(inputs, weights)+biases, weights, biases

def full_connect_softmax(inputs, weights_shape):
    results, weights, biases = full_connect(inputs, weights_shape)
    return tf.nn.softmax(results, -1), weights, biases

def full_connect_relu(inputs, weights_shape):
    results, weights, biases = full_connect(inputs, weights_shape)
    return tf.nn.relu(results), weights, biases

def full_connect_relu_BN(inputs, weights_shape):
    results, weights, biases = full_connect(inputs, weights_shape)
    _tmp_results = tf.contrib.layers.batch_norm(results, center=True, scale=True, is_training=True)
    return tf.nn.relu(_tmp_results), weights, biases

def full_connect_sigmoid(inputs, weights_shape):
    results, weights, biases = full_connect(inputs, weights_shape)
    return tf.nn.sigmoid(results), weights, biases

def full_connect_tanh(inputs, weights_shape):
    results, weights, biases = full_connect(inputs, weights_shape)
    return tf.nn.tanh(results), weights, biases

def full_connect_sigmoid_BN(inputs, weights_shape):
    results, weights, biases = full_connect(inputs, weights_shape)
    _tmp_results = tf.contrib.layers.batch_norm(results, center=True, scale=True, is_training=True)
    return tf.nn.sigmoid(_tmp_results), weights, biases

#================= evaluation =====================
# evaluate with true labels
def cal_hit_num(true_labels, y_inferred):
    '''
    return hit_num
    '''
    n_samples, _ = np.shape(true_labels)
    hit_num = np.sum(np.equal(
        np.reshape(true_labels, -1), np.argmax(y_inferred, 1)))
    return hit_num

#================= LAA =====================
def LAA_encoder(v, batch_size, source_num, category_size):
    '''
    inputs to y_classifier (label distribution)
    inputs: v[batch_size, source_num*category_size]
    outputs: y_classifier[batch_size, category_size]
    '''
    v_reshape = tf.reshape(v, [batch_size, source_num, 1, category_size])
    weights_classifier = tf.get_variable('weights_classifier', dtype=tf.float32, 
                                         initializer=tf.truncated_normal([source_num, category_size,category_size], mean=0.0, stddev=.01))
    biases_classifier = tf.get_variable('biases_classifier', dtype=tf.float32,
                                        initializer=tf.zeros([category_size], dtype=tf.float32))
    _tmp = v_reshape * weights_classifier
    _tmp = tf.reduce_sum(tf.reduce_sum(_tmp, -1), 1) + biases_classifier
    _tmp = tf.exp(_tmp)
    y_classifier = tf.div(_tmp, tf.reduce_sum(_tmp, -1, keepdims=True))
    return y_classifier, weights_classifier, biases_classifier

def LAA_loss_classifier(y_classifier, y_classifier_target):
    '''
    return cross entropy loss between y_classifier and y_classifier_target
    inputs
        y_classifier_target[batch_size, category_size]
        y_classifier[batch_size, category_size]
    '''
    _tmp_classifier_cross_entropy = - y_classifier_target * tf.log(1e-10 + y_classifier)
    loss_classifier = tf.reduce_mean(tf.reduce_sum(_tmp_classifier_cross_entropy, 1))
    return loss_classifier
    
def LAA_decoder(source_num, category_size):
    '''
    sourcewise reconstruction from constant samples.
    return
        v_reconstr with size [category_size, 1, reconstr_size]
        weights_reconstr
        biases_reconstr
    constant samples are vectors with only one element being 1.
    '''
    weights_reconstr = tf.get_variable('weights_reconstr', dtype=tf.float32,
                                       initializer=tf.truncated_normal(shape=[source_num, category_size, category_size], mean=0.0, stddev=.01))
    biases_reconstr = tf.get_variable('biases_reconstr', dtype=tf.float32,
                                      initializer=tf.zeros(shape=[source_num, category_size], dtype=tf.float32))
    constant_y = get_constant_y(1, category_size)
    _v_reconstr = []
    for i in range(category_size):
        _reconstr_tmp = constant_y[i, :, :] * weights_reconstr
        _reconstr_tmp = tf.reduce_sum(_reconstr_tmp, -1)
        _reconstr_tmp = _reconstr_tmp + biases_reconstr
        _reconstr_tmp = tf.exp(_reconstr_tmp)
        _reconstr_tmp = tf.div(_reconstr_tmp, tf.reduce_sum(_reconstr_tmp, -1, keepdims=True))
        _v_reconstr.append(tf.reshape(_reconstr_tmp, [1, -1]))
    v_reconstr = tf.stack(_v_reconstr)
    return v_reconstr, weights_reconstr, biases_reconstr

def decoder_constant_y(reconstr_size, category_size):
    '''
    Decoder from constant samples.
    return
        v_reconstr with size [category_size, 1, reconstr_size]
        weights_reconstr
        biases_reconstr
    constant samples are vectors with only one element being 1.
    '''
    weights_reconstr = tf.get_variable('weights_reconstr', dtype=tf.float32,
                                       initializer=tf.truncated_normal(shape=[category_size, reconstr_size], mean=0.0, stddev=.01))
    biases_reconstr = tf.get_variable('biases_reconstr', dtype=tf.float32,
                                      initializer=tf.zeros(shape=[reconstr_size], dtype=tf.float32))
    constant_y = get_constant_y(1, category_size)
    _v_reconstr = []
    for i in range(category_size):
        _reconstr_tmp = tf.matmul(constant_y[i, :, :], weights_reconstr) + biases_reconstr
        _v_reconstr.append(_reconstr_tmp)
    v_reconstr = tf.stack(_v_reconstr)
    return v_reconstr, weights_reconstr, biases_reconstr
    
    
def LAA_loss_reconstr(v, v_reconstr, y_classifier):
    '''
    return cross entropy loss between v_reconstr and v, according to probability y_classifier
    inputs
        v[batch_size, source_num*category_size]
        v_reconstr[category_size, 1, source_num*category_size]
        y_classifier[batch_size, category_size]
    '''
    # _tmp_cross_entropy with size[category_size, batch_size, source_num*category_size]
    _tmp_cross_entropy = - v * tf.log(1e-10 + v_reconstr)
    # cross_entropy_reconstr with size[batch_size, category_size]
    cross_entropy_reconstr = tf.transpose(tf.reduce_sum(_tmp_cross_entropy, 2))
    _loss_reconstr = y_classifier * cross_entropy_reconstr
    loss_reconstr = tf.reduce_mean(tf.reduce_sum(_loss_reconstr, 1))
    return loss_reconstr
    
def KL_divergence(y, y_target):
    '''
    return KL divergence between y and y_target
    '''
    return tf.reduce_mean(tf.reduce_sum(y*tf.log(1e-10+y) - y*tf.log(1e-10+y_target), 1))
    
#================= VAE =====================
def vae_encoder(x, feature_size, h1_size, h2_size, embedding_size, single_sigma=False):
    '''
    vae-style encoding for input x
    return
        mu[batch_size, embedding_size]
        sigma[batch_size, embedding_size]
    inputs
        x[batch_size, feature_size]
    '''
    with tf.variable_scope('feature_net_encoder_h1'):
        h1, w1, b1 = full_connect_relu_BN(x, [feature_size, h1_size])
    with tf.variable_scope('feature_net_encoder_h2'):
        h2, w2, b2 = full_connect_relu_BN(h1, [h1_size, h2_size])
    with tf.variable_scope('feature_net_encoder_mu'):
        mu, w_mu, b_mu = full_connect(h2, [h2_size, embedding_size])
    with tf.variable_scope('feature_net_encoder_sigma'):
        if not single_sigma:
            sigma, w_sigma, b_sigma = full_connect(h2, [h2_size, embedding_size])
        else:
            sigma, w_sigma, b_sigma = full_connect(h2, [h2_size, 1])
    return mu, sigma

def vae_decoder(h, embedding_size, h1_size, h2_size, reconstr_size, single_sigma=False):
    '''
    vae-style decoding for embedding h
    return
        rho[batch_size, reconstr_size] for Bernoulli
        mu[batch_size, reconstr_size] for Gaussian
        sigma[batch_size, reconstr_size] for Gaussian
    inputs
        h[batch_size, embedding_size]
    '''
    with tf.variable_scope('feature_net_decoder_h1'):
        h1, w1, b1 = full_connect_relu_BN(h, [embedding_size, h1_size])
    with tf.variable_scope('feature_net_decoder_h2'):
        h2, w2, b2 = full_connect_relu_BN(h1, [h1_size, h2_size])
    with tf.variable_scope('feature_net_decoder_rho'):
        rho, w_rho, b_rho = full_connect_sigmoid(h2, [h2_size, reconstr_size])    
    with tf.variable_scope('feature_net_decoder_mu'):
        mu, w_mu, b_mu = full_connect(h2, [h2_size, reconstr_size])
    with tf.variable_scope('feature_net_decoder_sigma'):
        if not single_sigma:
            sigma, w_sigma, b_sigma = full_connect(h2, [h2_size, reconstr_size])
        else:
            sigma, w_sigma, b_sigma = full_connect(h2, [h2_size, 1])
    return rho, mu, sigma

def convert_to_one_hot(l, output_size, smooth=0.0):
    num = len(l)
    one_hot = np.zeros([num, output_size])
    one_hot = one_hot + smooth / output_size
    for i in range(num):
        one_hot[i, l[i]] = one_hot[i, l[i]] + (1-smooth)
    return one_hot

def get_cluster_majority_y(cluster_index, user_labels, cluster_num, source_num, category_size):
    '''
    return cluster_majority_y[cluster_num, category_size] for each cluster by gathering user labels in the cluster
    inputs
        cluster_index[n_samples]
        user_labels[n_samples, source_num*category_size]
    '''
    n_samples = len(cluster_index)
    _tmp_user_labels = np.reshape(user_labels, [n_samples, source_num, category_size])
    cluster_majority_y = np.zeros([cluster_num, category_size])
    for i in range(n_samples):
        cluster_majority_y[cluster_index[i], :] = \
            cluster_majority_y[cluster_index[i], :] + np.sum(_tmp_user_labels[i, :, :], 0, keepdims=True)
    
    cluster_majority_y = cluster_majority_y / (1e-10+np.sum(cluster_majority_y, -1, keepdims=True))
    return cluster_majority_y

def cal_majority_cluster_label(p, user_labels, cluster_num, source_num, category_size):
    argmax_p = np.argmax(p, -1)
    pi_yz_cluster = get_cluster_majority_y(argmax_p, user_labels, cluster_num, source_num, category_size)
    object_num = np.size(p, 0)
    predict_label = np.zeros([object_num, category_size])
    for i in range(object_num):
        predict_label[i, :] = pi_yz_cluster[argmax_p[i], :]
    return predict_label

def smooth_dist(p, smooth=0.01):
    _p = p + smooth
    return _p / np.sum(_p, -1, keepdims=True)