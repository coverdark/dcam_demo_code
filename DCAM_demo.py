
# coding: utf-8

# In[ ]:


# Deep Clustering-based Aggregation Model (DCAM)
import numpy as np
import deep_laa_support as dls
import random
import sys
import tensorflow as tf
from sklearn.cluster import KMeans

#================= read data =====================
# filename = 'default_file'
data_all = np.load(filename +'.npz')
print('File ' + filename + '.npz ' 'loaded.')
user_labels = data_all['user_labels']
true_labels = data_all['true_labels']
category_size = data_all['category_num']
source_num = data_all['source_num']
feature = data_all['feature']
_, feature_size = np.shape(feature)
n_samples, _ = np.shape(true_labels)

#================= basic parameters =====================
# define batch size (use all samples in one batch)
batch_size = n_samples
cluster_num = 200

if np.max(feature) <= 1 and np.min(feature) >= 0:
    flag_node_type = 'Bernoulli'
else:
    flag_node_type = 'Gaussian'
print(flag_node_type + ' output nodes are used.')

#================= clustering regularizer l_cr =====================
with tf.name_scope('regularizer'):
    #================= reconstruction loss l_r =====================
    with tf.variable_scope('autoencoder'):
        x = tf.placeholder(dtype=tf.float32, shape=[batch_size, feature_size], name='x_input')
        h1_size_encoder = int(np.floor(feature_size/2.0))
        h2_size_encoder = 100
        embedding_size = 40
        h1_size_decoder = 100
        h2_size_decoder = int(np.floor(feature_size/2.0))
        
        with tf.variable_scope('feature_encoder_h1'):
            _h1_encoder, w1_encoder, b1_encoder = dls.full_connect_relu_BN(x, [feature_size, h1_size_encoder])
        with tf.variable_scope('feature_encoder_h2'):
            _h2_encoder, w2_encoder, b2_encoder = dls.full_connect_relu_BN(_h1_encoder, [h1_size_encoder, h2_size_encoder])
        with tf.variable_scope('feature_encoder_mu'):
            mu_h, w_mu_encoder, b_mu_encoder = dls.full_connect(_h2_encoder, [h2_size_encoder, embedding_size])
        with tf.variable_scope('feature_decoder_h1'):
            _h1_decoder, w1_decoder, b1_decoder = dls.full_connect_relu_BN(mu_h, [embedding_size, h1_size_decoder])
        with tf.variable_scope('feature_decoder_h2'):
            _h2_decoder, w2_decoder, b2_decoder = dls.full_connect_relu_BN(_h1_decoder, [h1_size_decoder, h2_size_decoder])
        with tf.variable_scope('feature_decoder_rho'):
            if flag_node_type == 'Bernoulli':
                x_reconstr, w_rho_decoder, b_rho_decoder = dls.full_connect_sigmoid(_h2_decoder, [h2_size_decoder, feature_size])
            elif flag_node_type == 'Gaussian':
                x_reconstr, w_rho_decoder, b_rho_decoder = dls.full_connect(_h2_decoder, [h2_size_decoder, feature_size])
        
        # Bernoulli
        loss_cross_entropy_AE = -tf.reduce_mean(tf.reduce_sum(x*tf.log(1e-10+x_reconstr) + (1.0-x)*tf.log(1e-10+(1.0-x_reconstr)), -1))
        # Gaussian
        loss_square_AE = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(x_reconstr - x), -1))
        # constraint on weights and biases
        constraint_w_AE = 0.5 * (tf.reduce_mean(tf.square(w1_encoder)) + tf.reduce_mean(tf.square(b1_encoder))
            + tf.reduce_mean(tf.square(w2_encoder)) + tf.reduce_mean(tf.square(b2_encoder))
            + tf.reduce_mean(tf.square(w_mu_encoder)) + tf.reduce_mean(tf.square(b_mu_encoder))
            + tf.reduce_mean(tf.square(w1_decoder)) + tf.reduce_mean(tf.square(b1_decoder))
            + tf.reduce_mean(tf.square(w2_decoder)) + tf.reduce_mean(tf.square(b2_decoder))
            + tf.reduce_mean(tf.square(w_rho_decoder)) + tf.reduce_mean(tf.square(b_rho_decoder)))
        
        # loss_AE
        if flag_node_type == 'Bernoulli':
            loss_AE = loss_cross_entropy_AE + constraint_w_AE
        elif flag_node_type == 'Gaussian':
            loss_AE = loss_square_AE + constraint_w_AE

        # pre-train autoencoder
        learning_rate_AE = 0.02
        optimizer_AE_minimize = tf.train.AdamOptimizer(learning_rate=learning_rate_AE).minimize(loss_AE)
        
        loss_reconstr = loss_AE
        
    #================= clustering loss l_c =====================
    with tf.variable_scope('clustering'):
        mu_c = tf.get_variable('mu_c', dtype=tf.float32, initializer=tf.random_normal(shape=[cluster_num, embedding_size], mean=0, stddev=1, dtype=tf.float32))
        mu_c_assign = tf.placeholder(dtype=tf.float32, shape=[cluster_num, embedding_size], name='mu_c_assign')
        initialize_mu_c = tf.assign(mu_c, mu_c_assign)

        mu_c_prior = tf.placeholder(dtype=tf.float32, shape=[cluster_num, embedding_size], name='mu_c_prior')
        prior_mu_c = -0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(mu_c - mu_c_prior), -1))
        
        # clustering distribution: q[batch_size, cluster_num]
        square_dist = tf.reduce_sum(tf.square(tf.reshape(mu_h, [batch_size, 1, embedding_size]) - mu_c), -1)
        nu = 1
        _q = (1 + square_dist/nu) ** (-(nu+1)/2)
        q = _q / (1e-10 + tf.reduce_sum(_q, -1, keepdims=True))

        # set s as a constant learning target
        fixed_s = tf.placeholder(dtype=tf.float32, shape=[batch_size, cluster_num], name='fixed_s')

        loss_clustering = -tf.reduce_mean(tf.reduce_sum(fixed_s * tf.log(1e-10 + q), -1))
        
    # loss clustering regularizer, given 
    loss_cr = loss_clustering + loss_reconstr
    
    print('Clustering regularizer is constructed.')

#================= likelihood p(l|y), p(y|z), and p(z|pi_z) =====================
with tf.name_scope('likelihood'):
    #================= p(z) =====================
    with tf.variable_scope('p_z'):
        # square_dist
        _log_pi_z = -square_dist / 2
        _log_pi_z_max = tf.reduce_max(_log_pi_z, 1, keepdims=True)
        pi_z = tf.exp(_log_pi_z - (_log_pi_z_max + tf.log(1e-10+tf.reduce_sum(tf.exp(_log_pi_z-_log_pi_z_max), 1, keepdims=True))))
        p_z = pi_z

    #================= p(y|z) =====================
    with tf.variable_scope('p_yz'):
        _pi_yz = tf.get_variable('pi_yz', dtype=tf.float32, 
                                initializer=tf.random_normal(shape=[cluster_num, category_size], mean=0, stddev=1, dtype=tf.float32))
        __pi_yz = tf.clip_by_value(_pi_yz, 1e-10, 1)
        pi_yz = __pi_yz / tf.reduce_sum(__pi_yz, -1, keepdims=True)
        
        # initialize pi_yz
        pi_yz_assign = tf.placeholder(dtype=tf.float32, shape=[cluster_num, category_size], name='pi_yz_assign')
        initialize_pi_yz = tf.assign(_pi_yz, pi_yz_assign)
        
        # prior for pi_yz
        pi_yz_prior = tf.placeholder(dtype=tf.float32, shape=[cluster_num, category_size], name='pi_yz_prior')
        prior_pi_yz = tf.reduce_mean(tf.reduce_sum(pi_yz_prior * tf.log(1e-10+pi_yz), -1))
        
    #================= p(l|y) =====================
    with tf.variable_scope('p_ly'):
        # l_reconstr[category_size, 1, source_num*category_size]
        output_size = source_num * category_size
        l = tf.placeholder(dtype=tf.float32, shape=[batch_size, output_size], name='l_input')
        weights_reconstr = tf.get_variable('weights_reconstr', dtype=tf.float32,
                                           initializer=tf.truncated_normal(shape=[source_num, category_size, category_size], mean=0.0, stddev=.01))
        biases_reconstr = tf.get_variable('biases_reconstr', dtype=tf.float32, initializer=tf.zeros(shape=[source_num, category_size]))
        w = weights_reconstr
        
        constant_y = dls.get_constant_y(1, category_size)
        _l_reconstr = []
        for i in range(category_size):
            _reconstr_tmp = constant_y[i, :, :] * w
            _reconstr_tmp = tf.reduce_sum(_reconstr_tmp, -1)
            _reconstr_tmp = _reconstr_tmp + biases_reconstr
            _reconstr_tmp = tf.exp(_reconstr_tmp)
            _reconstr_tmp = tf.div(_reconstr_tmp, tf.reduce_sum(_reconstr_tmp, -1, keepdims=True))
            _l_reconstr.append(tf.reshape(_reconstr_tmp, [1, -1]))
        l_reconstr = tf.stack(_l_reconstr)

        prior_w_ly = -0.5 * (tf.reduce_mean(tf.square(weights_reconstr)) + tf.reduce_mean(tf.square(biases_reconstr)))
    
        # _tmp_cross_entropy[category_size, batch_size, output_size]
        _tmp_cross_entropy_ly = -l * tf.log(1e-10 + l_reconstr)
        # cross_entropy_reconstr[batch_size, category_size]
        cross_entropy_reconstr_ly = tf.transpose(tf.reduce_sum(_tmp_cross_entropy_ly, 2))
        log_p_ly = - cross_entropy_reconstr_ly
    
    #================= p(l|y) p(y|z) p(z) =====================
    # log_p_ly[batch_size, category_size]
    reshaped_log_p_ly = tf.reshape(log_p_ly, [batch_size, 1, category_size])
    # p_yz[cluster_num, 1, cateogry_size]
    reshaped_p_yz = tf.reshape(pi_yz, [1, cluster_num, category_size])
    # p_z[batch_size, cluster_num]
    reshaped_p_z = tf.reshape(p_z, [batch_size, cluster_num, 1])
    # p_ly_p_yz_p_z[batch_size, cluster_num, category_size]
    p_ly_p_yz_p_z = tf.exp(reshaped_log_p_ly) * reshaped_p_yz * reshaped_p_z
    
    # inferred_y[batch_size, category_size]
    _inferred_y = tf.reduce_sum(p_ly_p_yz_p_z, 1)
    inferred_y = _inferred_y / (1e-10 + tf.reduce_sum(_inferred_y, -1, keepdims=True))
    
    log_likelihood = tf.reduce_mean(tf.log(1e-10 + tf.reduce_sum(tf.reduce_sum(p_ly_p_yz_p_z, 1), -1)))

    print('Log-likelihood is constructed.')

#================= loss overall =====================
with tf.name_scope('loss_overall'):
    # loss_cr
    # log_likelihood
    # prior_w_ly
    # prior_pi_yz
    loss_DCAM = -log_likelihood + loss_cr - prior_w_ly - prior_pi_yz - prior_mu_c

    learning_rate_DCAM = tf.placeholder(dtype=tf.float32, name='learning_rate_DCAM')
    optimizer_DCAM = tf.train.AdamOptimizer(learning_rate=learning_rate_DCAM)
    optimizer_DCAM_minimize = optimizer_DCAM.minimize(loss_DCAM)
    reset_optimizer_DCAM = tf.variables_initializer(optimizer_DCAM.variables())

    saver = tf.train.Saver()
    
    print('DCAM is constructed.')


# In[ ]:


#================= training and inference =====================
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # assign batch variables (use whole data in one batch)
    
    print("Pre-train autoencoder ...")
    epochs = 2000
    for epoch in range(epochs):
        _, monitor_loss_AE = sess.run([optimizer_AE_minimize, loss_AE], feed_dict={x:feature})
        if epoch % 50 == 0:
            print("epoch: {0} loss: {1}".format(epoch, monitor_loss_AE))    
    
    #================= calculate initial parameters =====================
    initial_mu_h = sess.run(mu_h, feed_dict={x:feature})
    clustering_result = KMeans(n_clusters=cluster_num).fit(initial_mu_h)
    
    #================= save current model =====================
    saved_path = saver.save(sess, './my_model')


# In[ ]:


def smooth_one_hot_s(input_q, smooth=0.0):
    max_idx = np.argmax(input_q, -1)
    s = dls.convert_to_one_hot(max_idx, cluster_num, smooth*(cluster_num-1))
    return s

with tf.Session() as sess:
    saver.restore(sess, './my_model')
    # initialize mu_c as clustering_result.cluster_centers_
    _ = sess.run(initialize_mu_c, {mu_c_assign:clustering_result.cluster_centers_})
    # initialize pi_yz as pi_yz_prior_cluster
    pi_yz_prior_cluster = dls.get_cluster_majority_y(
        clustering_result.labels_, user_labels, cluster_num, source_num, category_size)
    _ = sess.run(initialize_pi_yz, {pi_yz_assign:pi_yz_prior_cluster})
    
    print("Train DCAM ...")
    epochs = 1000
    learning_rate_overall = 0.0001
    for epoch in range(epochs):
        # calculate s
        monitor_q = sess.run(q, feed_dict={x:feature})
        s = smooth_one_hot_s(monitor_q, 1e-6)
        
        # reset optimizer given new s
        sess.run(reset_optimizer_DCAM)
        # optimize DCAM 
        _, monitor_loss_DCAM = sess.run(
            [optimizer_DCAM_minimize, loss_DCAM], 
            feed_dict={x:feature, l:user_labels, 
                       learning_rate_DCAM:learning_rate_overall, 
                       fixed_s:s, 
                       pi_yz_prior:pi_yz_prior_cluster,
                       mu_c_prior:clustering_result.cluster_centers_})

        # inference
        monitor_inferred_y = sess.run(inferred_y, feed_dict={x:feature, l:user_labels})
        hit_num_inferred_y = dls.cal_hit_num(true_labels, monitor_inferred_y)
        
        if epoch % 10 == 0:
            print("epoch: {0} loss: {1}".format(epoch, monitor_loss_DCAM))
            print("epoch: {0} accuracy: {1}".format(epoch, float(hit_num_inferred_y)/n_samples))
        
    print("Training DCAM is Done!")

