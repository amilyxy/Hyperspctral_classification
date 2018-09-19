# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 21:10:49 2018

@author: amily
"""
import Hyperspectral_input
import numpy as np
import tensorflow as tf
#from matplotlib import pyplot as plt
import sys
import time
import random
from sklearn import metrics
from functools import reduce 

# the training data dir
INDIANPDATA_DIR = './data/Indian_pines_corrected.mat' 
INDIANPGT_DIR = './data/Indian_pines_gt.mat'
PAVIAUDATA_DIR = './data/PaviaU.mat' 
PAVIAUGT_DIR = './data/PaviaU_gt.mat'
SALINASDATA_DIR = './data/Salinas_corrected.mat' 
SALINASGT_DIR = './data/salinas_gt.mat'

TRAIN_STEPS = 1000
TRAIN_BATCH = 128
RESTORE_BATCH = 10000
TEST_BATCH = 10000
INIT_LR = 1e-3

# indian_pines data
indianP_origin_data = Hyperspectral_input.input_data(INDIANPDATA_DIR, 'indian_pines_corrected')
indianP_gt = Hyperspectral_input.input_data(INDIANPGT_DIR, 'indian_pines_gt')
# Pavia University data
paviaU_origin_data = Hyperspectral_input.input_data(PAVIAUDATA_DIR, 'paviaU')
paviaU_gt = Hyperspectral_input.input_data(PAVIAUGT_DIR, 'paviaU_gt')
#Salinas data
Salinas_origin_data = Hyperspectral_input.input_data(SALINASDATA_DIR, 'salinas_corrected')
Salinas_gt = Hyperspectral_input.input_data(SALINASGT_DIR, 'salinas_gt')

DATA_FLAG = 2 #默认indian pines 数据集

def accuracy_eval(label_tr, label_pred):
    overall_accuracy = metrics.accuracy_score(label_tr, label_pred)
    avarage_accuracy = np.mean(metrics.precision_score(label_tr, label_pred, average = None))
    kappa = metrics.cohen_kappa_score(label_tr, label_pred)
    cm = metrics.confusion_matrix(label_tr, label_pred)
    return overall_accuracy, avarage_accuracy, kappa, cm
    
def weight_variable(shape):
    n = reduce(lambda x, y: x*y, shape[:-1])
#    initial = tf.truncated_normal(shape, stddev = 0.1)
    initial = tf.truncated_normal(shape, mean = 0, stddev = np.sqrt(2/n))
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = "SAME")

def max_pool_2X2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def add_conv_layer(input_tensor, weight_dim, bias_dim, act = tf.nn.relu, pool_flag = 0):
    W_conv = weight_variable(weight_dim)
    b_conv = bias_variable([bias_dim])
    h_conv = act(conv2d(input_tensor, W_conv) + b_conv)
    if pool_flag :
        h_pool = max_pool_2X2(h_conv)
        return h_pool
    else:
        return h_conv

def add_fc_layer(input_tensor, input_dim, output_dim, act = tf.nn.relu):
    W_fc = weight_variable([input_dim, output_dim])
    b_fc = bias_variable([output_dim])
    h_fc = act(tf.matmul(input_tensor, W_fc) + b_fc)
    return h_fc

def train(train_batch, train_labels, test_batch, test_labels, all_batch, all_labels, ground_truth): 
    #构造模型
    x = tf.placeholder(tf.float32, [None, 14, 14, 1])
    y_ = tf.placeholder(tf.float32, [None, 16])
    #卷积层
    conv1_output = add_conv_layer(x, [3, 3, 1, 2], 2)
    conv2_output = add_conv_layer(conv1_output, [3, 3, 2, 4], 4)
    conv3_output = add_conv_layer(conv2_output, [3, 3, 4, 8], 8)
    conv4_output = add_conv_layer(conv3_output, [3, 3, 8, 16], 16)
    conv5_output = add_conv_layer(conv4_output, [3, 3, 16, 32], 32, pool_flag = 1)
    #全连接层
    conv5_flat = tf.reshape(conv5_output, [-1, 7*7*32])
    keep_prob = tf.placeholder(tf.float32)
    fc1_output = add_fc_layer(conv5_flat, 7*7*32, 1024)
    fc1_drop = tf.nn.dropout(fc1_output, keep_prob)
    fc2_output = add_fc_layer(fc1_drop, 1024, 256, act = tf.sigmoid)
    fc2_drop = tf.nn.dropout(fc2_output, keep_prob)
    fc3_output = add_fc_layer(fc2_drop, 256, 16, act = tf.nn.softmax)
    #train
    cross_entrop = -tf.reduce_sum(y_*tf.log(fc3_output))
    lr = tf.placeholder(tf.float32)
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entrop)
    label_true = tf.argmax(y_, 1)
    label_prediction = tf.argmax(fc3_output, 1)
    correct_prediction = tf.equal(label_prediction, label_true)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    with tf.Session() as sess:
        #初始化变量
        init = tf.global_variables_initializer()
        sess.run(init)
        #准备数据集
        train_num = train_batch.shape[0]
        test_num = test_batch.shape[0]
        train_accuracy = np.zeros([1,TRAIN_STEPS])
        train_loss = np.zeros([1,TRAIN_STEPS])
        test_accuracy = np.zeros([1,TRAIN_STEPS])   
#        learning_rate = 5e-3
        for i in range(TRAIN_STEPS):
            if i < 200:
                learning_rate = 8e-3
            elif i < 500:
                learning_rate = 5e-3
            else:
                learning_rate = 2e-3
#            随机选取样本
            random_trainbatch = random.sample(range(train_num), TRAIN_BATCH)  #get random sequence
            #dropout=0.5训练网络 防止过拟合
            train_feed = {
                x: train_batch[random_trainbatch, :, :, :],
                y_: train_labels[random_trainbatch, :],
                keep_prob: 0.5,
                lr: learning_rate
                } 
            sess.run(train_step, feed_dict = train_feed)
            train_accuracy[0,i], train_loss[0,i] = sess.run([accuracy, cross_entrop], feed_dict = train_feed)
            #命令行测试输出
            if i%100 == 0 :
                print ("after %d training steps , train_accuracy :%g" % (i, train_accuracy[0,i]))  
#        每次trian完之后test(对比曲线看over-fitting和under-fitting)
        test_feed = {
            x: test_batch[:, :, :, :],
            y_: test_labels[:, :],
            keep_prob: 1.0,
            lr: learning_rate
                }
        total_accuracy = sess.run(accuracy, feed_dict = test_feed)
        test_accuracy[0,i] = total_accuracy
        if i == 999:
             label_tr, label_pred = sess.run([label_true, label_prediction], feed_dict = test_feed)
             OA, AA, Kappa, cm = accuracy_eval(label_tr, label_pred)
             print(cm)
        print ("all %d samples has the total_accuracy :%g, OA: %g, AA: %g, Kappa:%g " % (test_num, total_accuracy, OA, AA, Kappa))
#        Hyperspectral_input.dis_curves('accuracy', train_accuracy[0, :],  xlabel = 'times', ylabel = 'accuracy', label = 'train')
#        Hyperspectral_input.dis_curves('accuracy', test_accuracy[0, :], xlabel = 'times', ylabel = 'accuracy', label = 'test')
#        Hyperspectral_input.dis_curves('train_loss', train_loss[0, :], xlabel = 'times', ylabel = 'loss')  
#         
        #restore the ground-truth 
        all_num = all_batch.shape[0]
        all_pixels = np.where(ground_truth != 0)   
        all_category = np.zeros([all_num, 1])
        restore_pic = np.zeros_like(ground_truth)
        all_feed = {
            x: all_batch[:, :, :, :],
            y_: all_labels[:, :],
            keep_prob: 1.0
                }
        all_accuracy = sess.run(accuracy, feed_dict = all_feed)
        predict_cate = sess.run(label_prediction, feed_dict = all_feed)
        all_category[:, 0] = predict_cate[:]
        for k in range(all_num):
            row = all_pixels[0][k]
            col = all_pixels[1][k]
            restore_pic[row, col] = all_category[k, 0]+1
        Hyperspectral_input.dis_groundtruth(restore_pic, 'restore_pic')
        print ("all %d samples has the total_accuracy :%g" % (all_num, all_accuracy))
         
def main(argv=None):
    # 计算开始时间
    start = time.clock()
    #选择数据集 /indian pines/pavia university
    if DATA_FLAG == 0:
        hyperdate = Hyperspectral_input.Hyperimage(indianP_origin_data, indianP_gt) 
        gt_data = indianP_gt 
    elif DATA_FLAG == 1:
        hyperdate = Hyperspectral_input.Hyperimage(paviaU_origin_data, paviaU_gt) 
        gt_data = paviaU_gt 
    elif DATA_FLAG == 2:
        hyperdate = Hyperspectral_input.Hyperimage(Salinas_origin_data, Salinas_gt) 
        gt_data = Salinas_gt  
    
    #归一化输入数据
    norm_data = hyperdate.spectrum_normlized() 
    
    #得到训练和测试样本的像素点并显示
    (train_pixels,test_pixels) = hyperdate.batch_select(25)   
    
    #得到训练的总batch
    (train_batch, train_labels) = Hyperspectral_input.get_2dimbatches(norm_data, train_pixels)
    (test_batch, test_labels) = Hyperspectral_input.get_2dimbatches(norm_data, test_pixels) 
    (all_batch, all_labels) = Hyperspectral_input.get_2dimbatches(norm_data, Salinas_gt) 
    
    train(train_batch, train_labels, test_batch, test_labels, all_batch, all_labels, gt_data)

    # 计算程序结束时间
    end = time.clock()
    print("running time is %g min" % ((end-start)/60))       
    print("*****************分界线********************")    
    sys.exit(0)
 
if __name__ == "__main__":
    tf.app.run()  
    

















