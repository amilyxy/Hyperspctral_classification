# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 13:29:16 2018

@author: amily
describe: 三维卷积神经网络,5层卷积层+三层全连接层
"""
import Hyperspectral_input
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
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
PCA_BAND = 10

# indian_pines data 0
indianP_origin_data = Hyperspectral_input.input_data(INDIANPDATA_DIR, 'indian_pines_corrected')
indianP_gt = Hyperspectral_input.input_data(INDIANPGT_DIR, 'indian_pines_gt')
# Pavia University data 1
paviaU_origin_data = Hyperspectral_input.input_data(PAVIAUDATA_DIR, 'paviaU')
paviaU_gt = Hyperspectral_input.input_data(PAVIAUGT_DIR, 'paviaU_gt')
#Salinas data 2
Salinas_origin_data = Hyperspectral_input.input_data(SALINASDATA_DIR, 'salinas_corrected')
Salinas_gt = Hyperspectral_input.input_data(SALINASGT_DIR, 'salinas_gt')

DATA_FLAG = 0 #默认indian pines 数据集

def accuracy_eval(label_tr, label_pred):
    overall_accuracy = metrics.accuracy_score(label_tr, label_pred)
    avarage_accuracy = np.mean(metrics.precision_score(label_tr, label_pred, average = None))
    kappa = metrics.cohen_kappa_score(label_tr, label_pred)
    cm = metrics.confusion_matrix(label_tr, label_pred)
    return overall_accuracy, avarage_accuracy, kappa, cm
'''
初始化采用 MSRA
'''    
def weight_variable(shape):
    n = reduce(lambda x, y: x*y, shape[:-1])
#    initial = tf.truncated_normal(shape, stddev = 0.1)
    initial = tf.truncated_normal(shape, mean = 0, stddev = np.sqrt(2/n))
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv3d(x, W, padmode):
    return tf.nn.conv3d(x, W, strides = [1, 1, 1, 1, 1], padding = padmode)
    #output tensor [batch, in_depth, in_height, in_width, in_channels]
    
def leaky_relu(x=None, alpha=0.1, name="lrelu"):
    x = tf.maximum(x, alpha * x, name=name)
    return x

def add_conv_layer(input_tensor, weight_dim, bias_dim, padmode = "VALID", act = tf.nn.relu):
    w_conv = weight_variable(weight_dim)
    b_conv = bias_variable([bias_dim])
    h_conv = act(conv3d(input_tensor, w_conv, padmode) + b_conv)
    return h_conv

def add_fc_layer(input_tensor, input_dim, output_dim, act=tf.nn.relu):
    w_fc = weight_variable([input_dim, output_dim])
    b_fc = bias_variable([output_dim])
    h_fc = act(tf.matmul(input_tensor, w_fc) + b_fc)
    return h_fc

def train(train_batch, train_labels, test_batch, test_labels, all_batch, all_labels, ground_truth): 
    #构造网络模型
    kind = np.unique(ground_truth).shape[0]-1
    x = tf.placeholder(tf.float32, [None, PCA_BAND, 7, 7, 1])
    y_ = tf.placeholder(tf.float32, [None, kind])
    #卷积层
    #多尺度卷积核
#    conv1_output = add_conv_layer(x, [3, 3, 3, 1, 4], 4, padmode = 'SAME')  # filter: [depth, height, width, in_channels, out_channels]
    conv1_1X1output = add_conv_layer(x, [3, 1, 1, 1, 2], 2, padmode = 'SAME')
    conv1_3X3output = add_conv_layer(x, [3, 3, 3, 1, 2], 2, padmode = 'SAME')
    conv1_5X5output = add_conv_layer(x, [3, 5, 5, 1, 2], 2, padmode = 'SAME')
#    conv2_output = add_conv_layer(conv1_output, [3, 3, 3, 4, 8], 8, padmode = 'SAME')
    conv2_1X1output = add_conv_layer(conv1_1X1output, [3, 1, 1, 2, 4], 4, padmode = 'SAME')
    conv2_3X3output = add_conv_layer(conv1_3X3output, [3, 3, 3, 2, 4], 4, padmode = 'SAME')
    conv2_5X5output = add_conv_layer(conv1_5X5output, [3, 5, 5, 2, 4], 4, padmode = 'SAME')
    
    conv2_output = tf.concat([conv2_1X1output, conv2_3X3output, conv2_5X5output], 4) 
    conv3_output = add_conv_layer(conv2_output, [3, 3, 3, 12, 16], 16)
    conv4_output = add_conv_layer(conv3_output, [3, 3, 3, 16, 32], 32)
    conv5_output = add_conv_layer(conv4_output, [3, 3, 3, 32, 64], 64)

    #全连接层
    keep_prob = tf.placeholder(tf.float32)
    fc_input = tf.reshape(conv5_output, [-1, 4*64])
    fc1_out = add_fc_layer(fc_input, 4*64, 1024)
    fc1_drop = tf.nn.dropout(fc1_out, keep_prob)
    fc2_out = add_fc_layer(fc1_drop, 1024, 256, act = tf.sigmoid)
    fc2_drop = tf.nn.dropout(fc2_out, keep_prob)
    #softmax层    
    fc3_out = add_fc_layer(fc2_drop, 256, kind, act = tf.nn.softmax)  
    
    #评估模型
    cross_entrop = -tf.reduce_sum(y_*tf.log(fc3_out))    #损失函数采用交叉熵
    lr = tf.placeholder(tf.float32)
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entrop)
    label_true = tf.argmax(y_, 1)
    label_prediction = tf.argmax(fc3_out, 1)
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
        test_accuracy = np.zeros([1,TRAIN_STEPS])    #time is complex
        learningrate = 1e-4
        for i in range(TRAIN_STEPS):
            #随机选取样本
            random_trainbatch = random.sample(range(train_num), TRAIN_BATCH)  #get random sequence
            if i <400: #400
                learningrate = 2e-3 #2e-3
            elif i < 600: #600
                learningrate = 1e-3 #1e-3
            elif i < 800: #800
                learningrate = 5e-4 #5e-4
            else:
                learningrate = 1e-4 #1e-4
            train_feed = {
                x: train_batch[random_trainbatch, :, :, :, :],
                y_: train_labels[random_trainbatch, :],
                keep_prob: 0.5,
                lr: learningrate
                } 
            sess.run(train_step, feed_dict = train_feed)
            train_accuracy[0,i], train_loss[0,i] = sess.run([accuracy, cross_entrop], feed_dict = train_feed)
            #命令行测试输出
            if i%100 == 0 :
                print ("after %d training steps , train_accuracy :%g" % (i, train_accuracy[0,i])) 
        #每次trian完之后test(对比曲线看over-fitting和under-fitting)
        test_start = test_end = total_accuracy = test_times = 0
        label_tr = np.zeros([test_num,1])
        label_pred = np.zeros([test_num,1])
        while (test_end != test_num):
            test_start = test_end
            test_end = min(test_start+TEST_BATCH, test_num)
            test_feed = {
                x: test_batch[test_start:test_end, :, :, :, :],
                y_: test_labels[test_start:test_end, :],
                keep_prob: 1.0,
                lr: learningrate
                    }
            total_accuracy = sess.run(accuracy, feed_dict = test_feed)+total_accuracy
            label_temp1, label_temp2 = sess.run([label_true, label_prediction], feed_dict = test_feed)
            label_tr[test_start:test_end, 0] = label_temp1[:]
            label_pred[test_start:test_end, 0] = label_temp2[:]
        total_accuracy = total_accuracy/test_times
        OA, AA, Kappa,cm = accuracy_eval(label_tr, label_pred)
#        np.savetxt('cm.csv', cm, delimiter = ',')
#        print(cm)
#            test_accuracy[0,i] = total_accuracy
        print ("all %d samples has the total_accuracy :%g, OA: %g, AA: %g, Kappa:%g " % (test_num, total_accuracy, OA, AA, Kappa))
        Hyperspectral_input.dis_curves('accuracy', train_accuracy[0, :],  xlabel = 'times', ylabel = 'accuracy', label = 'train')
#        Hyperspectral_input.dis_curves('accuracy', test_accuracy[0, :], xlabel = 'times', ylabel = 'accuracy', label = 'test')
        Hyperspectral_input.dis_curves('train_loss', train_loss[0, :], xlabel = 'times', ylabel = 'loss')  
         
        #restore the ground-truth 
        all_start = all_end = all_accuracy = all_times = 0
        all_num = all_batch.shape[0]
        all_pixels = np.where(ground_truth != 0)   
        all_category = np.zeros([all_num, 1])
        restore_pic = np.zeros_like(ground_truth)
        while (all_end != all_num):
            all_times += 1
            all_start = all_end
            all_end = min(all_start+RESTORE_BATCH, all_num)
            all_feed = {
                x: all_batch[all_start:all_end, :, :, :, :],
                y_: all_labels[all_start:all_end, :],
                keep_prob: 1.0,
                lr: learningrate
                    }
            all_accuracy = sess.run(accuracy, feed_dict = all_feed) + all_accuracy
            predict_cate = sess.run(label_prediction, feed_dict = all_feed)
            all_category[all_start: all_end, 0] = predict_cate[:]
        all_accuracy = all_accuracy/all_times 
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
#    norm_data = hyperdate.spectrum_normlized() 
    Z, _, _ = hyperdate.pca_reduction(PCA_BAND) 
#    error = hyperdate.pca_chosingK(PCA_BAND)
#    print (error) 
    
    #显示真实地物分类和光谱值
#    plt.figure('原始图像')
#    plt.imshow(Salinas_origin_data[:, :, 11])
#    hyperdate.dis_Spectral('origin')  
#    norm_data = hyperdate.spectrum_normlized()
#    hypernorm = Hyperspectral_input.Hyperimage(norm_data, Salinas_gt)
#    hypernorm.dis_Spectral('norm')  #需要创建实例
    
    #得到训练和测试样本的像素点并显示
    (train_pixels,test_pixels) = hyperdate.batch_select(25)
#    Hyperspectral_input.dis_groundtruth(gt_data, 'ground_truth')
#    Hyperspectral_input.dis_groundtruth(train_pixels, 'train_pixels')
#    Hyperspectral_input.dis_groundtruth(test_pixels, 'test_pixels')   
    
    #得到训练的总batch
    (train_batch, train_labels) = Hyperspectral_input.get_batchs(Z, train_pixels)
    (test_batch, test_labels) = Hyperspectral_input.get_batchs(Z, test_pixels) 
    (all_batch, all_labels) = Hyperspectral_input.get_batchs(Z, gt_data) 
    
    train(train_batch, train_labels, test_batch, test_labels, all_batch, all_labels, gt_data)
    
    # 计算程序结束时间
    end = time.clock()
    print("running time is %g min" % ((end-start)/60))       
    print("*****************分界线********************")    
    sys.exit()
 
if __name__ == "__main__":
    tf.app.run()    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    