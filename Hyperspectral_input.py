# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 18:58:13 2018

@author: amily
describe: the classification of indian_pines data based on deep_learning
data: http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
"""

from scipy import io as spio
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import random

# load data  
def input_data(DATA_DIR, key):
    data = spio.loadmat(DATA_DIR)    #class: dict  print (origin_data.keys())
    return data[key]

''' get the type and shape of the data and visualize it
print (type(origin_data))       # <class 'numpy.ndarray'>
print (origin_data.shape)       # (145, 145, 200)
print (np.unique(output_data))  #[0~16]
'''
# colormap 
def colormap(kinds):
    cdict = ['#FFFFFF', '#FFDEAD', '#FFC0CB', '#FF1493', '#DC143C', '#FFD700', '#DAA520',
             '#D2691E', '#FF4500', '#00FA9A', '#00BFFF', '#6495ED', '#9932CC', '#8B008B', 
             '#228B22', '#000080', '#808080']
    cdict = cdict[:kinds]
    return colors.ListedColormap(cdict, 'indexed')

'''
functions： 显示地物类别
input：含有类别标记的矩阵, 画图的标题
output：地物分类的图片
'''
def dis_groundtruth(gt, title):
    plt.figure(title)
    plt.title(title)
    kinds = np.unique(gt).shape[0]
    plt.imshow(gt, cmap = colormap(kinds))
    plt.colorbar() 
 
'''
functions : 最简单的显示曲线, 可显示多条, 要求title一样即可
'''
def dis_curves(title, data, xlabel = '', ylabel = '',label=''):
    plt.figure(title)
    plt.title(title)
    plt.plot(data, label = label) 
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True) 
    plt.show()

'''
functions: 面向高光谱数据集的类
input: data, label (原始数据和分类标签)
'''
    
class Hyperimage(object):
    def __init__(self, data, label):
        self.Weight = data.shape[0]
        self.Height = data.shape[1]
        self.Bands = data.shape[2]
        self.Kinds = np.unique(label).shape[0]
        self.data = data
        self.label = label
    '''
    functions: 显示输入的多个波段的光谱值
    '''
    def dis_Spectral(self, name):
        Spectral_value = np.zeros([self.Kinds-1, self.Bands])
        for i in range(self.Kinds-1):
            temp1 = (self.label == (i+1))
            num = np.sum(temp1)   
            for j in range(self.Bands):
                temp2 = (np.sum(np.multiply(self.data[:, :, j], temp1)))/num
                Spectral_value[i, j] = temp2  
        plt.figure(name)
        plt.title('Spectral_distribution')
        for i in range(self.Kinds-1):
            plt.plot(Spectral_value[i,:]) 
        plt.xlabel('Band')
        plt.ylabel('Spectral_value')
        plt.grid(True) 
        plt.show()

    '''    
    #functions: 归一化输入 
    #两种方式： 1.max-min归一化 2.标准差归一化    
    '''
    def spectrum_normlized(self):
        x = np.zeros(shape = self.data.shape, dtype = 'float64')
        for i in range(self.Bands):
            temp = self.data[:, :, i]
            mean =  np.mean(temp)
            std = np.std(temp)
            x[:, :, i] = ((temp-mean)/std)
        return x
    
    '''
    functions: 主成分分析法进行数据降维
    input： 归一化数据 /[145,145,200]/mean normalization
    output： 降维之后的数据
    '''
    def pca_reduction(self, k):
        normlizeddata = self.spectrum_normlized()
        data_reshape = np.reshape(normlizeddata, [-1, self.Bands])
        m = data_reshape.shape[0]
        #协方差矩阵 [n, n] n为feature的维度
        sigma = np.dot(np.transpose(data_reshape), data_reshape)/m  
        #计算sigma的特征值和特征向量
        U, S, V = np.linalg.svd(sigma)
        #取k维
        U_reduce = U[:, 0:k]
        Z = np.dot(data_reshape, U_reduce) 
        Z = Z.reshape([self.Weight, self.Height, k])          
        return Z, U, S
    
    '''
    functions: 计算pca k维精度
    '''
    def pca_chosingK(self, k):
        _, _, S = self.pca_reduction(k)
        error = np.sum(S[0:k])/np.sum(S[:])
        return error  
    
    '''
    select train pixels and test pixels
    #input: 所有的像素点，用作训练集的像素百分比
    #output: 训练像素点和测试像素点
    '''
    def batch_select(self, pct):
        test_pixels = self.label.copy()
        train_pixels = self.label.copy()
        for i in range(self.Kinds-1):
            num = np.sum(self.label == (i+1))
            train_num = (num*pct)//100
            if train_num < 10:
                train_num = 10
            temp1 = np.where(self.label == (i+1))             #get all the i samples which has num number
            temp2 = random.sample(range(num), train_num)      #get random sequence
            for i in temp2:
                test_pixels[temp1[0][temp2],temp1[1][temp2]] = 0
        train_pixels = self.label - test_pixels
        return train_pixels, test_pixels

'''
functions: pca恢复原数据
inputs: Z: pca数据 U:特征向量 k:pca取k维数据
outputs: 原始数据的近似值
'''
def pca_recover(Z, U, k):
    X_rec = np.zeros([Z.shape[0], U.shape[0]])
    U_reduce = U[:, 0:k]
    #数据还原
    X_rec = np.dot((Z, np.transpose(U_reduce))) 
    return X_rec
    
'''
functions: 获取训练和测试集所需batches
input：总体数据,被选择的像素标签
output：所选择像素的batches batch/[batch, [batch, depth, height, width]
这个函数有点bug,data和label的padding有点矛盾,修改见后注释
'''          
def get_batchs(input_data, label_select):
    # add paddings [145, 145, 200]
    Band = input_data.shape[2]
    kind = np.unique(label_select).shape[0]-1
    paddingdata = np.pad(input_data, ((3,3),(3,3),(0,0)), "edge")  # 采用边缘值填充 [203, 203, 200]
    paddinglabel = np.pad(label_select, ((3,3),(3,3)), "constant")    # 此处"constant"应改为"edge"
    #得到 label的 pixel坐标位置
    pixel = np.where(paddinglabel != 0)         #pixel = np.where(label_select != 0)  
    #the number of batch
    num = np.sum(label_select != 0)
    batch_out = np.zeros([num, 7, 7, Band])
    batch_label = np.zeros([num, kind])
    for i in range(num):
        row_start = pixel[0][i]-3       #row_start = pixel[0][i]  
        row_end = pixel[0][i]+4         #row_end = pixel[0][i]+7
        col_start = pixel[1][i]-3       #col_start = pixel[1][i]
        col_end = pixel[1][i]+4         #col_end = pixel[1][i]+7
        batch_out[i, :, :, :] = paddingdata[row_start:row_end, col_start:col_end, :] 
        temp = (paddinglabel[pixel[0][i],pixel[1][i]]-1)    #temp = (label_selct[pixel[0][i],pixel[1][i]]-1) 
        batch_label[i, temp] = 1
    #修改合适三维卷积输入维度 [depth height weight]
    batch_out = batch_out.swapaxes(1,3)
    batch_out = batch_out[:, :, :, :, np.newaxis]
    return batch_out, batch_label    
     
'''
functions: 获取二维卷积所需要的batches
input： 总体数据和备选则像素标签
output：所选择像素的batches [batch, height, weight, 1]
'''
def get_2dimbatches(input_data, label_select) :
   pixel_num = np.sum(label_select != 0)
   kind = np.unique(label_select).shape[0]-1
   size_2dim = int(np.sqrt(input_data.shape[2]))
   reshape_band = pow(size_2dim, 2)
   batch_out = np.zeros([pixel_num, size_2dim, size_2dim])
   batch_label = np.zeros([pixel_num, kind])
   pixel = np.where(label_select != 0)
   for i in range(pixel_num):
       batch_out[i, :, :] = np.reshape(input_data[pixel[0][i], pixel[1][i], :reshape_band], (size_2dim, size_2dim))
       temp = (label_select[pixel[0][i],pixel[1][i]]-1)
       batch_label[i, temp] = 1
   batch_out = batch_out.swapaxes(1,2)
   batch_out = batch_out[:, :, :, np.newaxis]
   return batch_out, batch_label    


  



























