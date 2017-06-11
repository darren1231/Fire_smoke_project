# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 01:01:21 2017

@author: darren
"""

import tensorflow as tf

class cnn_model(object):
    
    def __init__(self):
        self.alpha=0.333
           
    
    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape,stddev=0.01)
        return tf.Variable(initial)
        
    def bias_variable(self,shape):
        initial = tf.constant(0.0,shape=shape)
        return tf.Variable(initial)
        
    def conv2d(self,x,W,stride):
        return tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding="VALID")
    
    def max_pool_3x3(self,x):
        return tf.nn.max_pool(x,ksize=[1,3,3,1],strides=[1,2,2,1],padding="VALID")
            
    def leaky_relu(self,x):
        x = tf.maximum(self.alpha*x,x)
        return x
    
    def cnn_onemap_model(self):
        
        with tf.name_scope("onemap_model"):
            with tf.name_scope("conv1"):
                w_conv1 = self.weight_variable([3,3,3,16])
                b_conv1 = self.bias_variable([16])   
                tf.summary.histogram('w_conv1', w_conv1)
                tf.summary.histogram('b_conv1', b_conv1)
            
            with tf.name_scope("conv2"):
                w_conv2 = self.weight_variable([3,3,16,16])
                b_conv2 = self.bias_variable([16])
                tf.summary.histogram('w_conv2', w_conv2)
                tf.summary.histogram('b_conv2', b_conv2)
                
            with tf.name_scope("conv3"):
                w_conv3 = self.weight_variable([3,3,16,16])
                b_conv3 = self.bias_variable([16])
                tf.summary.histogram('w_conv3', w_conv3)
                tf.summary.histogram('b_conv3', b_conv3)
                
            with tf.name_scope("conv4"):
                w_conv4 = self.weight_variable([3,3,16,1])
                b_conv4 = self.bias_variable([1])
                tf.summary.histogram('w_conv4', w_conv4)
                tf.summary.histogram('b_conv4', b_conv4)
                
            with tf.name_scope("fc1"):
                w_fc1 = self.weight_variable([144,100])
                b_fc1 = self.bias_variable([100])
                tf.summary.histogram('w_fc1', w_fc1)
                tf.summary.histogram('b_fc1', b_fc1)
                
            with tf.name_scope("fc2"):
                w_fc2 = self.weight_variable([100,100])
                b_fc2 = self.bias_variable([100])
                tf.summary.histogram('w_fc2', w_fc2)
                tf.summary.histogram('b_fc2', b_fc2)
            
            with tf.name_scope("fc3"):
                w_fc3 = self.weight_variable([100,2])
                b_fc3 = self.bias_variable([2])
                tf.summary.histogram('w_fc3', w_fc3)
                tf.summary.histogram('b_fc3', b_fc3)
                
            desired_input = tf.placeholder(tf.float32,[None,64,64,3])
            tf.summary.image('input', desired_input, 3)
            desired_output = tf.placeholder(tf.float32,[None,2])
            keep_prob = tf.placeholder(tf.float32)
            
            with tf.name_scope("layer1"):
                layer1= self.leaky_relu(self.conv2d(desired_input,w_conv1,1)+b_conv1)
            with tf.name_scope("layer2"):
                layer2= self.leaky_relu(self.conv2d(layer1,w_conv2,1)+b_conv2)
            with tf.name_scope("layer3"):  
                layer3= self.max_pool_3x3(layer2)
            with tf.name_scope("layer4"):     
                layer4= self.leaky_relu(self.conv2d(layer3,w_conv3,1)+b_conv3)
            with tf.name_scope("layer5"):  
                layer5= self.leaky_relu(self.conv2d(layer4,w_conv4,1)+b_conv4)
            with tf.name_scope("layer6"):  
                layer6 = self.max_pool_3x3(layer5)
                tf.summary.image('last_feature_map', layer6, 3)
                #print (layer6.get_shape())
            with tf.name_scope("layer6_reshape"):  
                layer6_reshape = tf.reshape(layer6,[-1,144])
            with tf.name_scope("layer7"):  
                layer7= tf.nn.dropout(self.leaky_relu(tf.matmul(layer6_reshape,w_fc1)+b_fc1),keep_prob)
            with tf.name_scope("layer8"):  
                layer8 = tf.nn.dropout(self.leaky_relu(tf.matmul(layer7,w_fc2)+b_fc2),keep_prob)
            with tf.name_scope("layer9"):  
                layer9 = tf.nn.softmax(tf.matmul(layer8,w_fc3)+b_fc3)
        
        return desired_input,desired_output,layer9,keep_prob
    
    def cnn_easy_model(self):
        
        with tf.name_scope("cnn_easy_model"):
            with tf.name_scope("conv1"):
                w_conv1 = self.weight_variable([3,3,3,16])
                b_conv1 = self.bias_variable([16])   
                tf.summary.histogram('w_conv1', w_conv1)
                tf.summary.histogram('b_conv1', b_conv1)
            
            with tf.name_scope("conv2"):
                w_conv2 = self.weight_variable([3,3,16,16])
                b_conv2 = self.bias_variable([16])
                tf.summary.histogram('w_conv2', w_conv2)
                tf.summary.histogram('b_conv2', b_conv2)
                
            with tf.name_scope("conv3"):
                w_conv3 = self.weight_variable([3,3,16,16])
                b_conv3 = self.bias_variable([16])
                tf.summary.histogram('w_conv3', w_conv3)
                tf.summary.histogram('b_conv3', b_conv3)
                
            with tf.name_scope("conv4"):
                w_conv4 = self.weight_variable([3,3,16,1])
                b_conv4 = self.bias_variable([1])
                tf.summary.histogram('w_conv4', w_conv4)
                tf.summary.histogram('b_conv4', b_conv4)
                
            with tf.name_scope("fc1"):
                w_fc1 = self.weight_variable([144,100])
                b_fc1 = self.bias_variable([100])
                tf.summary.histogram('w_fc1', w_fc1)
                tf.summary.histogram('b_fc1', b_fc1)
                
            with tf.name_scope("fc2"):
                w_fc2 = self.weight_variable([100,100])
                b_fc2 = self.bias_variable([100])
                tf.summary.histogram('w_fc2', w_fc2)
                tf.summary.histogram('b_fc2', b_fc2)
            
            with tf.name_scope("fc3"):
                w_fc3 = self.weight_variable([100,2])
                b_fc3 = self.bias_variable([2])
                tf.summary.histogram('w_fc3', w_fc3)
                tf.summary.histogram('b_fc3', b_fc3)
                
            desired_input = tf.placeholder(tf.float32,[None,64,64,3])
            tf.summary.image('input', desired_input, 3)
            desired_output = tf.placeholder(tf.float32,[None,2])
            keep_prob = tf.placeholder(tf.float32)
            
            with tf.name_scope("layer1"):
                layer1= tf.nn.relu(self.conv2d(desired_input,w_conv1,1)+b_conv1)
            with tf.name_scope("layer2"):
                layer2= tf.nn.relu(self.conv2d(layer1,w_conv2,1)+b_conv2)
            with tf.name_scope("layer3"):  
                layer3= self.max_pool_3x3(layer2)
            with tf.name_scope("layer4"):     
                layer4= tf.nn.relu(self.conv2d(layer3,w_conv3,1)+b_conv3)
            with tf.name_scope("layer5"):  
                layer5= tf.nn.relu(self.conv2d(layer4,w_conv4,1)+b_conv4)
            with tf.name_scope("layer6"):  
                layer6 = self.max_pool_3x3(layer5)
                tf.summary.image('last_feature_map', layer6, 3)
                #print (layer6.get_shape())
            with tf.name_scope("layer6_reshape"):  
                layer6_reshape = tf.reshape(layer6,[-1,144])
            with tf.name_scope("layer7"):  
                layer7= tf.nn.dropout(tf.nn.relu(tf.matmul(layer6_reshape,w_fc1)+b_fc1),keep_prob)
            with tf.name_scope("layer8"):  
                layer8 = tf.nn.dropout(tf.nn.relu(tf.matmul(layer7,w_fc2)+b_fc2),keep_prob)
            with tf.name_scope("layer9"):  
                layer9 = tf.nn.softmax(tf.matmul(layer8,w_fc3)+b_fc3)
                
        return desired_input,desired_output,layer9,keep_prob