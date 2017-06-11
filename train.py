# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 10:33:06 2017

@author: darren
"""

import tensorflow as tf
import numpy as np
import random
import process_data
import os
import CNN_model

os_location=os.getcwd()
LOGDIR= os_location+'/tensorboard_data/'

def loss_function(loss_mode,desired_output,net_output):
    
    if loss_mode==0:
        loss_name="softmax_error"
        return (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=desired_output, logits=net_output)),loss_name)
    else:
        loss_name='square_error'
        return (tf.reduce_mean(tf.reduce_sum(tf.square(desired_output - net_output),reduction_indices=1)),loss_name)
        
def optimizer(optimizer_mode,learning_rate):
    
    if optimizer_mode==0:
        optimizer_name = "gradient"
        return (tf.train.GradientDescentOptimizer(learning_rate),optimizer_name)
    elif optimizer_mode==1:
        optimizer_name = "Momentum"
        return (tf.train.MomentumOptimizer(learning_rate,0.9),optimizer_name)
    else:
        optimizer_name = "Adam"
        return (tf.train.AdamOptimizer(learning_rate),optimizer_name)

def train_net(initial_learning_rate,loss_mode,optimizer_mode,data):
    
    
    tf.reset_default_graph()
    train_data,test_desired_input,test_desired_output=data    
    
    
    desired_input,desired_output,layer9,keep_prob=CNN_model.cnn_model().cnn_onemap_model()
    model_name ="cnn_onemap_model"
    
        
    global_ = tf.Variable(tf.constant(0))  
    decay_learning_rate = tf.train.exponential_decay(initial_learning_rate, global_, 5, 0.95, staircase=True)  
        
    loss,loss_name = loss_function(loss_mode,desired_output,layer9)
    tf.summary.scalar("loss",loss)
    
    train_step = optimizer(optimizer_mode,decay_learning_rate)[0].minimize(loss)
    optimizer_name=optimizer(optimizer_mode,decay_learning_rate)[1]
    
    correct_prediction = tf.equal(tf.arg_max(desired_output,1),tf.arg_max(layer9,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    tf.summary.scalar("accuracy",accuracy)
    
    #tensorboard merged all
    merged= tf.summary.merge_all()
    
    sess=tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    
    writer = tf.summary.FileWriter(LOGDIR+model_name+'_'+str(initial_learning_rate)+'_'+loss_name+'_'+optimizer_name+"/",sess.graph)
    writer.add_graph(sess.graph)
    
    data_file = open( LOGDIR  + '/'+str(initial_learning_rate)+'_'+loss_name+'_'+optimizer_name+".txt", 'w')
    
    for i in range(2):
        
        minibatch = random.sample(train_data,100)
        image_batch = [d[0] for d in minibatch]        
        label_batch = [d[1] for d in minibatch]
        
        sess.run(train_step,feed_dict={desired_input:image_batch,desired_output:label_batch,keep_prob: 0.5})
        sess.run(decay_learning_rate,feed_dict={global_:i})
        
        training_loss=sess.run(loss,feed_dict={desired_input:image_batch,desired_output:label_batch,keep_prob: 1})
        test_loss=sess.run(loss,feed_dict={desired_input:test_desired_input,desired_output:test_desired_output,keep_prob: 1.0})        
               
        accuracy_num = sess.run(accuracy,feed_dict={desired_input:test_desired_input,desired_output:test_desired_output,keep_prob: 1.0})
        #accuracy_num = sess.run(accuracy,feed_dict={desired_input:image_batch,take_desired_output:label_batch,keep_prob: 1.0})
        print ("step:",i,"accuracy:",accuracy_num,"training_loss",training_loss,"test_loss",test_loss)
        data_file.write(str(i)+','+str(accuracy_num)+','+str(training_loss)+','+str(test_loss)+'\n')
        summary_data=sess.run(merged,feed_dict={desired_input:image_batch,desired_output:label_batch,keep_prob: 1})
        writer.add_summary(summary_data,i)
        
def main():
    train_data,test_data= process_data.Process_data().creat_train_test_data()
    test_desired_input=[d[0] for d in test_data]
    test_desired_output=[d[1] for d in test_data]
    data=tuple([train_data,test_desired_input,test_desired_output])    
     
    iterate = 0
    for initial_learning_rate in [0.1,0.01,0.001,0.0001]:
        for loss_mode in [0,1]:
            for optimizer_mode in [0,1,2]:
                iterate+=iterate
                print (iterate,":initial_learning_rate",initial_learning_rate,"loss_mode",loss_mode,"optimizer_mode",optimizer_mode)
                train_net(initial_learning_rate,loss_mode,optimizer_mode,data)
    
    
            
    
if __name__ == '__main__':
    main()