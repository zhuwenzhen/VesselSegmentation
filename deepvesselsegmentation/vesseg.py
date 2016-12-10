import scipy.ndimage
import PIL
import tensorflow as tf
import pickle
import random
import logging
import numpy as np
import itertools
import time
import multiprocessing as mp
import threading as th
import collections
np.set_printoptions(threshold=np.nan)
logging.basicConfig(format="%(levelname)s:%(asctime)s:%(module)s:%(funcName)s:%(lineno)d:%(message)s")
#sess =  tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess =  tf.Session()
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
#def conv2d2(x, W):
#    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

from tensorflow.contrib.layers import batch_norm

x = tf.placeholder(tf.float32, shape=[None, 4225])
y_ = tf.placeholder(tf.float32, shape=[None, 2])
train_phase = tf.placeholder(tf.bool, name='phase_train')

def batch_norm_layer(x,train_phase, scope_bn):
    bn_train = batch_norm(x, decay=0.999, center=True, scale=True,
    updates_collections=None,
    reuse=None, # is this right?
    is_training=True,
    trainable=True,
    scope=scope_bn)
    bn_inference = batch_norm(x, decay=0.999, center=True, scale=True,
    updates_collections=None,
    reuse=True, # is this right?
    is_training=False,
    trainable=True,
    scope=scope_bn)
    z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
    return z

#first layer
x_image = tf.reshape(x, [-1,65,65,1])
x_image_bn = batch_norm_layer(x_image, train_phase, "bn_input")
W_conv1 = weight_variable([6, 6, 1, 48], name="w_conv1")
b_conv1 = bias_variable([48], name="b_conv1")
h_conv1 = tf.nn.elu(conv2d(x_image_bn, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

h_pool1_bn = batch_norm_layer(h_pool1, train_phase, "bn_pool1")

#first layer
W_conv2 = weight_variable([5, 5, 48, 48], name="w_conv2")
b_conv2 = bias_variable([48], name="b_conv2")
h_conv2 = tf.nn.elu(conv2d(h_pool1_bn, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

h_pool2_bn = batch_norm_layer(h_pool2, train_phase, "bn_pool2")

#third layer
W_conv3 = weight_variable([4, 4, 48, 48], name="w_conv3")
b_conv3 = bias_variable([48], name="b_conv3")
h_conv3 = tf.nn.elu(conv2d(h_pool2_bn, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

h_pool3_bn = batch_norm_layer(h_pool3, train_phase, "bn_pool3")

#fourth layer
W_conv4 = weight_variable([2, 2, 48, 48], name="w_conv4")
b_conv4 = bias_variable([48], name="b_conv4")
h_conv4 = tf.nn.elu(conv2d(h_pool3_bn, W_conv4) + b_conv4)
h_pool4 = max_pool_2x2(h_conv4)
h_pool4_bn = batch_norm_layer(h_pool4, train_phase, "bn_pool4")


#fc1
W_fc1 = weight_variable([2 * 2 * 48, 100], name="fc1")
b_fc1 = bias_variable([100], name="b_fc1")

h_pool4_flat = tf.reshape(h_pool4_bn, [-1, 2*2*48])
h_fc1 = tf.nn.elu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)

h_fc1_bn = batch_norm_layer(h_fc1, train_phase, "bn_fc1")


keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1_bn, keep_prob)

#fc2
W_fc2 = weight_variable([100, 2], name="fc2")
b_fc2 = bias_variable([2], name="b_fc2")

h_fc2 = tf.nn.elu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


y_conv = h_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
for var in tf.trainable_variables():
    tf.histogram_summary(var.name, var)
#opt = tf.train.AdamOptimizer(1e-10)
opt = tf.train.GradientDescentOptimizer(1e-10)
gradients = opt.compute_gradients(cross_entropy)
for grad, var in gradients:
    if var == None:
        continue
    if var.name == None:
        continue
    if grad == None:
        continue
    tf.histogram_summary(var.name + '/gradient', grad)
train_step = opt.apply_gradients(gradients)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar("accuracy", accuracy)
train_writer = tf.train.SummaryWriter('/mnt/logs/train', sess.graph)
merged = tf.summary.merge_all()
#sess.run(tf.global_variables_initializer())
queue = mp.Queue(1100)
soft_max = tf.nn.softmax(y_conv)

def simple_batch():
    f = open("data_positive.pkl", "rb")
    f2 = open("data_negative.pkl", "rb")
    depleted = False
    while True:
        pos_features = np.ndarray((0, 4225), dtype=int)
        pos_labels = np.ndarray((0, 2), dtype=int)
        try:
            for i in range(20):
                pos_f, pos_l = pickle.load(f)
                pos_features = np.append(pos_features, pos_f, axis=0)
                pos_labels = np.append(pos_labels, pos_l, axis=0)
        except Exception as e:
            depleted = True
            break
        neg_features = np.ndarray((0, 4225), dtype=int)
        neg_labels = np.ndarray((0, 2), dtype=int)
        try:
            for i in range(20):
                neg_f, neg_l = pickle.load(f2)
                neg_features = np.append(neg_features, neg_f, axis=0)
                neg_labels = np.append(neg_labels, neg_l, axis=0)
        except Exception as e:
            depleted = True
            break
        if depleted:
            break
        pos=list(zip(pos_features, pos_labels))
        neg=list(zip(neg_features, neg_labels))
        samples = pos + neg
        np.random.shuffle(samples)
        #np.random.shuffle(neg)
        #while True:
        for i in range(0, len(samples)-100, 100):
            #yield samples[0:10]
            yield samples[i: i+100]
            #yield neg[i: i+1000]
    f.close()
    f2.close()

def queued_batch():
    while True:
        it = simple_batch()
        while True:
            try:
                batch = next(it)
                queue.put(batch)
            except:
                queue.put(Exception)
                break
def simple_batch_test():
    f = open("data_positive_test.pkl", "rb")
    f2 = open("data_negative_test.pkl", "rb")
    depleted = False
    while True:
        pos = []
        try:
            pos = pickle.load(f)
        except Exception as e:
            depleted = True
            break
        neg=[]
        try:
            neg = pickle.load(f2)
        except Exception as e:
            depleted = True
            break
        if depleted:
            break
        pos=list(zip(pos[0], pos[1]))
        neg=list(zip(neg[0], neg[1]))
        random.shuffle(pos)
        random.shuffle(neg)

        for i in range(0, len(pos)-100, 100):
            temp =  pos[i: i+100]
            temp.extend(neg[i: i+100])
            random.shuffle(temp)
            yield temp
    f.close()
    f2.close()

#batches = collections.deque()
#semaphore = th.semaphore(value=0)
def get_batches():
    while True:
        batch = queue.get()
        batches.append(batch)
        if batch == Exception:
            semaphore.acquire()

#p = mp.Process(target=queued_batch)
#p.start()

#it2 = itertools.cycle(simple_batch_test())
#with sess.as_default():
   # saver = tf.train.Saver()
 #   saver.restore(sess, "./model3.ckpt") #TODO remove restore
  #  counter = 0
    #start_time = time.time() TODO
 #   start_time = time.time()
    #epoch = 0
   # while True:
  #      print("epoch: ", epoch)
        ##it = simple_batch()
       # while True:
      #      try:
     #           #batch = batches.popleft()
    #            batch = queue.get()
   #             if batch == Exception:
  #                  raise Exception()
           # except:
          #      save_path = saver.save(sess, "./model3.ckpt")
         #       print("end epoch")
        #        print("Model saved in file: %s" % save_path)
       #         break
      #      if counter%100 == 0:
     #           endtime = time.time()
    #            print("speed: ", 1000*10/(endtime - start_time))
   #             start_time = time.time()
  #          if counter%1000 == 0:
            #    train_accuracy = accuracy.eval(feed_dict={
           #     x:[e[0] for e in batch], y_: [e[1] for e in batch] , keep_prob: 1.0, train_phase:False})
          #      print("step %d, training accuracy %g"%(counter, train_accuracy))
         #       test_batch = next(it2)
        #        testing_accuracy = accuracy.eval(feed_dict={
       #         x:[e[0] for e in test_batch], y_: [e[1] for e in test_batch], keep_prob: 1.0, train_phase:False})
    #            print("step %d, testing accuracy %g"%(counter, testing_accuracy))
     #           save_path = saver.save(sess, "./model3.ckpt")
      #          print("Model saved in file: %s" % save_path)
                #endtime = time.time() TODO
                #print("speed: ", 1000*100/(endtime - start_time)) TODO
                #start_time = time.time() TODO

   #         counter +=1
  #          summary, _ = sess.run(fetches=[merged, train_step], feed_dict={x:[e[0] for e in batch], y_: [e[1] for e in batch], keep_prob: 0.7, train_phase: True})
 #           train_writer.add_summary(summary, counter)
#        epoch+=1
with sess.as_default():
    saver = tf.train.Saver()
    saver.restore(sess, "./model3.ckpt") #TODO remove restore
    counter = 0
    #start_time = time.time() TODO
    start_time = time.time()
    epoch = 0
    img = scipy.ndimage.imread("test.tif", mode="RGB")
    img = img[:,:,1]
    seg = PIL.Image.new("L",img.shape)
    for y1 in range(32, img.shape[0]-33):
        for x1 in range(32, img.shape[1]-33):
            window = img[y1-32:y1+33,x1-32:x1+33]
#            print(window.shape)
            probs = soft_max.eval(feed_dict={x:window.flatten().reshape(1, 4225), train_phase:False, keep_prob:1.0})
#            print(probs)
#            print(probs[0,0])
            if y1 % 10 == 0:
                print((x1,y1))
            seg.putpixel((x1,y1), (int(round(probs[0,0]*255)),))
    seg.save("pred.jpg")
