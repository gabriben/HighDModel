import numpy as np
import pandas as pd
import os
import pickle


X_test = np.loadtxt('data/X_test.dat')
y_test = np.loadtxt('data/y_test.dat')

X_train = np.loadtxt('data/X_train.dat')
y_train = np.loadtxt('data/y_train.dat')


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as pyplot
pyplot.figure()

from scipy.stats import gaussian_kde
density = gaussian_kde(X_norm[:,1])
xs = np.linspace(0,8,200)
density.covariance_factor = lambda : .25
density._compute_covariance()
pyplot.plot(xs,density(xs))
pyplot.savefig('exploratory/X_norm.png')

pyplot.figure()
pyplot.hist(X_norm[:,100])
pyplot.savefig('exploratory/X_norm.png')

pyplot.figure()
pyplot.hist(X[:,100])
pyplot.savefig('exploratory/X.png')


#batch size
# b = 376

select = 500

# from pyHSICLasso import HSICLasso
# hsic_lasso = HSICLasso()
# hsic_lasso.input(X_train,y_train)
# hsic_lasso.regression(1000)

# related = np.zeros(999)

# for i in range(999):
#     related[i] = hsic_lasso.get_index_neighbors(feat_index = i,
#                                                 num_neighbors = 1)[0]

# indices = hsic_lasso.get_index()

allRelated = np.loadtxt('selectionModel/hsiclasso_500.sav')

indices = allRelated[:,1]
related = allRelated[:,2]
related = np.unique(related)
related = related[np.invert(np.isin(related, indices))]

related = related.astype(int)
indices = indices.astype(int)

rel = X_train[:, related]
rel_test = X_test[:, related]
X = X_train[:, indices]
Xt = X_test[:, indices]


from sklearn import preprocessing
# Normalize Training Data 
std_scale = preprocessing.StandardScaler().fit(X)
X_norm = std_scale.transform(X)
#Converting numpy array to dataframe

# Normalize Testing Data by using mean and SD of training set
x_test_norm = std_scale.transform(test_norm)
testing_norm_col = pd.DataFrame(x_test_norm, index=test_norm.index,
                                columns=test_norm.columns) 
x_test.update(testing_norm_col)
print (x_train.head())

## NN

import tensorflow as tf

tf.reset_default_graph()

inputA = tf.placeholder(tf.float32,
                        shape=(None, X.shape[1]), name='inputA')
inputB = tf.placeholder(tf.float32,
                        shape=(None, rel.shape[1]), name='inputB')
# I = tf.placeholder(tf.float32,
#                         shape=(None, None), name='I')
y = tf.placeholder(tf.float32, shape=(None, 1), name='y')

## First layer
hid1_size = 100
# input A
w1A = tf.get_variable(shape = [hid1_size, X.shape[1]], name='w1A',
                      trainable = True)
b1A = tf.get_variable(shape= [hid1_size, 1], name='b1A', trainable = True)
y1A = tf.nn.dropout(tf.nn.relu(tf.add(
    tf.matmul(w1A, tf.transpose(inputA)), b1A)), rate=0.2)
# input B
w1B = tf.get_variable(shape = [hid1_size, rel.shape[1]], name='w1B',
                      trainable = True)
b1B = tf.get_variable(shape= [hid1_size, 1], name='b1B', trainable = True)
y1B = tf.nn.dropout(tf.nn.relu(tf.add(
    tf.matmul(w1B, tf.transpose(inputB)), b1B)), rate=0.2)

## Second layer
hid2_size = 25
# Network A
w2A = tf.get_variable(shape = [hid2_size, hid1_size], name='w2A',
                      trainable = True)
b2A = tf.get_variable(shape = [hid2_size, 1], name='b2A', trainable = True)
y2A = tf.nn.dropout(tf.nn.relu(tf.add(
                    tf.matmul(w2A, y1A), b2A)),
                    rate=0.2)
# Network B
w2B = tf.get_variable(shape = [hid2_size, hid1_size], name='w2B',
                      trainable = True)
b2B = tf.get_variable(shape = [hid2_size, 1], name='b2B', trainable = True)
y2B = tf.nn.dropout(tf.nn.relu(tf.add(
                    tf.matmul(w2B, y1B), b2B)),
                    rate=0.2)
## Third layer
hid3_size = 25
# Network A
w3A = tf.get_variable(shape = [hid3_size, hid2_size], name='w3A',
                      trainable = True)
b3A = tf.get_variable(shape = [hid3_size, 1], name='b3A', trainable = True)
y3A = tf.nn.dropout(tf.nn.relu(tf.add(
                    tf.matmul(w3A, y2A), b3A)),
                    rate=0.2)

## Third layer
hid4_size = 25
# Network A
w4A = tf.get_variable(shape = [hid4_size, hid3_size], name='w4A',
                      trainable = True)
b4A = tf.get_variable(shape = [hid4_size, 1], name='b4A', trainable = True)
y4A = tf.nn.dropout(tf.nn.relu(tf.add(
                    tf.matmul(w4A, y3A), b4A)),
                    rate=0.2)

# similarity
a = tf.matmul(tf.transpose(y3A),y2B) # n x n
# c = tf.reduce_sum(a,axis = 1)
# I = tf.diag(tf.ones_like(c))
#p = c * a

D = - tf.reduce_mean(a) + tf.reduce_sum(tf.math.reduce_logsumexp(a,axis=1))

# I = tf.eye(b)
# S = -tf.reduce_sum( I * tf.log(p + 1e-10))
# S = -tf.reduce_sum(I*tf.log(tf.clip_by_value(p,1e-10,1.0)))
# S = tf.losses.softmax_cross_entropy(I, p)
# S = - tf.multiply(tf.log(p), I)

# Output layer
wo = tf.get_variable(shape = [hid2_size, hid2_size], name='wo', trainable = True)
bo = tf.get_variable(shape = [hid2_size, 1], name='bo', trainable = True)
yo = tf.add(tf.matmul(wo, y2A), tf.add(tf.matmul(wo, y2B), bo))

l = 1

# loss = tf.reduce_mean(tf.square(y-yo))
loss = tf.reduce_mean(tf.square(y-yo)) + l * D
#tf.reduce_sum(S)
optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


b = 25 # batch size 
nEpochs = 1000

rmseTrain = np.zeros(nEpochs) 
rmseTest = np.zeros(nEpochs)

# training
for learning_rate in [0.0001]:
    for epoch in range(nEpochs):
        avg_cost = 0.0

        # For each epoch, we go through all the samples we have.
        for i in xrange(0,X.shape[0] , b):
            # Finally, this is where the magic happens: run our optimizer,
            # feed the current example into X and the current target into Y
            # n = X.shape[0]
            # if n - i < b:
            #     A = np.zeros((n-i, i))
            #     B = np.eye(n-i)
            #     K = np.column_stack([A,B])
            # else:
            #     A = np.zeros((b, i))
            #     B = np.eye(b)
            #     C = np.zeros((b, n - i - b))
            #     K = np.column_stack([A,B,C])
            
            _, c = sess.run([optimizer, loss],
                            feed_dict={inputA: X[i : (i + b), :],
                                       inputB: rel[i : (i + b), :],
                                       # I: K,
                                       y: np.expand_dims(y_train[i : (i + b)],
                                                          axis = -1)
                            })
            avg_cost += c
        avg_cost /= X.shape[0]    

        # Print the cost in this epcho to the console.
        if epoch % 10 == 0:
            print("Epoch: {:3d}    Train Cost: {:.4f}".format(epoch, avg_cost))

        yTrainPred = sess.run(yo,
                        feed_dict={inputA: X,
                                   inputB: rel,
                                   y: np.expand_dims(y_train, axis= -1)
                        })
        rmseTrain[epoch] = np.sqrt(np.mean(np.square(yTrainPred - y_train)))
        # (sum(yTrainPred - y_train)**2 /
        #           y_train.size)**(.5)
        
        yTestPred = sess.run(yo,
                        feed_dict={inputA: Xt,
                                  inputB: rel_test,
                                   y: np.expand_dims(y_test, axis = -1)
                            })
        rmseTest[epoch] = np.sqrt(np.mean(np.square(yTestPred - y_test)))
        # rmseTest[epoch] = (sum((np.concatenate(yTestPred) - y_test)**2) /
        #           y_test1.size)**(.5)

writer = tf.summary.FileWriter('architecture', sess.graph)

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as pyplot

pyplot.figure()
pyplot.plot(rmseTrain, label='train')
pyplot.plot(rmseTest, label='test')
pyplot.legend()
pyplot.legend(loc='upper right')
pyplot.savefig('history/siameseRel - ' + 'TwoMoreLayer - l = 1' + '.png')


# pyplot.show()
# pyplot.title(fs + '_' + method + '_' + select)
pyplot.ylabel('Loss')
pyplot.xlabel('Epoch')


np.savetxt('prediction accuracy/hsiclasso_SSML_500.dat', np.array([rmseTest[-1]]))
