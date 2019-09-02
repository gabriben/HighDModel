import numpy as np
import pandas as pd
import os
import pickle


X_test = np.loadtxt('data/X_test.dat')
y_test = np.loadtxt('data/y_test.dat')

X_train = np.loadtxt('data/X_train.dat')
y_train = np.loadtxt('data/y_train.dat')


select = 1000

indices = np.loadtxt('selected/1000/X_hsiclasso.dat')
indices = indices.astype(int)


d = X_train.shape[0] + X_test.shape[0]
import random
i = xrange(X_train.shape[0])
random.seed(1)
iTrain = random.sample(i, d * 7/10)
iVal = list(set(i) - set(iTrain))
# iTrain = iTrain.astype(int)
# iVal = iVal.astype(int)

X = X_train[:, indices]
X = X[iTrain, :]
Xv = X_train[:, indices]
Xv = Xv[iVal, :]
Xt = X_test[:, indices]
yT = y_train[iTrain]
yv = y_train[iVal]

def standardize(x):
    x-=np.mean(x)
    if np.std(x) > 0:
        x/=np.std(x)
    return x

X = np.apply_along_axis(standardize, 0, X)
Xv = np.apply_along_axis(standardize, 0, Xv)
Xt = np.apply_along_axis(standardize, 0, Xt)

# def mask(x):
#     s = x.size
#     random.seed(1)
#     i = random.sample(xrange(s), int(0.5 * s))
#     i = np.in1d(xrange(s), i)
#     i = np.array(i).reshape(x.shape)
#     m = x
#     m[i] = -10
#     return m


from HighDModel import swap
# import pandas as pd
# import random
X_swapped = swap(pd.DataFrame(X),1, 0.8)
Xv_swapped = swap(pd.DataFrame(Xv),1, 0.8)
Xt_swapped = swap(pd.DataFrame(Xt),1, 0.8)

## NN

# mod = swapped

# def autoencoder(dims = [512, 256, 128, 64, 32]):

#     clean = tf.placeholder(tf.float32,
#                         shape=(None, X.shape[1]), name='clean')
#     swapped = tf.placeholder(tf.float32,
#                         shape=(None, X.shape[1]), name='swapped')
#     y = tf.placeholder(tf.float32, shape=(None, 1), name='y')
    
#     # encoder

#     for i in dims:
#         n_inp = int(mod.get_shape()[1])
#         w = tf.get_variable(shape = [n_out, n_inp],
#                       trainable = True)
#         b = tf.get_variable(shape= [n_out, 1], trainable = True)

#         ws.append(w)
#         bs.append(b)

#         mod = tf.nn.dropout(tf.tanh(tf.add(
#                 tf.matmul(w, tf.transpose(mod)), b)), rate=0.1)

#     z = mod
    
#     ws.reverse()
#     bs.reverse()

#     # decoder
    
#     for i, n_out in enumerate(dims[::-1]):

#         w = tf.transpose(ws[i]) # tied weights
#         b = tf.get_variable(shape= [n_out, 1], trainable = True)

#         if i == len(dims):
#             # output layers
#             n_out = int(clean.get_shape()[1])
#             ws = tf.get_variable(shape = [n_out, hid4_size], name='ws', trainable = True)
#             bs = tf.get_variable(shape = [n_out, 1], name='bs', trainable = True)
#             Xs = tf.transpose(tf.add(tf.matmul(w, y4), b))

#         else:
#             mod = tf.nn.dropout(tf.tanh(tf.add(
#                 tf.matmul(w, tf.transpose(mod)), b)), rate=0.1)

#     y = mod
    
#     loss = tf.reduce_mean(tf.square(y - x))
    

        




import tensorflow as tf

tf.reset_default_graph()

clean = tf.placeholder(tf.float32,
                        shape=(None, X.shape[1]), name='clean')
swapped = tf.placeholder(tf.float32,
                        shape=(None, X.shape[1]), name='swapped')
y = tf.placeholder(tf.float32, shape=(None, 1), name='y')
# pred = tf.placeholder(tf.bool, shape = (), name = 'pred')

# parameters

## First layer
hid1_size = 25
# input 
w1 = tf.get_variable(shape = [hid1_size, X.shape[1]], name='w1',
                      trainable = True)
b1 = tf.get_variable(shape= [hid1_size, 1], name='b1', trainable = True)

## Second layer
hid2_size = 50
# Network 
w2 = tf.get_variable(shape = [hid2_size, hid1_size], name='w2',
                      trainable = True)
b2 = tf.get_variable(shape = [hid2_size, 1], name='b2', trainable = True)

## Middle layer
hidMid_size = 100
# Network 
wm = tf.get_variable(shape = [hidMid_size, hid2_size], name='wm',
                      trainable = True)
bm = tf.get_variable(shape = [hidMid_size, 1], name='bm', trainable = True)

# ## Third layer
hid3_size = 50
# Network 
w3 = tf.get_variable(shape = [hid3_size, hidMid_size], name='w3',
                      trainable = True)
b3 = tf.get_variable(shape = [hid3_size, 1], name='b3', trainable = True)

# ## Fourth layer
hid4_size = 25
# Network 
w4 = tf.get_variable(shape = [hid4_size, hid3_size], name='w4',
                      trainable = True)
b4 = tf.get_variable(shape = [hid4_size, 1], name='b4', trainable = True)

# pretraining

## First layer
y1 = tf.nn.dropout(tf.tanh(tf.add(
    tf.matmul(w1, tf.transpose(swapped)), b1)), rate=0.1)

## Second layer
y2 = tf.nn.dropout(tf.tanh(tf.add(
                    tf.matmul(w2, y1), b2)),
                    rate=0.1)

## Middle layer
ym = tf.add(tf.matmul(wm, y2), bm)

# ## Third layer
y3 = tf.nn.dropout(tf.tanh(tf.add(
                    tf.matmul(w3, ym), b3)),
                    rate=0.1)

# ## Fourth layer
y4 = tf.nn.dropout(tf.tanh(tf.add(
                    tf.matmul(w4, y3), b4)),
                    rate=0.1)

# finetuning

## First layer
y1c = tf.nn.dropout(tf.tanh(tf.add(
    tf.matmul(w1, tf.transpose(clean)), b1)), rate=0.1)

## Second layer
y2c = tf.nn.dropout(tf.tanh(tf.add(
                    tf.matmul(w2, y1c), b2)),
                    rate=0.1)

## Middle layer
ym = tf.add(tf.matmul(wm, y2), bm)


# # ## Third layer
# y3c = tf.nn.dropout(tf.tanh(tf.add(
#                     tf.matmul(w3, y2c), b3)),
#                     rate=0.1)

# # ## Fourth layer
# y4c = tf.nn.dropout(tf.tanh(tf.add(
#                     tf.matmul(w4, y3c), b4)),
#                     rate=0.1)





# # ## Fifth layer
# hid5_size = 100
# # Network 
# w5 = tf.get_variable(shape = [hid5_size, hid4_size], name='w5',
#                       trainable = True)
# b5 = tf.get_variable(shape = [hid5_size, 1], name='b5', trainable = True)
# y5 = tf.nn.dropout(tf.tanh(tf.add(
#                     tf.matmul(w5, y4), b5)),
#                     rate=0.1)

# Output layer
ws = tf.get_variable(shape = [X.shape[1], hid4_size], name='ws', trainable = True)
bs = tf.get_variable(shape = [X.shape[1], 1], name='bs', trainable = True)
Xs = tf.transpose(tf.add(tf.matmul(ws, y4), bs))

out_size = 1
wo = tf.get_variable(shape = [out_size, hidMid_size], name='wo', trainable = True)
bo = tf.get_variable(shape = [out_size, 1], name='bo', trainable = True)
yo = tf.transpose(tf.add(tf.matmul(wo, ym), bo))

l = 0.0001

loss_self = tf.reduce_mean(tf.square(clean - Xs))
loss_pred = tf.reduce_mean(tf.square(y-yo))

loss_total = l * loss_self + loss_pred
    
#tf.reduce_sum(S)
optimizerSelf = tf.train.AdamOptimizer(0.001).minimize(loss_self)
optimizerTotal = tf.train.AdamOptimizer(0.001).minimize(loss_total)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

def runModel(X_clean, Xv_clean, X_swapped, Xv_swapped,
             selfLearning = True, number_epochs = 1000):
    b = 25 # batch size 
    nEpochs = number_epochs

    if selfLearning:
        optimizer = optimizerSelf
        loss = loss_self
    else :
        optimizer = optimizerTotal
        loss = loss_total
    
    rmseTrain = np.zeros(nEpochs) 
    rmseTest = np.zeros(nEpochs)

    saver = tf.train.Saver()

    stopEarly = False
    best_loss = 10**10
    a = 100 # generalization loss threshold
    # training   0.0001
    for learning_rate in [0.0001]:
        for epoch in range(nEpochs):
            avg_cost = 0.0

            # For each epoch, we go through all the samples we have.
            for i in xrange(0, X.shape[0] , b):

                _, c = sess.run([optimizer, loss],
                                feed_dict={clean: X[i : (i + b), :],
                                           swapped: X_swapped[i : (i + b), :],
                                           y: np.expand_dims(yT[i : (i + b)],
                                                             axis = -1)
                                })
                avg_cost += c
            avg_cost /= X.shape[0]    

            # Print the cost in this epcho to the console.
            if epoch % 10 == 0:
                print("Epoch: {:3d}    Train Cost: {:.4f}".format(epoch, avg_cost))

            loss_train = sess.run(loss,
                            feed_dict={clean: X,
                                       swapped: X_swapped,
                                       y: np.expand_dims(yT, axis = -1)
                            })
            rmseTrain[epoch] = np.sqrt(loss_train)

            loss_test = sess.run(loss,
                            feed_dict={clean: Xv,
                                       swapped: Xv_swapped,
                                       y: np.expand_dims(yv, axis = -1)
                            })
            rmseTest[epoch] = np.sqrt(loss_test)

            # implementing https://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf
            GL = 100 * (loss_test/best_loss - 1)
            if (loss_test < best_loss):
                # stopping_step = 0
                saver.save(sess, 'intModels_swapped/siamesesplitLASSO - epoch = ' +
                           str(epoch) + '.ckpt')
                best_loss = loss_test
            else:
                # stopping_step += 1
                # if stopping_step >= patience:
                if(GL > a):
                    print("Early stopping is trigger at epoch: {}" \
                          .format(epoch,loss_test))
                    stopEarly = True
                    break

            if stopEarly == True:
                break

        if stopEarly == True:
            break

    return rmseTest, rmseTrain


rmseTest, rmseTrain = runModel(X, Xv, X_swapped, Xv_swapped,
                               selfLearning = True, number_epochs = 1000)

rmseTest, rmseTrain = runModel(X, Xv, X_swapped, Xv_swapped,
             selfLearning = False, number_epochs = 1000)


best = np.where(rmseTest == np.min(rmseTest[rmseTest <> 0]))
best = best[0][0]

saver = tf.train.Saver()
saver.restore(sess, 'intModels_swapped/siamesesplitLASSO - epoch = ' +
                       str(best) + '.ckpt')

from bokeh.plotting import figure, output_file, show
output_file("test.html")
p = figure()
p.line(xrange(1, len(rmseTest)), rmseTest, line_color = 'olive')
p.line(xrange(1, len(rmseTrain)), rmseTrain)
show(p)





loss_test = sess.run(loss_total,
                feed_dict={clean: Xv,
                           swapped: Xv_swapped,
                           y: np.expand_dims(yv, axis = -1)
                })
np.sqrt(loss_test)

loss_testSet = sess.run(loss_total,
                feed_dict={clean: Xt,
                           swapped: Xt_swapped,
                           y: np.expand_dims(y_test, axis = -1)
                })
np.sqrt(loss_testSet)

# np.savetxt('prediction accuracy/X_hsiclasso_SSMasked-DAE_1000.dat',
#            np.array([np.sqrt(loss_testSet)]))
