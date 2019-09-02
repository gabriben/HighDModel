def siamese(X1, X2, X1t, X2t, y, l = 0.0001,
            epochs = 1000, batchSize = 25):

    import tensorflow as tf

    tf.reset_default_graph()

    inputA = tf.placeholder(tf.float32,
                            shape=(None, X1.shape[1]), name='inputA')
    inputB = tf.placeholder(tf.float32,
                            shape=(None, X2.shape[1]), name='inputB')
    # I = tf.placeholder(tf.float32,
    #                         shape=(None, None), name='I')
    y = tf.placeholder(tf.float32, shape=(None, 1), name='y')


    ## First layer
    hid1_size = 1500
    # input A
    w1A = tf.get_variable(shape = [hid1_size, X1.shape[1]], name='w1A',
                          trainable = True)
    b1A = tf.get_variable(shape= [hid1_size, 1], name='b1A', trainable = True)
    y1A = tf.nn.dropout(tf.nn.relu(tf.add(
        tf.matmul(w1A, tf.transpose(inputA)), b1A)), rate=0.1)
    # input B
    w1B = tf.get_variable(shape = [hid1_size, X2.shape[1]], name='w1B',
                          trainable = True)
    b1B = tf.get_variable(shape= [hid1_size, 1], name='b1B', trainable = True)
    y1B = tf.nn.dropout(tf.nn.relu(tf.add(
        tf.matmul(w1B, tf.transpose(inputB)), b1B)), rate=0.1)

    ## Second layer
    hid2_size = 100
    # Network A
    w2A = tf.get_variable(shape = [hid2_size, hid1_size], name='w2A',
                          trainable = True)
    b2A = tf.get_variable(shape = [hid2_size, 1], name='b2A', trainable = True)
    y2A = tf.nn.dropout(tf.nn.relu(tf.add(
                        tf.matmul(w2A, y1A), b2A)),
                        rate=0.1)
    # Network B
    w2B = tf.get_variable(shape = [hid2_size, hid1_size], name='w2B',
                          trainable = True)
    b2B = tf.get_variable(shape = [hid2_size, 1], name='b2B', trainable = True)
    y2B = tf.nn.dropout(tf.nn.relu(tf.add(
                        tf.matmul(w2B, y1B), b2B)),
                        rate=0.1)
    # ## Third layer
    hid3_size = 100
    # Network A
    w3A = tf.get_variable(shape = [hid3_size, hid2_size], name='w3A',
                          trainable = True)
    b3A = tf.get_variable(shape = [hid3_size, 1], name='b3A', trainable = True)
    y3A = tf.nn.dropout(tf.nn.relu(tf.add(
                        tf.matmul(w3A, y2A), b3A)),
                        rate=0.1)
    # Network B
    w3B = tf.get_variable(shape = [hid3_size, hid2_size], name='w3B',
                          trainable = True)
    b3B = tf.get_variable(shape = [hid3_size, 1], name='b3B', trainable = True)
    y3B = tf.nn.dropout(tf.nn.relu(tf.add(
                        tf.matmul(w3B, y2B), b3B)),
                        rate=0.1)

    # similarity
    a = tf.matmul(tf.transpose(y3A),y3B) # n x n
    D = - tf.reduce_mean(a) + tf.reduce_sum(tf.math.reduce_logsumexp(a,axis=1))

    # Output layer
    out_size = 1
    wo1 = tf.get_variable(shape = [out_size, hid3_size], name='wo1', trainable = True)
    wo2 = tf.get_variable(shape = [out_size, hid3_size], name='wo2', trainable = True)
    bo = tf.get_variable(shape = [out_size, 1], name='bo', trainable = True)
    yo = tf.transpose(tf.add(tf.matmul(wo1, y3A), tf.add(tf.matmul(wo2, y3B), bo)))

    # loss = tf.reduce_mean(tf.square(y-yo))
    loss_pred = tf.reduce_mean(tf.square(y-yo))
    loss = loss_pred + l * D
    #tf.reduce_sum(S)
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)


    b = batchSize # batch size
    nEpochs = epochs

    rmseTrain = np.zeros(nEpochs) 
    rmseTest = np.zeros(nEpochs)

    saver = tf.train.Saver()

    stopEarly = False
    best_loss = 10**10
    a = 50 # generalization loss threshold
    # training
    for learning_rate in [0.0001]:
        for epoch in range(nEpochs):
            avg_cost = 0.0

            # For each epoch, we go through all the samples we have.
            for i in xrange(0, X1.shape[0], b):

                _, c = sess.run([optimizer, loss],
                                feed_dict={inputA: X1[i : (i + b), :],
                                           inputB: X2[i : (i + b), :],
                                           # I: K,
                                           y: np.expand_dims(y_train[i : (i + b)],
                                                              axis = -1)
                                })
                avg_cost += c
            avg_cost /= X1.shape[0]    

            # Print the cost in this epcho to the console.
            if epoch % 10 == 0:
                print("Epoch: {:3d}    Train Cost: {:.4f}".format(epoch, avg_cost))

            loss_train = sess.run(loss_pred,
                            feed_dict={inputA: X1,
                                       inputB: X2,
                                       y: np.expand_dims(y_train, axis= -1)
                            })
            rmseTrain[epoch] = np.sqrt(loss_train)

            loss_test = sess.run(loss_pred,
                            feed_dict={inputA: X1t,
                                      inputB: X2t,
                                       y: np.expand_dims(y_test, axis = -1)
                                })
            rmseTest[epoch] = np.sqrt(loss_test)

            # implementing https://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf
            GL = 100 * (loss_test/best_loss - 1)
            if (loss_test < best_loss):
                # stopping_step = 0
                saver.save(sess, 'intModels_Split/siamesesplitLASSO - epoch = ' +
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

    best = np.where(rmseTest == np.min(rmseTest[rmseTest <> 0]))
    best = best[0][0]

    # saver = tf.train.Saver()
    # saver.save(sess, 'models/siamesesplitLASSO - ' + ' std' + '.ckpt')
    saver.restore(sess, 'intModels_Split/siamesesplitLASSO - epoch = ' +
                           str(best) + '.ckpt')

    np.savetxt('prediction accuracy/X_hsiclasso_SSsplitLASSO-V2_1000.dat',
               np.array([rmseTest[best]]))
