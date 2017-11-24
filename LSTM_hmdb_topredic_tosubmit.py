import numpy as np
import tensorflow as tf
import time
import datetime, time
import os
import random
import csv

# IN this file, X refers to data and y to labels 

class init_vars:
    """ Initialize network variables, will be updated later in learnExternalData
        and predictExternalData
    """
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
    n_hidden = 32 
    n_classes = 1 #Total classes: will be updated later
    learning_rate = 0.03
    lambda_loss_amount = 0.003 
    its = 1000
    batch_size = 350 
    display_iter = 1000 
    test_num = 72
    valid_num = " "

def write_vars(values, folder):
    """Write variables to file
    """
    savefile = os.path.join(folder, str(values.st) + "net.txt")
    file = open(savefile, "w")
    file.write("n_hidden " + str(values.n_hidden) + "\n")
    file.write("n_classes " + str(values.n_classes) + "\n")
    file.write("learning_rate " + str(values.learning_rate) + "\n")
    file.write("lambda_loss_amount " + str(values.lambda_loss_amount) + "\n")
    file.write("training_iters " + str(values.its) + "\n")
    file.write("batch size " + str(values.batch_size) + "\n")
    file.write("lstm_cells " + str(1) + "\n")
    file.write("test_num " + str(values.test_num) + "\n")
    file.write("valid_num " + values.valid_num + "\n")
    file.close() 

def LSTM_RNN(_X, _weights, _biases, _n_hidden, n_input, n_steps):
    """Function to define the strucure of the network
    """
    # Reshape to prepare input to hidden activation 
    # Input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    _X = tf.reshape(_X, [-1, n_input]) 
    
    # Linear activation
    _X = tf.matmul(_X, _weights['hidden']) + _biases['hidden']
    
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(0, n_steps, _X) 

    # Define network with three stacked LSTM cells with _n_hidden units
    lstm_cell_1 = tf.nn.rnn_cell.BasicLSTMCell(_n_hidden, forget_bias=3)
    lstm_cell_2 = tf.nn.rnn_cell.BasicLSTMCell(_n_hidden, forget_bias=3) 
    lstm_cell_3 = tf.nn.rnn_cell.BasicLSTMCell(_n_hidden, forget_bias=3)
    lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2, lstm_cell_3])
    
    # Get LSTM cell output
    outputs, states = tf.nn.rnn(lstm_cells, _X, dtype=tf.float32)

    # Linear activation
    return tf.matmul(outputs[-1], _weights['out']) + _biases['out']

def extract_batch_size(_train, step, batch_size, arr): 
    """Function to fetch a batch of data from "_train" of size "batch_size"
    """
    shape = list(_train.shape)
    shape[0] = batch_size
    batch_s = np.empty(shape)
    for i in range(batch_size):
        index = arr[((step-1)*batch_size + i) % len(_train)]
        batch_s[i] = _train[index] 

    return batch_s

def one_hot(y_, n_values):
    """Function to convert single labels to matrix form
    """
    y_ = y_.reshape(len(y_))
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  


def calculate_cost(output, target):
    """Function to calculate the cross entropy cost
    """
    # Compute cross entropy for each frame.
    cross_entropy = -tf.reduce_sum(target*tf.log(tf.clip_by_value(output,1e-10,1.0)))
    #cross_entropy = target * tf.log(output)
    cross_entropy = -tf.reduce_sum(cross_entropy) 
    mask = tf.sign(tf.reduce_max(tf.abs(target))) 
    cross_entropy *= mask
    # Average over actual sequence lengths.
    cross_entropy = tf.reduce_sum(cross_entropy) 
    cross_entropy /= tf.reduce_sum(mask) 
    return tf.reduce_mean(cross_entropy)


def setup_and_train_LSTM(var_, X_train, y_train, X_test, y_test, test_num, valid_ids, fname, savefolder):
    """Function to set up and train LSTM (in this function, test refers to validation data)
    """
    tf.reset_default_graph()
    prev_acc = 0.000
    
    # Initialize network variables
    training_data_count = len(X_train)  
    test_data_count = len(X_test)  
    n_steps = len(X_train[0])  
    n_input = len(X_train[0][0])  
    n_hidden = var_.n_hidden
    n_classes = var_.n_classes 
    learning_rate = var_.learning_rate
    lambda_loss_amount = var_.lambda_loss_amount 
    training_iters = training_data_count * var_.its 
    batch_size = var_.batch_size
    display_iter = var_.display_iter  
    
    # Graph input/output
    x = tf.placeholder(tf.float32, [None, n_steps, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])

    # Graph weights
    weights = {
        'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }
    biases = {
        'hidden': tf.Variable(tf.random_normal([n_hidden])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    pred = LSTM_RNN(x, weights, biases, n_hidden, n_input, n_steps)

    # Set up loss, optimizer and evaluation with L2 regularization
    l2 = lambda_loss_amount * sum(
        tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
    )
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) + l2 # Softmax loss
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = var_.learning_rate
    decayed_learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,100000, 0.96, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=decayed_learning_rate).minimize(cost, global_step=global_step)
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
 
    # Initialize variables to store loss and accuracy
    test_losses = []
    test_accuracies = []
    test_step = []
    train_losses = []
    train_accuracies = []  

    # Launch the graph
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
    init = tf.initialize_all_variables()
    sess.run(init)
    train_writer = tf.train.SummaryWriter(os.path.join(save_folder, 'tensorboard') , sess.graph)
    
    # Perform batch training 
    step = 1
    highest_acc = 0.00
    n_examples = len(y_train)
    steps_per_epoch =  int(n_examples/batch_size)
    arr = np.arange(n_examples)
    np.random.shuffle(arr)
    while step * batch_size <= training_iters:
        if (step % steps_per_epoch == 0):
            arr = np.arange(n_examples)
            np.random.shuffle(arr)
            print("NEW EPOCH")
        batch_xs = extract_batch_size(X_train, step, batch_size, arr)
        batch_ys = one_hot(extract_batch_size(y_train, step, batch_size, arr), n_classes)

        # Fit training using batch data
        _, loss, acc = sess.run(
            [optimizer, cost, accuracy],
            feed_dict={
                x: batch_xs, 
                y: batch_ys
            }
        )
        train_losses.append(loss)
        train_accuracies.append(acc)
		
        acc_training = np.float32(acc)
        
        # Evaluate network only at some steps for faster training: 
        if (step * batch_size % display_iter == 0) or (step == 1) or (step * batch_size > training_iters) or (step % steps_per_epoch == 0):
            # Evaluation on the test set (no learning here)
            loss, acc = sess.run(
                [cost, accuracy], 
                feed_dict={
                    x: X_test,
                    y: one_hot(y_test, n_classes)
                }
            )
            test_losses.append(loss)
            
            if(np.float32(acc) > highest_acc):
               highest_acc = np.float32(acc)
               # Store session
                session_file = os.path.join(save_folder, 'models_' + fname,
                                            'models_' + fname + str(test_num) + "b" + str(batch_size)\
                                            + "h" + str(n_hidden) + "v" + valid_ids + ".ckpt")
               saver.save(sess,  session_file)
               print("BEST ACC SO FAR: " + str(highest_acc))
            print ("PERFORMANCE ON TEST SET: " + "Batch Loss = {}".format(loss) + ", Accuracy = {}".format(acc))
            summary = tf.Summary(value=[tf.Summary.Value(tag="summary_tag", simple_value=np.float32(acc).item(acc)),])
            train_writer.add_summary(summary)
        step += 1

    print ("Optimization Finished!")

   
	# Make predictions on test data
    one_hot_predictions, accuracy, final_loss = sess.run(
        [pred, accuracy, cost],
        feed_dict={
            x: X_test,
            y: one_hot(y_test, n_classes)
        }
    )

    test_losses.append(final_loss)
    test_accuracies.append(accuracy)
    test_step.append(step+1)
    
    print ("FINAL RESULT: " + "Batch Loss = {}".format(final_loss) + ", Accuracy = {}".format(accuracy))
    print(test_accuracies)
    print(test_step)
	
	
def predict(var_, X_train, X_test, y_test, model_name, model_num, fname, folder):
    """Function to make predictions on test data from pre-trained LSTM
    """
    tf.reset_default_graph()
    prev_acc = 0.000
    
    # Initialize network variables
    training_data_count = len(X_train)  
    test_data_count = len(X_test)  
    n_steps = len(X_train[0])  
    n_input = len(X_train[0][0])  
    n_hidden = var_.n_hidden
    n_classes = var_.n_classes 
    print(n_classes)
    learning_rate = var_.learning_rate
    lambda_loss_amount = var_.lambda_loss_amount 
    training_iters = training_data_count * var_.its 
    batch_size = var_.batch_size
    display_iter = var_.display_iter  
    
    # Graph input/output
    x = tf.placeholder(tf.float32, [None, n_steps, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])

    # Graph weights
    weights = {
        'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }
    biases = {
        'hidden': tf.Variable(tf.random_normal([n_hidden])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    pred = LSTM_RNN(x, weights, biases, n_hidden, n_input, n_steps)

    
    # Set up loss, optimizer and evaluation with L2 regularization
    l2 = lambda_loss_amount * sum(
        tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
    )
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) + l2 
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = var_.learning_rate
    decayed_learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,100000, 0.96, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=decayed_learning_rate).minimize(cost, global_step=global_step)
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
     
    saver = tf.train.Saver()
    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
    init = tf.initialize_all_variables()
    sess.run(init)
    
    # Restore session of learned network from training data
    saver.restore(sess, model_name)
    
    # Make predictions
    pred, acc, loss = sess.run(
        [pred, accuracy, cost],
        feed_dict={
            x: X_test,
            y: one_hot(y_test, n_classes)
        }
    )
    print(acc)
    print(loss)
    
    # Directory to save predictions
    csv_path = os.path.join(folder, "predictions_" + fname, "predictions" + str(model_num) + ".csv")
    csv_path2 = os.path.join(folder, "predictions_" + fname, "proba" + str(model_num) + ".csv")
    
    myfile = open(csv_path, "w") 
    wr = csv.writer(myfile)
    myfile2 = open(csv_path2, "w") 
    wr2 = csv.writer(myfile2)
    for j in range(len(pred)): 
        val_pred = np.argmax(pred[j]) 
        dat = np.array([y_test[j], val_pred])
        wr.writerow(dat)
        dat2 = np.array([y_test[j]])
        dat2 = np.append(dat2,pred[j])
        wr2.writerow(dat2)
        
def learnExternalData(classesnum, X_train, y_train, X_valid, y_valid, valid_num, valid_ids, fname):
    """Save file with hypreparameters and launch learning
    """
    max_feats = 2048
    max_time = 20	
            
    v = init_vars
    v.n_hidden = 90
    v.learning_rate = 0.001
    v.lambda_loss_amount = 0.003
    v.batch_size = 50
    v.its = 350 
    ts = time.time()
    v.n_classes = classesnum
    v.st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')	
    v.test_num = valid_num
    v.valid_num = valid_ids
    write_vars(v)
    # Start Learning (learned model will be save in learnExternalData)
    setup_and_train_LSTM(v, X_train, y_train, X_valid, y_valid, valid_num, valid_ids, fname)

def predictExternalData(classesnum, X_train, X_test, y_test, model_name, model_num, batch_s, hidden_s, lr, fname):
    """Select model and launch prediction
    """
    max_feats = 2048
    max_time = 20			
    v = init_vars
    v.n_hidden = 90
    v.learning_rate = 0.001
    v.lambda_loss_amount = 0.003
    v.batch_size = 50 
    v.its = 350 
    ts = time.time()
    v.n_classes = classesnum
    v.st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')	
    # Return predictions
    return predict(v,X_train, X_test,y_test, model_name, model_num,fname)

