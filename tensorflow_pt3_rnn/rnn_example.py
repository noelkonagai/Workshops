import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
from sklearn.preprocessing import MinMaxScaler

num_inputs = 1
num_outputs = 1
batch_size = 1

parser = argparse.ArgumentParser(description='Add some hyper parameters')

parser.add_argument('--timesteps', dest = 'num_time_steps', default = 12, type = int, help='add an integer for the time steps to be predicted')
parser.add_argument('--neurons', dest = 'num_neurons', default = 100, type = int, help='add an integer for the number of neurons you want to use')
parser.add_argument('--lr', dest = 'learning_rate', default = 0.03, type = float, help='add a float for the learning rate')
parser.add_argument('--iter', dest = 'num_train_iterations', default = 4000, type = int, help='add an integer for the number of training iterations')
parser.add_argument('--layer', dest = 'num_layers', default = 1, type = int, help='add an integer for the number of layers used')

args = parser.parse_args()

### VARIABLES
num_time_steps = args.num_time_steps
num_neurons = args.num_neurons
learning_rate = args.learning_rate
num_train_iterations = args.num_train_iterations
num_layers = args.num_layers

milk = pd.read_csv('monthly-milk-production.csv',index_col='Month')
milk.head()
milk.index = pd.to_datetime(milk.index)

train_set = milk.head(156)
test_set = milk.tail(num_time_steps)

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_set)
test_scaled = scaler.transform(test_set)

def next_batch(training_data,batch_size,steps):
	# Grab a random starting point for each batch
	rand_start = np.random.randint(0,len(training_data)-steps) 

	# Create Y data for time series in the batches
	y_batch = np.array(training_data[rand_start:rand_start+steps+1]).reshape(1,steps+1)

	return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1) 

X = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
y = tf.placeholder(tf.float32, [None, num_time_steps, num_outputs])

cell_1 = tf.contrib.rnn.OutputProjectionWrapper(
    tf.contrib.rnn.BasicLSTMCell(num_units = num_neurons, activation = tf.nn.relu),
    output_size = num_outputs) 

cell_2 = tf.contrib.rnn.OutputProjectionWrapper(
    tf.contrib.rnn.GRUCell(num_units = num_neurons, activation = tf.nn.relu),
    output_size = num_outputs)

cell = tf.nn.rnn_cell.MultiRNNCell([cell_2, cell_2], state_is_tuple=True)

outputs, states = tf.nn.dynamic_rnn(cell, X, dtype = tf.float32)

loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
	sess.run(init)

	for iteration in range(num_train_iterations):

		X_batch, y_batch = next_batch(train_scaled,batch_size,num_time_steps)
		sess.run(train, feed_dict={X: X_batch, y: y_batch})

		if iteration % 100 == 0:

			mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
			print(iteration, "\tMSE:", mse)

	# Save Model for Later
	saver.save(sess, "./ex_time_series_model")

with tf.Session() as sess:
	# Use your Saver instance to restore your saved rnn time series model
	saver.restore(sess, "./ex_time_series_model")

	# Create a numpy array for your genreative seed from the last 12 months of the 
	# training set data. Hint: Just use tail(num_time_steps) and then pass it to an np.array
	train_seed = list(train_scaled[-num_time_steps:])

	## Now create a for loop that 
	for iteration in range(num_time_steps):
		X_batch = np.array(train_seed[-num_time_steps:]).reshape(1, num_time_steps, 1)
		y_pred = sess.run(outputs, feed_dict={X: X_batch})
		train_seed.append(y_pred[0, -1, 0])

results = scaler.inverse_transform(np.array(train_seed[num_time_steps:]).reshape(num_time_steps,1))

test_set.is_copy = False
test_set['Generated'] = results

test_set.plot()
plt.show()