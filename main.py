import gym
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf

print('cool ~(<.<)~')

def pre_proc(I):
	I = I[35:195]	
	I = I[::2,::2,0]
	I[I == 144] = 0
	I[I == 109] = 0
	I[I != 0] = 1
	return I.astype(np.float).ravel()

def append_to_file(obs):
	with open("test1.txt", "a") as myfile:
		for i in range(len(obs)):
			if i == len(obs)-1:
				myfile.write(str(obs[i]))
			else:
				myfile.write(str(obs[i])+" ")
		myfile.write('\n')


env_name1 = 'Breakout-v0'
env_name2 = 'CartPole-v0'
env_name3 = 'Pong-v0'
env_name4 = 'Phoenix-v0'
env_name5 = 'Assault-v0'
env = gym.make(env_name5)

observation = env.reset()
prev_img = pre_proc(observation)
img_shape = np.shape(prev_img)
img = [[0.0 for i in range(80)]for j in range(80)]

for i in range(5):
	action = env.action_space.sample()	

	obs, reward, done, info = env.step(action)

	cur_img = pre_proc(obs)

	img= np.subtract(cur_img, prev_img)

	prev_img = cur_img

	plt.imshow(img.reshape(80,80),cmap='gray',aspect='auto',animated=True)
	plt.show()


#print(np.shape(img))
#img = img.reshape(8,10)
#print(np.shape(img))


'''
Step 1) Create Placeholders

Step 2) Create Variables

Step 3) Create Graph operations

Step 4) Create Loss Function

Step 5) Create Optimizer

Step 6) Initialize variables and create Session

Step 7) Evaluate the model
'''
batch_size = 1
num_classes = 3
num_steps = 2000
num_layers = 1
nodes_per_lay = 10
epochs = 5

#open AI variables
num_inputs = 4
num_outputs = 1
step_limit = 500
avg_steps = []


def add_layers(nodes_per_lay,num_lay,lay_1):
	w = tf.Variable(tf.random_uniform([nodes_per_lay,nodes_per_lay]))
	b = tf.Variable(tf.random_uniform([nodes_per_lay]))
	y = tf.nn.relu(tf.matmul(lay_1,w)+b)
	if num_lay == 0:
		return y
	else:
		return add_layers(nodes_per_lay,num_lay-1,y)


batch_size = 100
num_classes = 10
num_steps = 2000
num_layers = 2
nodes_per_lay = 10
num_inputs = 4
epochs = 20


#Step 1) Create Placeholders
y_sen = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32,shape=[batch_size,num_inputs])
y_true = tf.placeholder(tf.float32,[batch_size,num_classes])

#Step 2) Create Variables
W_in = tf.Variable(tf.truncated_normal([num_inputs,nodes_per_lay],stddev=.1))
b_in = tf.Variable(tf.truncated_normal([nodes_per_lay],stddev=.1))

W_out = tf.Variable(tf.truncated_normal([nodes_per_lay,num_classes],stddev=.1))
b_out = tf.Variable(tf.truncated_normal([num_classes],stddev=.1))

#Step 3) Create Graph
y_in = tf.nn.relu(tf.matmul(x,W_in) + b_in)

y_hid = add_layers(nodes_per_lay,num_layers,y_in)

y = tf.matmul(y_hid,W_out) + b_out

#Step 4) Loss Function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y))

#Step 5) Create optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=.03).minimize(cost)

'''
#Step 6) Create Session
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)

	for ep in range(epochs):
		for steps in range(num_steps):
			batch_x,batch_y = mnist.train.next_batch(100)
			sess.run(optimizer,feed_dict={x:batch_x,y_true:batch_y})

		correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_true,1))

		acc = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	
		print(sess.run(acc,feed_dict={x:mnist.test.images,y_true:mnist.test.labels}))
'''

