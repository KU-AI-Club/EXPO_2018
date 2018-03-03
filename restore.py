#!/usr/bin/env python
from __future__ import print_function
import gym
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys, gym, time

env_name1 = 'Breakout-v0'
env_name2 = 'CartPole-v0'
env_name3 = 'Pong-v0'
env_name4 = 'Phoenix-v0'
env_name5 = 'Assault-v0'
env = gym.make(env_name4)

if not hasattr(env.action_space, 'n'):
    raise Exception('Keyboard agent only supports discrete action spaces')
ACTIONS = env.action_space.n
fig = plt.figure()
human_agent_action = 0
human_wants_restart = False
human_sets_pause = False

def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause
    if key==0xff0d: human_wants_restart = True
    if key==32: human_sets_pause = not human_sets_pause
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    human_agent_action = a

def key_release(key, mod):
    global human_agent_action
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    if human_agent_action == a:
        human_agent_action = 0

def pre_proc(I):
	I = I[35:195]	
	I = I[::2,::2,0]
	I[I == 144] = 0
	I[I == 109] = 0
	I[I != 0] = 1
	return I.astype(np.float).ravel()

env.render()
env.unwrapped.viewer.window.on_key_press = key_press
env.unwrapped.viewer.window.on_key_release = key_release

def rollout(env):
	pix_list = []
	act_list = []
	fin_act = []

	img = [[0.0 for i in range(80)]for j in range(80)]
	cur_img = [[0.0 for i in range(80)]for j in range(80)]
	prev_img = [[0.0 for i in range(80)]for j in range(80)]

	global human_agent_action, human_wants_restart, human_sets_pause
	obs = env.reset()
	prev_img = pre_proc(obs)
	skip = 0
	total_timesteps = 0
	while 1:
		if not skip:
			a = human_agent_action
			total_timesteps += 1

		obs, r, done, info = env.step(a)
		cur_img = pre_proc(obs)

		img = np.subtract(cur_img, prev_img)

		prev_img = cur_img

		act_list.append(a)
		pix_list.append(img)
		if(r == -1):
			del act_list[-15:]
			del pix_list[-15:]
			#del img_list[-15:]
			print("reward: -1")
		if(r > 0):
			for i in range(15):
				act_list.append(a)
				pix_list.append(img)
				#img_list.append(img)
			print("reward: "+str(r))

		window_still_open = env.render()
		if window_still_open==False: 
			for i in range(len(act_list)):
				if(act_list[i] == 0):
					fin_act.append([1,0,0,0,0,0])
				elif(act_list[i] == 1):
					fin_act.append([0,1,0,0,0,0])
				elif(act_list[i] == 2):
					fin_act.append([0,0,1,0,0,0])
				elif(act_list[i] == 3):
					fin_act.append([0,0,0,1,0,0])
				elif(act_list[i] == 4):
					fin_act.append([0,0,0,0,1,0])
				else:
					fin_act.append([0,0,0,0,0,1])

			return False,pix_list,fin_act
		if done: return False,pix_list,fin_act

		while human_sets_pause:
		    env.render()
		    time.sleep(0.001)
		time.sleep(0.05)

while 1:
	window_still_open,pix_list,fin_act = rollout(env)	
	if window_still_open==False: break


print('cool ~(<.<)~')
print(len(fin_act))

def add_layers(nodes_per_lay,num_lay,lay_1):
	w = tf.Variable(tf.random_uniform([nodes_per_lay,nodes_per_lay]))
	b = tf.Variable(tf.random_uniform([nodes_per_lay]))
	y = tf.nn.relu(tf.matmul(lay_1,w)+b)
	if num_lay == 0:
		return y
	else:
		return add_layers(nodes_per_lay,num_lay-1,y)

def get_next_batch(batch_size,i,pix_list):
	return pix_list[i:i+batch_size]

batch_size = 100#len(fin_act)
num_inputs = 6400
num_classes = 6
num_layers = 1
nodes_per_lay = 5
epochs = 2
num_steps = 1000
j = 0


#Step 1) Create Placeholders
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

#Step 6) Create Session
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	saver = tf.train.Saver()

	for ep in range(epochs):
		for steps in range(num_steps):
			if(num_steps > int(len(pix_list)/batch_size)):
				j = 0
			batch_x = get_next_batch(batch_size,j,pix_list)
			batch_y = get_next_batch(batch_size,j,fin_act)
			#if(len(batch_x.get_shape()) == 0):
			#	batch_x = pix_list[0:batch_size]
			#	batch_y = fin_act[0:batch_size]
			sess.run(optimizer,feed_dict={x:batch_x,y_true:batch_y})
			j = j+batch_size

		correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_true,1))

		acc = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
		print("accuracy: ")
		print(sess.run(acc,feed_dict={x:pix_list[0:batch_size],y_true:fin_act[0:batch_size]}))
	#save_path = saver.save(sess,"Ryan/temp1")


	img_list = []
	pix_list = []
	act_list = []

	img = [[0.0 for i in range(80)]for j in range(80)]
	cur_img = [[0.0 for i in range(80)]for j in range(80)]
	prev_img = [[0.0 for i in range(80)]for j in range(80)]

	observation = env.reset()
	prev_img = pre_proc(observation)

	action = env.action_space.sample()
	obs, reward, done, info = env.step(action)

	while(done==False):
		cur_img = pre_proc(obs)

		img = np.subtract(cur_img, prev_img)
		#ret = plt.imshow(img.reshape(80,80),animated=True)
		#img_list.append([ret])

		brand_new_data = [[img[i] for i in range(len(img))] for j in range(batch_size)]

		prediction = y.eval(feed_dict={x:brand_new_data})
		for i in range(batch_size):
			for j in range(len(prediction[i])):
				if (prediction[i][j] != prediction[j].max()):
					prediction[i][j] = 0
				else:
					prediction[i][j] = 1

		if(prediction[0][0]==1):
			action = 1
		elif(prediction[0][1]==1):
			action = 2
		elif(prediction[0][2]==1):
			action = 3
		elif(prediction[0][3]==1):
			action = 4
		elif(prediction[0][4]==1):
			action = 5
		else:
			action = 6

		print(action)

		obs, reward, done, info = env.step(action)
		env.render()

	#ani = animation.ArtistAnimation(fig,img_list,interval=50,blit=True,repeat_delay = 1000)
	#plt.show()


