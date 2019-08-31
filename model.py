# %matplotlib inline

import logging
import config
import numpy as np

import matplotlib.pyplot as plt

from keras.models import Sequential, load_model, Model  #加载目录下的模型.h5文件
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, LeakyReLU, add
from keras.optimizers import SGD
from keras import regularizers

from loss import softmax_cross_entropy_with_logits

import loggers as lg

import keras.backend as K

from settings import run_folder, run_archive_folder

#该文件使用了AlphaGo Zero论文中的神经网络结构的浓缩版本，由许多残差层构成，然后分裂成价值和策略两个分支

class Gen_Model():
	def __init__(self, reg_const, learning_rate, input_dim, output_dim):
		self.reg_const = reg_const
		self.learning_rate = learning_rate
		self.input_dim = input_dim
		self.output_dim = output_dim

	def predict(self, x):
		return self.model.predict(x)
	#可以通过history来查看训练过程，loss值等等
	def fit(self, states, targets, epochs, verbose, validation_split, batch_size):
		#fit中自动包含了compile的模块，序贯模型
		return self.model.fit(states, targets, epochs=epochs, verbose=verbose, validation_split = validation_split, batch_size = batch_size)

	def write(self, game, version):
		self.model.save(run_folder + 'models/version' + "{0:0>4}".format(version) + '.h5')

	def read(self, game, run_number, version):
		return load_model( run_archive_folder + game + '/run' + str(run_number).zfill(4) + "/models/version" + "{0:0>4}".format(version) + '.h5', custom_objects={'softmax_cross_entropy_with_logits': softmax_cross_entropy_with_logits})

    #大部分是keras自带的方法
	#打印权重
	def printWeightAverages(self):
		layers = self.model.layers  #组成模型图的各个层，
		for i, l in enumerate(layers): #i用来计数，l（L）用来接收layer
			try:
				#权重层 [0]：说明layer.get_weights()返回是二维数组
				x = l.get_weights()[0]
				lg.logger_model.info('WEIGHT LAYER %d: ABSAV = %f, SD =%f, ABSMAX =%f, ABSMIN =%f', i, np.mean(np.abs(x)), np.std(x), np.max(np.abs(x)), np.min(np.abs(x)))
			except:
				pass
		lg.logger_model.info('------------------')
		for i, l in enumerate(layers):
			try:
				#偏置层
				x = l.get_weights()[1]
				lg.logger_model.info('BIAS LAYER %d: ABSAV = %f, SD =%f, ABSMAX =%f, ABSMIN =%f', i, np.mean(np.abs(x)), np.std(x), np.max(np.abs(x)), np.min(np.abs(x)))
			except:
				pass
		lg.logger_model.info('******************')

	#查看神经网络的单个卷积过滤和密集连接的层，代码不用管
	def viewLayers(self):
		layers = self.model.layers
		for i, l in enumerate(layers):
			x = l.get_weights()
			print('LAYER ' + str(i))

			try:
				#layer权重
				weights = x[0]
				#shape的元素个数代表维度，元素的值代表每一维的长度
				s = weights.shape
				#至少是4维，且第三维和第四维是长和宽的大小s[2]：width   s[3]：height
				fig = plt.figure(figsize=(s[2], s[3]))  # width, height in inches
				channel = 0
				filter = 0
				for i in range(s[2] * s[3]):

					sub = fig.add_subplot(s[3], s[2], i + 1)
					sub.imshow(weights[:,:,channel,filter], cmap='coolwarm', clim=(-1, 1),aspect="auto")
					channel = (channel + 1) % s[2]
					filter = (filter + 1) % s[3]

			except:
	
				try:
					fig = plt.figure(figsize=(3, len(x)))  # width, height in inches
					for i in range(len(x)):
						sub = fig.add_subplot(len(x), 1, i + 1)
						if i == 0:
							clim = (0,2)
						else:
							clim = (0, 2)
						sub.imshow([x[i]], cmap='coolwarm', clim=clim,aspect="auto")
						
					plt.show()

				except:
					try:
						fig = plt.figure(figsize=(3, 3))  # width, height in inches
						sub = fig.add_subplot(1, 1, 1)
						sub.imshow(x[0], cmap='coolwarm', clim=(-1, 1),aspect="auto")
						
						plt.show()

					except:
						pass

			plt.show()
				
		lg.logger_model.info('------------------')


class Residual_CNN(Gen_Model):
	#reg_const:用于规则化，解决过拟合问题
	#input_dim：input_shape    output_dim：action_size     hidden_layers：config.HIDDEN_CNN_LAYERS = 6
	#其中input_shape = (2,6,7)   2：黑白 6：行 7：列
	def __init__(self, reg_const, learning_rate, input_dim,  output_dim, hidden_layers):
		Gen_Model.__init__(self, reg_const, learning_rate, input_dim, output_dim)
		self.hidden_layers = hidden_layers
		self.num_layers = len(hidden_layers)
		self.model = self._build_model()

	#残差块
	def residual_layer(self, input_block, filters, kernel_size):

		x = self.conv_layer(input_block, filters, kernel_size)
		#卷积
		x = Conv2D(
		filters = filters
		, kernel_size = kernel_size
		, data_format="channels_first"
		, padding = 'same'
		, use_bias=False
		, activation='linear'
		, kernel_regularizer = regularizers.l2(self.reg_const)
		)(x)
		#正则化
		x = BatchNormalization(axis=1)(x)
		#Add shortcut value to main path
		x = add([input_block, x])
		#pass it through a RELU activation
		x = LeakyReLU()(x)

		return (x)
	#正常的卷积层
	def conv_layer(self, x, filters, kernel_size):

		x = Conv2D(
		filters = filters
		, kernel_size = kernel_size
		, data_format="channels_first"
		, padding = 'same'
		, use_bias=False
		, activation='linear'
		, kernel_regularizer = regularizers.l2(self.reg_const)
		)(x)

		x = BatchNormalization(axis=1)(x)
		x = LeakyReLU()(x)

		return (x)

	#价值网络
	def value_head(self, x):

		x = Conv2D(
		filters = 1
		, kernel_size = (1,1)
		, data_format="channels_first"
		, padding = 'same'
		, use_bias=False
		, activation='linear'
		, kernel_regularizer = regularizers.l2(self.reg_const)
		)(x)


		x = BatchNormalization(axis=1)(x)
		x = LeakyReLU()(x)

		x = Flatten()(x)

		x = Dense(
			20
			, use_bias=False
			, activation='linear'
			, kernel_regularizer=regularizers.l2(self.reg_const)
			)(x)

		x = LeakyReLU()(x)

		x = Dense(
			1
			, use_bias=False
			, activation='tanh'
			, kernel_regularizer=regularizers.l2(self.reg_const)
			, name = 'value_head'
			)(x)
		# 输出的数组长度是1，表示对棋盘局势的判断【注：论文中说返回的是一个预测值】
		return (x)

	#策略网络
	def policy_head(self, x):

		x = Conv2D(
		filters = 2
		, kernel_size = (1,1)
		, data_format="channels_first"
		, padding = 'same'
		, use_bias=False
		, activation='linear'
		, kernel_regularizer = regularizers.l2(self.reg_const)
		)(x)

		x = BatchNormalization(axis=1)(x)
		x = LeakyReLU()(x)

		x = Flatten()(x)

		x = Dense(
			self.output_dim   #output_dim为棋盘的大小
			, use_bias=False
			, activation='linear'
			, kernel_regularizer=regularizers.l2(self.reg_const)
			, name = 'policy_head'
			)(x)
		#输出的数组长度为42，代表了棋盘上的42个点 【注：论文中说返回的是一组概率分布】
		return (x)

	def _build_model(self):

		main_input = Input(shape = self.input_dim, name = 'main_input')
		#第一层为卷积层
		x = self.conv_layer(main_input, self.hidden_layers[0]['filters'], self.hidden_layers[0]['kernel_size'])
		#从第二层开始，到第六层，共五层，都使用残差网络
		if len(self.hidden_layers) > 1:
			for h in self.hidden_layers[1:]:
				x = self.residual_layer(x, h['filters'], h['kernel_size'])
		#返回价值（得分为0到1之间，为1的时候是获胜）
		vh = self.value_head(x)
		#返回策略（棋盘上落子的位置的概率分布）
		ph = self.policy_head(x)

		"""
		我们起初将Functional一词译作泛型，想要表达该类模型能够表达任意张量映射的含义，
		但表达的不是很精确，在Keras 2里我们将这个词改译为“函数式”，对函数式编程有所
		了解的同学应能够快速get到该类模型想要表达的含义。函数式模型称作Functional，
		但它的类名是Model，因此我们有时候也用Model来代表函数式模型
		"""
		#This creates a model that includes,这样的模型可以被像Keras的Sequential一样被训练
		model = Model(inputs=[main_input], outputs=[vh, ph])

		#keras自带的模型编译，优化及损失函数
		model.compile(loss={'value_head': 'mean_squared_error', 'policy_head': softmax_cross_entropy_with_logits},
			optimizer=SGD(lr=self.learning_rate, momentum = config.MOMENTUM),	
			loss_weights={'value_head': 0.5, 'policy_head': 0.5}	
			)

		return model

	#输入格式的转换
	def convertToModelInput(self, state):
		#返回当前玩家棋盘上自己点的位置，以及对手点的位置
		inputToModel =  state.binary #np.append(state.binary, [(state.playerTurn + 1)/2] * self.input_dim[1] * self.input_dim[2])
        #input_dim = input_shape = (2,6,7)   2：黑白 6：行 7：列
		inputToModel = np.reshape(inputToModel, self.input_dim)
		return (inputToModel)
