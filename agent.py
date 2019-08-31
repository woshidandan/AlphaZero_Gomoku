# %matplotlib inline

import numpy as np
import random

import MCTS as mc
from game import GameState
from loss import softmax_cross_entropy_with_logits

import config
import loggers as lg
import time

import matplotlib.pyplot as plt
from IPython import display
import pylab as pl


class User():
	def __init__(self, name, state_size, action_size):
		self.name = name
		self.state_size = state_size
		self.action_size = action_size

	def act(self, state, tau):
		#t范围是0~41
		t = input('Enter your chosen action: ')
		action = int(t)
		#pi为当前表示棋盘落子情况的一维数组
		pi = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.int)
		#表示在下标action的位置落子
		pi[action] = 1
		value = 0
		NN_value = 0
		return (action, pi, value, NN_value)



class Agent():
	def __init__(self, name, state_size, action_size, mcts_simulations, cpuct, model):
		self.name = name

		self.state_size = state_size
		self.action_size = action_size

		self.cpuct = cpuct

		self.MCTSsimulations = mcts_simulations
		self.model = model

		self.mcts = None

		self.train_overall_loss = []
		self.train_value_loss = []
		self.train_policy_loss = []
		self.val_overall_loss = []
		self.val_value_loss = []
		self.val_policy_loss = []

	
	def simulate(self):

		lg.logger_mcts.info('ROOT NODE...%s', self.mcts.root.state.id)
		self.mcts.root.state.render(lg.logger_mcts)
		lg.logger_mcts.info('CURRENT PLAYER...%d', self.mcts.root.state.playerTurn)

		##### MOVE THE LEAF NODE
		#moveToLeaf return 的是currentNode, value, done, breadcrumbs。
		#移的也是不存在的点，但是是上次保存的最好的一棵树，只是不存在与棋盘上
		leaf, value, done, breadcrumbs = self.mcts.moveToLeaf()
		leaf.state.render(lg.logger_mcts)

		##### EVALUATE THE LEAF NODE ，而价值-1所在的分支，则会显著提高得分
		"""
		value:-1,0,或者其他，类似[0.02709729]
		"""
		value, breadcrumbs = self.evaluateLeaf(leaf, value, done, breadcrumbs)

		##### BACKFILL THE VALUE THROUGH THE TREE，在回传的过程中统计，累加计算
		"""
		value的值为-1或者其他，如[0.02723296]
		"""
		#回传的主要目的对于真实棋盘来说，是为棋盘选择下一步的真实落子，在众多子树的根节点中，找到手段
		self.mcts.backFill(leaf, value, breadcrumbs)


	def act(self, state, tau):
		#act在funcs.py中以players[state.playerTurn]['agent'].act(state, 1)调用
		if self.mcts == None or state.id not in self.mcts.tree:
			#不存在则建立
			self.buildMCTS(state)
		else:
			#存在则将当前的棋盘状态存入新的root中
			self.changeRootMCTS(state)
		#Expansion是随机的 An unvisited child position is randomly chosen,
		# and a new record node is added to the tree of statistics.
		#### run the simulation
		for sim in range(self.MCTSsimulations):
			lg.logger_mcts.info('***************************')
			lg.logger_mcts.info('****** SIMULATION %d ******', sim + 1)
			lg.logger_mcts.info('***************************')
			##### BACKFILL THE VALUE THROUGH THE TREE，得到了目前MCTS树上已存在节点的所有信息
			#value的值为 - 1 或者其他，如[0.02723296]
			self.simulate()

		"""getAV
		get action values，收集上面模拟出来的概率和价值信息
		收集模拟得到的数据，因为BackFill没有返回值，而是储存在边中
		之所以会有getAV方法，一方面是因为模拟过程中最后BackFill方法没有返回值，而是直接将信息存储在了边中
		另一方面，是对传回来的数据重新进行加工处理：归一化
		values：赢时为下面情况，1的位置为分出胜负的位置。否则为全0
		[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1（注意） 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 		0 0 0 0 0]
		"""
		#这个values还是之前的，不是模拟得到的概率最大处落子后的价值
		pi, values = self.getAV(1)

		####pick the action   value:0或者1
		##其实，可以把game中判断终态的action以这里的action代替，现在game里的action只是表示
		#棋盘上合法落子处的终态判定，而不是选出的最大概率的落子处的判断，之后考虑如何能做到这点
		action, value = self.chooseAction(pi, values, tau)

		nextState, _, _ = state.takeAction(action)

		#对于棋盘采取行动后的下个局面进行预测，
		NN_value = -self.get_preds(nextState)[0]

		lg.logger_mcts.info('ACTION VALUES...%s', pi)
		lg.logger_mcts.info('CHOSEN ACTION...%d', action)
		#两者都为对整体盘面的评分（价值），value为当前局面得分，NN_value为下个局面得分，若value大，则NN_value必然小
		lg.logger_mcts.info('MCTS PERCEIVED VALUE...%f', value)
		lg.logger_mcts.info('NN PERCEIVED VALUE...%f', NN_value)


		return (action, pi, value, NN_value)


	def get_preds(self, state):
		#predict the leaf[bool]类型什么意思
		inputToModel = np.array([self.model.convertToModelInput(state)])

		#这里的model.predict方法，是隐藏在keras自带的方法中，通过与残差层所得到的结果进行预测
		preds = self.model.predict(inputToModel)
		"""preds
		[array([[0.02701382]], dtype=float32), array([[-0.04089829,  0.08351101, -0.02361932,  0.05547938, -0.00054779,
        -0.01910474, -0.0731377 , -0.08106802, -0.0101286 , -0.14144059,
         0.14131209,  0.06545704, -0.10707675, -0.06580708,  0.07679133,
         0.04345218, -0.01657991,  0.0480116 , -0.00542973,  0.05496042,
         0.00699861, -0.09781487, -0.00731746, -0.04284807,  0.00093259,
        -0.04270414, -0.06927239, -0.02143533, -0.06765351,  0.0138868 ,
        -0.04875196, -0.06810682,  0.01954002,  0.02299483, -0.05302591,
        -0.02226484, -0.03062767, -0.03324477, -0.01096612,  0.00257285,
        -0.0247683 , -0.05152546]], dtype=float32)]
		"""
		##preds[0]：[[0.02767947]]
		value_array = preds[0]

		# 返回策略（棋盘上落子的位置）
		"""preds[1]
		[[-1.00000000e+02 -1.00000000e+02 -1.00000000e+02 -1.00000000e+02
  		-1.00000000e+02 -1.00000000e+02 -1.00000000e+02 -1.00000000e+02
 		 -1.00000000e+02 -1.00000000e+02 -1.00000000e+02 -1.00000000e+02
 		 -1.00000000e+02 -1.00000000e+02 -1.00000000e+02 -1.00000000e+02
		-1.00000000e+02 -1.00000000e+02 -1.00000000e+02 -1.00000000e+02
		-1.00000000e+02 -1.00000000e+02 -1.00000000e+02 -1.00000000e+02
		-1.00000000e+02 -1.00000000e+02 -1.00000000e+02 -1.00000000e+02
		-1.00000000e+02  2.06381418e-02 -1.00000000e+02 -1.00000000e+02
		3.01597808e-02 -1.00000000e+02 -1.00000000e+02 -1.01091927e-02
		-1.00000000e+02 -1.53663065e-02 -7.51875713e-03 -1.00000000e+02
		-1.67443343e-02 -5.12181446e-02]]
		"""
		logits_array = preds[1]

		#value:[0.02695561]
		value = value_array[0]

		logits = logits_array[0]
		#当前棋面的合法下法，实则也是为一种预测
		allowedActions = state.allowedActions

		#以下三段代码无用，删去不影响！！！
		# print(mask[allowedActions])
		mask = np.ones(logits.shape,dtype=bool)
		mask[allowedActions] = False
		logits[mask] = -100
		"""
		allowedActions:[28, 29, 37, 38, 39, 40, 41]
		mask[allowedActions]:[False False False False False False False]
		logits[mask]:[-100. -100. -100. -100. -100. -100. -100. -100. -100. -100. -100. -100.
					 -100. -100. -100. -100. -100. -100. -100. -100. -100. -100. -100. -100.
					 -100. -100. -100. -100. -100. -100. -100. -100. -100. -100. -100.]
		"""

		#SOFTMAX 计算最后每一步的概率，即得到先验概率
		odds = np.exp(logits)
		probs = odds / np.sum(odds) ###put this just before the for?
		"""这只是MCTS树上进行预测时的概率，没有考虑有没有合法，在probs = probs[allowedActions]之后的调用中考虑
		[0.02331666 0.02625092 0.02360239 0.02610105 0.02448202 0.02369746
		 0.02209597 0.02192812 0.02384036 0.02100579 0.02774534 0.02544567
		 0.02175146 0.02254345 0.02610964 0.0252581  0.02360901 0.0258019
		 0.02382462 0.02582104 0.02434498 0.02174089 0.02392007 0.02335073
		 0.02376547 0.02296119 0.02281758 0.02340958 0.02244577 0.02452217
		 0.02302461 0.0225678  0.02448444 0.02483897 0.02287729 0.02349304
		 0.02334984 0.02365649 0.02371265 0.02380427 0.02391164 0.02276963]
		"""

		return ((value, probs, allowedActions))


	def evaluateLeaf(self, leaf, value, done, breadcrumbs):

		lg.logger_mcts.info('------EVALUATING LEAF------')
		#评估该叶子节点，如果不是终态，则进行预测
		if done == 0:
			#经过预测得到的value值已经不是0或者-1了，因为对于非终态（目前棋盘上的棋子暂时无法判断胜负）
			# 的叶子节点返回的价值，无法决定整个棋盘的局势，需要通过神经网络来预测
			value, probs, allowedActions = self.get_preds(leaf.state)
			lg.logger_mcts.info('PREDICTED VALUE FOR %d: %f', leaf.state.playerTurn, value)
			#
			probs = probs[allowedActions]

			for idx, action in enumerate(allowedActions):
				# takeAction return (newState, value, done)
				newState, _, _ = leaf.state.takeAction(action)
				if newState.id not in self.mcts.tree:
					#获取节点的信息
					#节点的id其实是用棋盘黑白棋的状态合并来表示的
					#Node id：000000000000000000000000000000000001000000000000000000000000000000000010000000000000
					#node本身是一个包含了众多信息的“类”
					node = mc.Node(newState)
					#添加到树中
					self.mcts.addNode(node)
					lg.logger_mcts.info('added node...%s...p = %f', node.id, probs[idx])
				else:
					#tree相当于一个数组，id是索引，node是值，用来存储当前的树
					node = self.mcts.tree[newState.id]
					lg.logger_mcts.info('existing node...%s...', node.id)
				#Edge： inNode, outNode, prior, action
				#inNode是currentNode，outNode是采取行动后新到达的节点
				#传入了 ['P'] 值
				newEdge = mc.Edge(leaf, node, probs[idx], action)

				leaf.edges.append((action, newEdge))
				
		else:
			#如果叶子节点即为终态，则直接返回价值0或者-1
			lg.logger_mcts.info('GAME VALUE FOR %d: %f', leaf.playerTurn, value)
		#返回当前行棋序列下（棋盘）的价值
		return ((value, breadcrumbs))


	#tau是温度，使其倾向发生改变
	def getAV(self, tau):

		#注意，是众多子树的根节点
		edges = self.mcts.root.edges

		pi = np.zeros(self.action_size, dtype=np.integer)

		#values是每步棋在整体局面上的评分，是一个数组，当某个值为1时，则代表采取行动后，已经获胜
		values = np.zeros(self.action_size, dtype=np.float32)
		
		for action, edge in edges:
			#对于每条边和每个举动计算概率和价值
			#pi为行动的概率，对于每个行动的概率利用公式进行计算，总之与访问的次数成正比（指数）
			pi[action] = pow(edge.stats['N'], 1/tau)
			#价值是通过MCTS.py计算得到
			values[action] = edge.stats['Q']
		#整个概率的数组每个元素除以总的概率之和，相当于归一化操作
		pi = pi / (np.sum(pi) * 1.0)
		return pi, values

	def chooseAction(self, pi, values, tau):
		#当温度趋向于0时，选取具有最大访问次数的，也即概率最高的，这是属于训练后期了
		if tau == 0:
			#argwhere 寻找不为0的下标，这里的下标指的是action
			actions = np.argwhere(pi == max(pi))
			#若有多个相同概率的，则随机选取，并取第一个元素
			action = random.choice(actions)[0]



			# 在训练初期，为了确保多样化的选择，采取多项分布的随机选取
			# multinomial(1, pi) 1：实验次数为1  pi：每次的概率

			"""
			模拟阶段的扩展是通过随机的
			什么时候需要扩展？
			occurs when you can no longer apply UCB1
			也就是模拟到树上的某个分支时，发现下面还需要模拟，但是树上已经没有存在的节点了
			
			"""
		else:
			action_idx = np.random.multinomial(1, pi)
			#随机.大概意思是找到与某概率相对应的action的id
			action = np.where(action_idx==1)[0][0]

		#如果action直接获胜，则value返回1，否则都为0，无需神经网络参与了
		value = values[action]

		return action, value

	#利用以前游戏的记忆，训练神经网络
	def replay(self, ltmemory):
		lg.logger_mcts.info('******RETRAINING MODEL******')


		for i in range(config.TRAINING_LOOPS):
			minibatch = random.sample(ltmemory, min(config.BATCH_SIZE, len(ltmemory)))

			training_states = np.array([self.model.convertToModelInput(row['state']) for row in minibatch])
			"""
			重中之重！
			是利用已经跑过的数据，来训练更好的神经网络
			预测在前，MCTS在后
			已走过的MCTS -> 根据树上节点拿到里面的概率 ->  与神经网络预测的概率相比获得误差  ->训练网络
			"""
			training_targets = {'value_head': np.array([row['value'] for row in minibatch])
								, 'policy_head': np.array([row['AV'] for row in minibatch])} 

			fit = self.model.fit(training_states, training_targets, epochs=config.EPOCHS, verbose=1, validation_split=0, batch_size = 32)
			lg.logger_mcts.info('NEW LOSS %s', fit.history)

			self.train_overall_loss.append(round(fit.history['loss'][config.EPOCHS - 1],4))
			self.train_value_loss.append(round(fit.history['value_head_loss'][config.EPOCHS - 1],4)) 
			self.train_policy_loss.append(round(fit.history['policy_head_loss'][config.EPOCHS - 1],4)) 

		plt.plot(self.train_overall_loss, 'k')
		plt.plot(self.train_value_loss, 'k:')
		plt.plot(self.train_policy_loss, 'k--')

		plt.legend(['train_overall_loss', 'train_value_loss', 'train_policy_loss'], loc='lower left')

		display.clear_output(wait=True)
		display.display(pl.gcf())
		pl.gcf().clear()
		time.sleep(1.0)

		print('\n')
		self.model.printWeightAverages()

	def predict(self, inputToModel):
		preds = self.model.predict(inputToModel)
		return preds

	def buildMCTS(self, state):
		lg.logger_mcts.info('****** BUILDING NEW MCTS TREE FOR AGENT %s ******', self.name)
		#建立一颗新树，传入当前节点的信息到MCTS树中的根节点。当mcts不存在时，通过此方法建立root
		self.root = mc.Node(state)
		#import MCTS as mc  MCTS：return currentNode, value, done, breadcrumbs
		self.mcts = mc.MCTS(self.root, self.cpuct)

	def changeRootMCTS(self, state):
		lg.logger_mcts.info('****** CHANGING ROOT OF MCTS TREE TO %s FOR AGENT %s ******', state.id, self.name)
		#当mcts树已经存在时，通过此方法改变root值
		self.mcts.root = self.mcts.tree[state.id]