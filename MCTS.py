import numpy as np
import logging
import config

from utils import setup_logger
import loggers as lg

class Node():

	def __init__(self, state):
		self.state = state
		self.playerTurn = state.playerTurn
		#state id是当前棋盘状态下，除去白棋棋盘上1和0的黑棋序列以及出去黑棋棋盘上1和0的白棋序列组成的
		self.id = state.id
		self.edges = []

	def isLeaf(self):
		#判断是否是叶子节点（边缘）
		if len(self.edges) > 0:
			return False
		else:
			return True

class Edge():

	def __init__(self, inNode, outNode, prior, action):
		self.id = inNode.state.id + '|' + outNode.state.id
		self.inNode = inNode  #一条边的入节点
		self.outNode = outNode #一条边的出节点
		self.playerTurn = inNode.state.playerTurn
		self.action = action

		self.stats =  {
					'N': 0,   #节点访问的次数 visit count
					'W': 0,   #总的行动的价值 total action-value
					'Q': 0,   #平均行动的价值 mean action-value
					'P': prior, #先验概率 prior probability，从
				}
				

class MCTS():

	def __init__(self, root, cpuct):
		self.root = root
		self.tree = {}
		self.cpuct = cpuct    #探索水平
		self.addNode(root)
	
	def __len__(self):
		return len(self.tree)

	#相当于Select MCTS树上已存在节点（具有先验概率的），直到叶子节点为止的过程
	def moveToLeaf(self):

		lg.logger_mcts.info('------MOVING TO LEAF------')
		#用于神经网络评估的队列，相当于一个行棋的序列
		breadcrumbs = []
		currentNode = self.root

		done = 0
		value = 0
		#如果当前的节点不是叶子节点，一直将currentNode判断，直到currentNode是叶子节点
		while not currentNode.isLeaf():

			lg.logger_mcts.info('PLAYER TURN...%d', currentNode.state.playerTurn)
		
			maxQU = -99999

			if currentNode == self.root:
				epsilon = config.EPSILON
				#引入狄利克雷噪声，干扰（在模拟树上，对已存在的节点，利用先验概率进行选择时），使其尝试别的节点
				nu = np.random.dirichlet([config.ALPHA] * len(currentNode.edges))
			else:
				epsilon = 0
				nu = [0] * len(currentNode.edges)

			Nb = 0
			#之所以这么算，是没有当前节点（没有自身的）的直接结果，比如当前节点为3次，他下面的点可能是2次，1次
			#因此需要累加得到1+2 = 3次，也可以直接得到3
			for action, edge in currentNode.edges:
				Nb = Nb + edge.stats['N']

			# 当currentNode是叶子节点时，那么action以及edge即为在叶子节点时的操作和边了
			for idx, (action, edge) in enumerate(currentNode.edges):
				# 注意：此时的"\"不是整数除法计算后的整数部分，而是Return value*self. "/"是除号,是普通除法计算。
				# ((1-epsilon) * edge.stats['P'] + epsilon * nu[idx] )是为了引入噪声，确保所有位置被尝试
				# np.sqrt(Nb) / (1 + edge.stats['N']) 两种步数，Nb为之前所有节点总共走的次数，edge.stats['N']为当前动作已经尝试的总次数
				# U为对尚未充分探索过的节点继续探索的必要性
				# edge.stats['P']只用在叶子节点时，参与选择下一步的候选节点，从神经网络得到，只用一次，不参与之后树上已存在节点的选择
				# 用于初始化叶子节点，但是不用于实际选择时
				#
				'''
				在模拟过程中的一直在用edge.stats['P']的
				edge.stats['P']是通过之前模拟时的神经网络的预测得到，但是存在了节点中，后面再模拟时，是可以利用的。
				到达叶子节点时，这种edge.stats['P']已经不存在了（指的是叶子节点的下个节点的概率），因此需要evaluateLeaf，并且存储
				但是，后面实际落子时，用到的并不是这个P，而是一个pai，这个pai只与访问次数和温度有关，并且只会用到一次，选一个子
				'''
				U = self.cpuct * \
					((1-epsilon) * edge.stats['P'] + epsilon * nu[idx] )  * \
					np.sqrt(Nb) / (1 + edge.stats['N'])
					
				Q = edge.stats['Q']

				lg.logger_mcts.info('action: %d (%d)... N = %d, P = %f, nu = %f, adjP = %f, W = %f, Q = %f, U = %f, Q+U = %f'
					, action, action % 7, edge.stats['N'], np.round(edge.stats['P'],6), np.round(nu[idx],6), ((1-epsilon) * edge.stats['P'] + epsilon * nu[idx] )
					, np.round(edge.stats['W'],6), np.round(Q,6), np.round(U,6), np.round(Q+U,6))

				# 如果当前的平均价值加探索的必要性大于之前存储的，表明此落棋点会更好，因此采取此行动
				if Q + U > maxQU:
					maxQU = Q + U
					simulationAction = action
					simulationEdge = edge

			lg.logger_mcts.info('action with highest Q + U...%d', simulationAction)
			#采取模拟的这一步之后所得到的新的棋盘状态，价值，以及是否结束，value要不是0要不是-1
			#在叶子节点currentNode时采取simulationAction = action操作，到达叶子节点takeAction扩展的下个节点

			newState, value, done = currentNode.state.takeAction(simulationAction) #the value of the newState from the POV of the new playerTurn
			#simulationEdge.outNode取得出度，也就是新的叶子节点currentNode
			currentNode = simulationEdge.outNode
			# The leaf node is added to a queue for neural network evaluation
			#breadcrumbs一个一直延伸到叶子节点的序列
			breadcrumbs.append(simulationEdge)

		lg.logger_mcts.info('DONE...%d', done)

		#currentNode是最新的叶子节点
		return currentNode, value, done, breadcrumbs


	#
	def backFill(self, leaf, value, breadcrumbs):
		lg.logger_mcts.info('------DOING BACKFILL------')

		currentPlayer = leaf.state.playerTurn

		# breadcrumbs一个一直延伸到叶子节点的序列
		for edge in breadcrumbs:
			playerTurn = edge.playerTurn
			if playerTurn == currentPlayer:
				direction = 1
			else:
				direction = -1

			#Backup
			edge.stats['N'] = edge.stats['N'] + 1  #访问次数增加


			"""########################################################################
			###这里面最关键的value是通过神经网络计算得来的 -- rollout行动价值计算###
			核心问题是：为何当结果出来时的那次back，path上的所有W的值会变成整数，以及如何叠加的？
			之所以会有这个问题，是因为：
			（1）value由value network而来，而其预测出来的行动的价值，并不总是整数（-1，1或者0以及其他）
			也就是小数的情况为何会最后变成整数
			（2）由edge.stats['W'] / edge.stats['N']中可以看出，分子也是为整数
			
			原因见笔记《UCB&&UCT.docx》
			"""#########################################################################
			edge.stats['W'] = edge.stats['W'] + value * direction

			# 整体盘面平均价值（为1时获胜），都是相对于处于终态时的单个节点而言
			#edge.stats['Q']：1或者-1或者其他，如：[[0.02854559]]，[[-0.02830142]]。。。。。。
			edge.stats['Q'] = edge.stats['W'] / edge.stats['N']

			lg.logger_mcts.info('updating edge with value %f for player %d... N = %d, W = %f, Q = %f'
				, value * direction
				, playerTurn
				, edge.stats['N']
				, edge.stats['W']
				, edge.stats['Q']
				)
			#BackFill没有返回值，而是储存在边中
			edge.outNode.state.render(lg.logger_mcts)

	def addNode(self, node):
		self.tree[node.id] = node

