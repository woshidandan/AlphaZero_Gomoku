#### SELF PLAY
EPISODES = 30
MCTS_SIMS = 50  #每局棋蒙特卡罗搜索树模拟的数目，表示到达多少次模拟时，如果没出结果，强制返回
MEMORY_SIZE = 30000

# turn on which it starts playing deterministically 是改变温度参数的一个阈值，象征了棋盘上已经落子的个数
#落子数到达此值，便会改变参数此值猜测可能会影响训练的收敛速度
TURNS_UNTIL_TAU0 = 10

"""
蒙特卡罗搜索树探索的等级，探索水平，修改此值可能会对额外的地区进行探索，论文中是设置成5
在使用此套代码时，默认值是1，但是在训练过程中，导致其对棋盘上某些关键点的探索没有进行，掌控的不好，猜测是此值造成的
"""
CPUCT = 1

EPSILON = 0.2 #噪声对先验概率的干扰比例
ALPHA = 0.8


#### RETRAINING
BATCH_SIZE = 256
EPOCHS = 1   #训练的轮数
REG_CONST = 0.0001  #规则化
LEARNING_RATE = 0.1  #学习速率
MOMENTUM = 0.9
TRAINING_LOOPS = 10

#总共有6个残差层（根据图上画的是有6个，但是在构建的时候，第一个不是残差层，而是正常层，总共有5个残差层），每个残差层里的卷积核如下：
HIDDEN_CNN_LAYERS = [
	{'filters':75, 'kernel_size': (4,4)}
	 , {'filters':75, 'kernel_size': (4,4)}
	 , {'filters':75, 'kernel_size': (4,4)}
	 , {'filters':75, 'kernel_size': (4,4)}
	 , {'filters':75, 'kernel_size': (4,4)}
	 , {'filters':75, 'kernel_size': (4,4)}
	]

#### EVALUATION
EVAL_EPISODES = 20
SCORING_THRESHOLD = 1.3