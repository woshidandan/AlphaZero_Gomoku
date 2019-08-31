import numpy as np
import random

import loggers as lg

from game import Game, GameState
from model import Residual_CNN

from agent import Agent, User

import config
global numchess

def playMatchesBetweenVersions(env, run_version, player1version, player2version, EPISODES, logger, turns_until_tau0, goes_first = 0):
    #-1代表的是玩家
    if player1version == -1:
        player1 = User('player1', env.state_size, env.action_size)
    else:
        #Residual_CNN 返回的是一个x
        player1_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, env.input_shape,   env.action_size, config.HIDDEN_CNN_LAYERS)
        if player1version > 0:
            #如果不是玩家，则读取训练好的版本及相关权重
            player1_network = player1_NN.read(env.name, run_version, player1version)
            player1_NN.model.set_weights(player1_network.get_weights())

        #对其进行模拟，以及mcts树的构建
        player1 = Agent('player1', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, player1_NN)

    if player2version == -1:
        player2 = User('player2', env.state_size, env.action_size)
    else:
        player2_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, env.input_shape,   env.action_size, config.HIDDEN_CNN_LAYERS)
        
        if player2version > 0:
            player2_network = player2_NN.read(env.name, run_version, player2version)
            player2_NN.model.set_weights(player2_network.get_weights())
        player2 = Agent('player2', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, player2_NN)

    scores, memory, points, sp_scores = playMatches(player1, player2, EPISODES, logger, turns_until_tau0, None, goes_first)

    return (scores, memory, points, sp_scores)

def getNumchess():
    return numchess

def playMatches(player1, player2, EPISODES, logger, turns_until_tau0, memory = None, goes_first = 0):

    env = Game()
    scores = {player1.name:0, "drawn": 0, player2.name:0}
    sp_scores = {'sp':0, "drawn": 0, 'nsp':0}
    points = {player1.name:[], player2.name:[]}

    for e in range(EPISODES):

        logger.info('====================')
        logger.info('EPISODE %d OF %d', e+1, EPISODES)
        logger.info('====================')

        # print (str(e+1) + ' ', end='')

        state = env.reset()
        
        done = 0
        turn = 0
        player1.mcts = None
        player2.mcts = None

        #判断谁先手
        if goes_first == 0:
            player1Starts = random.randint(0,1) * 2 - 1
        else:
            player1Starts = goes_first

        #日志输出
        if player1Starts == 1:
            players = {1:{"agent": player1, "name":player1.name}
                    , -1: {"agent": player2, "name":player2.name}
                    }
            logger.info(player1.name + ' plays as X')
        else:
            players = {1:{"agent": player2, "name":player2.name}
                    , -1: {"agent": player1, "name":player1.name}
                    }
            logger.info(player2.name + ' plays as X')
            logger.info('--------------')

        env.gameState.render(logger)

        while done == 0:
            #棋盘上已落子的数目，通过正反方交换的次数来确定
            turn = turn + 1
            numchess = turn
            #### Run the MCTS algo and return an action
            #turns_until_tau0步长
            if turn < turns_until_tau0:
                """
                act返回的是(action, pi, value, NN_value) NN_value是下个局面的价值 1：tau温度
                在棋盘上的实际落子数到达10之前，温度为1，更倾向于选择不同的落子方式
                当到达10之后，温度为0，则更倾向于选择概率更大（访问次数最高的）落子方式
                """
                action, pi, MCTS_value, NN_value = players[state.playerTurn]['agent'].act(state, 1)
            else:
                action, pi, MCTS_value, NN_value = players[state.playerTurn]['agent'].act(state, 0)

            if memory != None:
                ####Commit the move to memory
                memory.commit_stmemory(env.identities, state, pi)


            logger.info('action: %d', action)
            #grid_shape = (6, 7) 棋盘的维度
            for r in range(env.grid_shape[0]):
                # 打印棋盘到log日志中
                logger.info(['----' if x == 0 else '{0:.2f}'.format(np.round(x,2)) for x in pi[env.grid_shape[1]*r : (env.grid_shape[1]*r + env.grid_shape[1])]])
                #打印棋盘到输入界面
                print(['----' if x == 0 else '{0:.2f}'.format(np.round(x, 2)) for x in pi[env.grid_shape[1] * r: (env.grid_shape[1] * r + env.grid_shape[1])]])
            logger.info('MCTS perceived value for %s: %f', state.pieces[str(state.playerTurn)] ,np.round(MCTS_value,2))
            logger.info('NN perceived value for %s: %f', state.pieces[str(state.playerTurn)] ,np.round(NN_value,2))
            logger.info('====================')

            ### Do the action
            #step -> takeAction ->isEndGame -> _checkForEndGame(return 1或者0)
            #value 要不是-1要不是0，在游戏结束时，是-1
            state, value, done, _ = env.step(action) #the value of the newState from the POV of the new playerTurn i.e. -1 if the previous player played a winning move

            env.gameState.render(logger)
            #如果在采取这一步过后，赢了的话：
            if done == 1: 
                if memory != None:
                    #### If the game is finished, assign the values correctly to the game moves
                    for move in memory.stmemory:
                        """
                        为何需要判断是黑棋还是白棋？当前轮到我方下棋，但是此时判断胜负，
                        棋盘状态是来自上次对手棋盘状态的下一步，所以我方不可能胜出，因为我方胜出只会在我方结束
                        对方开始时的判断
                        也就是说，黑胜是在白棋下的时候，白胜是在黑棋下的时候
                        所以value的值应该只有-1
                        在game.py的_getValue中有这样一句话：
                        if the previous player played a winning move, you lose
                        所以当游戏结束时，取到的value只有-1或者0
                        """
                        """
                        move['value']这句代码有问题，其中的'value'在stmemory中并不存在，所以值最后谁接收了？
                        """
                        if move['playerTurn'] == state.playerTurn:
                            move['value'] = value
                        else:
                            move['value'] = -value
                         
                    memory.commit_ltmemory()

                #就当前棋手所在的视角来看，如果value为1，则说明与当前相符的playerTurn获胜，反之，则-playerTurn获胜
                #score的作用是为了判断当前模型的好坏（以先下方得分高的做为好的模型），并且是按照黑胜、白胜以及其他情况来分别统计三者的得分
                #其中sp代表了先落子的一方，nsp代表后落子的一方

                #value == 1的条件是多余的，不可能会出现
                if value == 1:
                    logger.info('%s WINS!', players[state.playerTurn]['name'])
                    scores[players[state.playerTurn]['name']] = scores[players[state.playerTurn]['name']] + 1
                    if state.playerTurn == 1: 
                        sp_scores['sp'] = sp_scores['sp'] + 1
                    else:
                        sp_scores['nsp'] = sp_scores['nsp'] + 1

                elif value == -1:
                    logger.info('%s WINS!', players[-state.playerTurn]['name'])
                    #players[-state.playerTurn]['name']输出的是player1，如果player1赢的话
                    scores[players[-state.playerTurn]['name']] = scores[players[-state.playerTurn]['name']] + 1

                    if state.playerTurn == 1: 
                        sp_scores['nsp'] = sp_scores['nsp'] + 1
                    else:
                        sp_scores['sp'] = sp_scores['sp'] + 1

                #棋盘下满了
                else:
                    logger.info('DRAW...')
                    scores['drawn'] = scores['drawn'] + 1
                    sp_scores['drawn'] = sp_scores['drawn'] + 1

                pts = state.score
                points[players[state.playerTurn]['name']].append(pts[0])
                points[players[-state.playerTurn]['name']].append(pts[1])

    return (scores, memory, points, sp_scores)
