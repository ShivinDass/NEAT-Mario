import torch as t
import torch.nn as nn
import numpy as np
import _pickle as cPickle
import random
from copy import deepcopy
from dqn import DQN
import collections

class mario():
	def __init__(self, gamma, learningRate, minEps = 0.01, memorySize = 1000, maxEps = 1, actionSpace = list(range(0, 12)), epsDecayRate = 0.001):
		self.gamma = gamma
		self.Eps = maxEps
		self.minEps = minEps
		self.learningRate = learningRate
		self.actionSpace = actionSpace
		self.memorySize = memorySize
		#self.memoryCounter = 0
		self.Q_eval = DQN(learningRate)
		# self.Q_next = DQN(learningRate)
		self.recallMemory = collections.deque(maxlen=self.memorySize)
		self.steps = 0
		self.learnStepCounter = 0
		self.epsDecayRate = epsDecayRate
		self.intelligent=False
		#for i in range(memorySize):
		#	self.recallMemory.append((0, 0, 0, 0))


	def addToMemory(self, currState, action, reward, nextState, done):
		self.recallMemory.append((currState, action, reward, nextState, done))
		'''self.recallMemory[self.memoryCounter] = (currState, action, reward, nextState)
		self.memoryCounter = (self.memoryCounter + 1) % self.memorySize'''


	# def makeMove(self, state):
	# 	moveProbability = self.Q_eval.forward(state)
	# 	if(random.uniform(0, 1) < self.Eps):
	# 		move = np.random.choice(self.actionSpace)
	# 	else:
	# 		#using the middle frame, we're using 3
	# 		move = t.argmax(moveProbability[2]).item()
	# 	self.steps += 1
	# 	return move
	def makeMove(self, state):
		moveProbability = self.Q_eval.forward(state)
		#print("MoveProb",moveProbability.shape)
		if(random.uniform(0, 1) < self.Eps):
			move = np.random.choice(self.actionSpace)
		else:
			print(moveProbability)
			print(t.argmax(moveProbability[0]))
			print()
			move = t.argmax(moveProbability[0]).cpu().numpy()
		self.steps += 1
		return move

	def makeMoveIntelligent(self, state, steps):
		moveProbability = self.Q_eval.forward(state)
		#print("MoveProb",moveProbability.shape)
		if(steps>300 and random.random()<0.5):
			move = np.random.choice(self.actionSpace)
		else:
			#print(moveProbability)
			#print(t.argmax(moveProbability[0]))
			#print()
			move = t.argmax(moveProbability[0]).cpu().numpy()
		self.steps += 1
		return move


	def getBatchFromMemory(self, batchSize):
		return random.sample(self.recallMemory,min(batchSize,len(self.recallMemory)))
		'''total = self.memSize - self.memoryCounter
		if(total < batchSize):
			assert(len(self.recallMemory[memoryCounter:] + self.recallMemory[:total-batchSize]) == batchSize)
			return np.array(self.recallMemory[memoryCounter:] + self.recallMemory[:total-batchSize])
		else:
			assert(total == batchSize)
			return np.array(self.recallMemory[memoryCounter:memoryCounter + batchSize])'''

	def updateEps(self):
		if(self.steps >= 500):
			self.Eps = max(self.minEps, self.Eps - 0.0001)
		# self.Eps = self.minEps + (1 - self.minEps) * np.exp(-self.epsDecayRate * self.steps)

	# def train(self, batchSize):
	# 	self.Q_eval.optimizer.zero()

	# 	batch = getBatchFromMemory(self, batchSize)

	# 	Qvaluepredicted = self.Q_eval.forward(list(batch[:, 0][:])).to(self.Q_eval.device)
	# 	Qnextvaluepredicted = self.Q_eval.forward(list(batch[:, 3][:])).to(self.Q_eval.device)

	# 	#dim = 1 because middle frame?
	# 	bestAction = t.argmax(Qnextvaluepredicted, dim = 1).to(self.Q_eval.device)
	# 	rewards = t.Tensor(list(batch[:, 2])).to(self.Q_eval.device)

	# 	Qtarget = Qvaluepredicted
	# 	Qtarget[:, bestAction] = rewards + self.gamma*t.max(Qnextvaluepredicted[1]) #understand

	# 	updateEps()

	# 	loss = self.Q_eval.loss(Qtarget, Qvaluepredicted).to(self.Q_eval.device)
	# 	loss.backward()
	# 	self.Q_eval.optimizer.step()
	# 	self.learnStepCounter += 1
	def train(self, batchSize):
		self.Q_eval.opt.zero_grad()

		batch = np.array(self.getBatchFromMemory(batchSize))

		Qvaluepredicted = self.Q_eval.forward(list(batch[:, 0][:])).to(self.Q_eval.dev)
		Qnextvaluepredicted = self.Q_eval.forward(list(batch[:, 3][:])).to(self.Q_eval.dev)

		print(Qvaluepredicted)
		print(Qnextvaluepredicted)
		#dim = 1 because middle frame?
		bestAction = t.argmax(Qnextvaluepredicted, dim = 1).to(self.Q_eval.dev)
		rewards = t.Tensor(list(batch[:, 2])).to(self.Q_eval.dev)

		#print(bestAction)
		#print(rewards)

		#Qtarget = deepcopy(Qvaluepredicted.cpu().numpy())
		#Qtarget=t.Tensor(Qtarget)
		Qtarget=t.empty(len(Qvaluepredicted),len(Qvaluepredicted[0]))
		for i in range(len(Qvaluepredicted)):
			for j in range(len(Qvaluepredicted[i])):
				Qtarget[i,j]=Qvaluepredicted[i,j]
		Qtarget[:, bestAction] = rewards + self.gamma*t.max(Qnextvaluepredicted[:]) #understand

		#print("Targ",Qtarget[:,bestAction])
		#print("Val Pred",Qvaluepredicted[:,bestAction])
		#print(Qtarget-Qvaluepredicted)

		self.updateEps()

		loss = self.Q_eval.lossFunction(Qvaluepredicted, Qtarget).to(self.Q_eval.dev)
		#print(loss)
		loss.backward()
		self.Q_eval.opt.step()
		self.learnStepCounter += 1


	def save(self, agentNum):
		#pickle_out = open("savedAgent/agent"+str(agentNum)+".ag", "wb+")
		pickle_out = open("savedAgent/agent.ag", "wb+")
		cPickle.dump(self, pickle_out)

	def load(self, agentNum):
		#pickle_in = open("savedAgent/agent"+str(agentNum)+".ag", "rb")
		pickle_in = open("savedAgent/agent.ag", "rb")
		other = cPickle.load(pickle_in)
		self.gamma = deepcopy(other.gamma)
		self.Eps = deepcopy(other.Eps)
		self.minEps = deepcopy(other.minEps)
		self.learningRate = deepcopy(other.learningRate)
		self.actionSpace = deepcopy(other.actionSpace)
		self.memorySize = deepcopy(other.memorySize)
		# self.memoryCounter = deepcopy(other.memoryCounter)
		self.Q_eval = deepcopy(other.Q_eval)
		# self.Q_next = deepcopy(other.Q_next)
		self.recallMemory = deepcopy(other.recallMemory)
		self.steps = deepcopy(other.steps)
		self.learnStepCounter = deepcopy(other.learnStepCounter)
		self.epsDecayRate = deepcopy(other.epsDecayRate)
		self.intelligent=deepcopy(other.intelligent)




