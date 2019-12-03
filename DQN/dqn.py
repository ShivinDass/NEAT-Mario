import torch
import torch.nn as nn
import numpy as np


#learningRate=0
x=120
y=128


class DQN(nn.Module):

	def __init__(self,learningRate):
		super(DQN,self).__init__()

		self.layerC1=nn.Conv2d(4,32,8,4)
		self.layerC2=nn.Conv2d(32,64,4,2)
		self.layerC3=nn.Conv2d(64,64,3,1)

		self.layerF1=nn.Linear(64*11*12,512)
		self.layerF2=nn.Linear(512,12)

		self.opt=torch.optim.RMSprop(self.parameters(), lr=learningRate)

		self.lossFunction=nn.MSELoss()
		#self.lossFunction=nn.CrossEntropyLoss()
		s='cpu'
		if torch.cuda.is_available():
			s='cuda:0'
		self.dev=torch.device(s)
		self.to(self.dev)


	def forward(self,inp):
		inp=torch.Tensor(inp).to(self.dev)
		inp=inp.view(-1,4,x,y)
		#print(inp.shape)
		inp=nn.functional.relu(self.layerC1(inp))
		inp=nn.functional.relu(self.layerC2(inp))
		inp=nn.functional.relu(self.layerC3(inp))
		inp=inp.view(-1,64*11*12)
		inp=nn.functional.relu(self.layerF1(inp))
		return self.layerF2(inp)


