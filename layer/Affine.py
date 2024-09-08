import numpy as np
from function import AdaGrad

class Affine:
  def __init__(self,input,output):
    self.input=input
    self.output=output

    self.W=np.random.uniform(low=-0.1,high=0.1,size=(self.output,self.input))
    self.b=np.zeros(output)

    self.x=None
    self.dW=None
    self.db=None

    self.AdaGradW=AdaGrad(0.01)
    self.AdaGradb=AdaGrad(0.01)

  def forward(self,x):
    self.x=x.flatten()

    return np.dot(self.W,self.x)+self.b

  def backward(self,dout,lr=0.1):
    self.dW=np.dot(dout.reshape(-1,1),self.x.reshape(1,-1))
    self.db=dout
    dx=np.dot(self.W.T,dout)

    self.W=self.AdaGradW.update(self.W,self.dW)
    self.b=self.AdaGradb.update(self.b,self.db)

    # self.W-=lr*self.dW
    # self.b-=lr*self.db

    return dx