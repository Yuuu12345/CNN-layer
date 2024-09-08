import numpy as np
from function import AdaGrad,im2col,col2im

class CNN:
  def __init__(self,filter,stride=1,pad=0):

    self.FN,self.C,self.FH,self.FW=filter
    self.S=stride
    self.P=pad

    self.F=np.random.uniform(low=-0.1,high=0.1,size=(self.FN,self.C,self.FH,self.FW))
    self.b=np.zeros((1,self.FN))
    self.x=None
    self.col=None

    self.AdaGradW=AdaGrad(0.01)
    self.AdaGradb=AdaGrad(0.01)
    self.AdaGradX=AdaGrad(0.01)

  def forward(self,x):
    self.N,self.C,self.H,self.W=x.shape
    OH=int(((self.H+2*self.P-self.FH)/self.S)+1)
    OW=int(((self.W+2*self.P-self.FW)/self.S)+1)

    self.col=im2col(x,self.FH,self.FW,self.S,self.P)
    filter=self.F.flatten().reshape(-1,self.FN)#(C・F,FN)

    self.x=self.col
    self.F=filter

    y=np.dot(self.col,filter)+self.b

    return y.reshape(1,self.FN,OH,OW)#(FN,OH,OW)

  def backward(self,dout,lr=0.01):

    dout=dout.reshape(-1,self.FN)

    dF=np.dot(self.x.T,dout)
    dx=np.dot(dout,self.F.T)
    dout_b=np.sum(dout,axis=0)
    db=dout_b

    self.F=self.AdaGradW.update(self.F,dF)
    self.x=self.AdaGradX.update(self.col,dx)

    # self.F-=lr*dF
    # self.x-=lr*dx
    # self.b-=lr*db

    self.dx=col2im(self.x,(self.N,self.C,self.H,self.W),self.FH,self.FW,self.S,self.P)

    return self.dx