import numpy as np
from function import im2col,col2im

class Pooling:
  def __init__(self,PL_size,stride=1,padding=0):
    self.size=PL_size
    self.S=stride
    self.P=padding

    self.argmax=None
    self.col=None

    self.N=None
    self.C=None
    self.H=None
    self.W=None

  def forward(self,x):
    self.N,self.C,self.H,self.W=x.shape
    OH=int(((self.H+2*self.P-self.size)/self.S)+1)
    OW=int(((self.W+2*self.P-self.size)/self.S)+1)

    col=im2col(x,self.size,self.size,stride=self.S,pad=self.P)

    #colの行にはCも含まれてるため、Cを列方向に移す作業が必要
    col=col.reshape(-1,self.size*self.size)
    self.col=col

    self.argmax=np.argmax(col,axis=1)
    max=np.max(col,axis=1)
    y=max.reshape(self.N,self.C,OH,OW)
    return y

  def backward(self,dout):
    dout=dout.flatten()
    W_=np.zeros(self.col.shape)
    W_[np.arange(len(dout)),self.argmax]=dout

    W_=W_.reshape(-1,self.size**2*self.C)
    dx=col2im(W_,(self.N,self.C,self.H,self.W),self.size,self.size,self.S,self.P)
    
    return dx