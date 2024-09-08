from layer.Affine import Affine
from layer.CNN import CNN
from layer.Pooling import Pooling
from function import *

class LeNet:
  def __init__(self,lr):
    self.CNN1=CNN((6,1,3,3),1,2)
    self.Pool1=Pooling(2,2,1)
    self.Affine1=Affine(1536,50)
    self.Affine2=Affine(50,10)

    self.Relu=None

    self.lr=lr

  def __call__(self,x):

    h=self.CNN1.forward(x)
    h=self.Pool1.forward(h)
    h=self.Affine1.forward(h)
    h=Relu(h)
    self.Relu=h
    h=self.Affine2.forward(h)
    h=softmax(h)

    return h

  def backward(self,dout):

    dout=self.Affine2.backward(dout,self.lr)
    dout=Relu_backward(dout,self.Relu)
    dout=self.Affine1.backward(dout,self.lr)
    dout=self.Pool1.backward(dout)
    dout=self.CNN1.backward(dout,self.lr)