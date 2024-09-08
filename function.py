import numpy as np

# def softmax(x):
#   exp_x = np.exp(x - np.max(x,axis=1,keepdims=True))
#   return exp_x/np.sum(exp_x,axis=1,keepdims=True)

def softmax(x):
  exp_x = np.exp(x - np.max(x))
  return exp_x/np.sum(exp_x)

def xavier(inn,out):
  limit=np.sqrt(6/(inn+out))
  return np.random.uniform(-limit,limit,size=(inn,out))

def Relu(x):
  return np.maximum(x,0)

def Relu_backward(dout,x):
  return dout*(x>0)

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):

    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

def output_size(H,W,FH,FW,S=1,P=0):
  OH=int(((H+2*P-FH)/S)+1)
  OW=int(((W+2*P-FW)/S)+1)
  return OH,OW

class AdaGrad:
  def __init__(self,lr):
    self.h=None
    self.lr=lr

  def update(self,w,dout):
    if self.h is None:
      self.h=np.zeros_like(dout)
    self.h=self.h+(dout*dout)
    w=w-self.lr*(1/np.sqrt(self.h+1e-7))*dout
    return w