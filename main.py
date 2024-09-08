import numpy as np
from dataset import preprocess_data
from model import LeNet


train_x, train_y, test_x, test_y = preprocess_data(60000,10000)
model=LeNet(0.01)
epock=5000

# 学習
for i in range(epock):

  t=train_y[i]
  y=model(train_x[i].reshape(1,1,28,28))

  loss=y-t
  model.backward(loss)
  
  
# 推論
test_num=np.random.choice(np.arange(10000),1001,replace=False)
count=0

for i in test_num:
  t=test_y[i]
  y=model(test_x[i].reshape(1,1,28,28))
  if np.argmax(y)==np.argmax(t):
    count+=1
print((count/len(test_num))*100)