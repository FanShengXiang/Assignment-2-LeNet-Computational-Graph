# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 21:34:51 2023

@author: AutoLab
"""

from app import *
#%%#%%
X_train, X_test = train_images/float(255), test_images/float(255)
X_val = val_images/float(255)
X_train = X_train.reshape(-1,1,28,28)
X_val = X_val.reshape(-1,1,28,28)
X_test = X_test.reshape(-1,1,28,28)

Y_train=train_labels
Y_val=val_labels
Y_test=test_labels
from sklearn.utils import shuffle
X_train, Y_train= shuffle(X_train, Y_train, random_state=0)

batch_size = 100
D_in = 784
D_out = 50

print("batch_size: " + str(batch_size) + ", D_in: " + str(D_in) + ", D_out: " + str(D_out))

### TWO LAYER NET FORWARD TEST ###
#H=400
#model = nn.TwoLayerNet(batch_size, D_in, H, D_out)
H1=300
H2=100
# model = ThreeLayerNet(batch_size, D_in, H1, H2, D_out)
model = LeNet5()

losses = []
acces = []
val_losses = []
val_acces = []
#optim = optimizer.SGD(model.get_params(), lr=0.0001, reg=0)
optim = SGDMomentum(model.get_params(), lr=0.001, momentum=0.80, reg=0.00003)
criterion = CrossEntropyLoss()

from tqdm import tqdm 
# TRAIN
ITER = 250
for i in tqdm(range(ITER)):
	# get batch, make onehot
	X_batch, Y_batch = get_batch(X_train, Y_train, batch_size)
	Y_batch = MakeOneHot(Y_batch, D_out)

	# forward, loss, backward, step
	Y_pred = model.forward(X_batch)
	loss, dout = criterion.get(Y_pred, Y_batch)
	model.backward(dout)
	optim.step()
    

	if i % 10 == 0:
		print("%s%% iter: %s, loss: %s" % (100*i/ITER,i, loss))
		result = np.argmax(Y_pred, axis=1) - np.argmax(Y_batch, axis=1)
		result = list(result)
		acc = result.count(0)/X_batch.shape[0]
		Y_pred_val = model.forward(X_val)
		result_val = np.argmax(Y_pred_val, axis=1) - Y_val
		result_val = list(result)
		val_acc = result_val.count(0)/X_val.shape[0]
		losses.append(loss)
		acces.append(acc)
		val_acces.append(val_acc)

# save params
weights = model.get_params()
with open("weights.pkl","wb") as f:
	pickle.dump(weights, f)

draw_losses(losses)
draw_losses(acces)


# TRAIN SET ACC
Y_pred = model.forward(X_train)
result = np.argmax(Y_pred, axis=1) - Y_train
result = list(result)
print("TRAIN--> Correct: " + str(result.count(0)) + " out of " + str(X_train.shape[0]) + ", acc=" + str(result.count(0)/X_train.shape[0]))

# TEST SET ACC
Y_pred = model.forward(X_val)
result = np.argmax(Y_pred, axis=1) - Y_val
result = list(result)
print("VAL--> Correct: " + str(result.count(0)) + " out of " + str(X_val.shape[0]) + ", acc=" + str(result.count(0)/X_val.shape[0]))


# TEST SET ACC
Y_pred = model.forward(X_test)
result = np.argmax(Y_pred, axis=1) - Y_test
result = list(result)
print("TEST--> Correct: " + str(result.count(0)) + " out of " + str(X_test.shape[0]) + ", acc=" + str(result.count(0)/X_test.shape[0]))
