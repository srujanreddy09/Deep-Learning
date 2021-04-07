import numpy as np
import sys

#################
### Read data ###

f = open(sys.argv[1])
data = np.loadtxt(f)
train = data[:,1:]
trainlabels = data[:,0]

onearray = np.ones((train.shape[0],1))
train = np.append(train,onearray,axis=1)

print("train=",train)
print("train shape=",train.shape)

f = open(sys.argv[2])
data = np.loadtxt(f)
test = data[:,1:]
testlabels = data[:,0]

onearray = np.ones((test.shape[0],1))
test = np.append(test,onearray,axis=1)

rows = train.shape[0]
cols = train.shape[1]

hidden_nodes = 3

##############################
### Initialize all weights ###

w = np.random.rand(hidden_nodes)
print("initial w=",w)


W = np.random.rand(hidden_nodes, cols)

print("initial W=",W)

epochs = 1000
eta = .01
prevobj = np.inf
i=0

###########################
### Calculate objective ###

hidden_layer = np.matmul(train, np.transpose(W))
print("hidden_layer=",hidden_layer)
print("hidden_layer shape=",hidden_layer.shape)

sigmoid = lambda x: 1/(1+np.exp(-x))

hidden_layer = np.array([sigmoid(xi) for xi in hidden_layer])
print("activated hidden_layer=",hidden_layer)
print("activated hidden_layer shape=",hidden_layer.shape)

output_layer = np.matmul(hidden_layer, np.transpose(w))
obj = np.sum(np.square(output_layer - trainlabels))


print("Initial Obj=",obj)

###############################
### Begin gradient descent ####
#exit()
while((prevobj - obj) > 0.0000001 or i < epochs):
#while(prevobj - obj > 0):
	
	#Update previous objective
	prevobj = obj

	#Calculate gradient update for final layer (w)
	dellw = (np.dot(hidden_layer[0,:],w)-trainlabels[0])*hidden_layer[0,:]
	for j in range(1, rows):
		dellw += (np.dot(hidden_layer[j,:],np.transpose(w))-trainlabels[j])*hidden_layer[j,:]

	#Update w
	w = w - eta*dellw
	
	#Calculate gradient update for hidden layer weights (W)
	
	dells = np.sum(np.dot(hidden_layer[0,:],w)-trainlabels[0])*w[0] * (hidden_layer[0,0])*(1-hidden_layer[0,0])*train[0]
	for j in range(1, rows):
		dells += np.sum(np.dot(hidden_layer[j,:],w)-trainlabels[j])*w[0] * (hidden_layer[j,0])*(1-hidden_layer[j,0])*train[j]
	

	dellu = np.sum(np.dot(hidden_layer[0,:],w)-trainlabels[0])*w[1] * (hidden_layer[0,1])*(1-hidden_layer[0,1])*train[0]
	for j in range(1, rows):
		dellu += np.sum(np.dot(hidden_layer[j,:],w)-trainlabels[j])*w[1] * (hidden_layer[j,1])*(1-hidden_layer[j,1])*train[j]


	dellv = np.sum(np.dot(hidden_layer[0,:],w)-trainlabels[0])*w[2] * (hidden_layer[0,2])*(1-hidden_layer[0,2])*train[0]
	for j in range(1, rows):
		dellv += np.sum(np.dot(hidden_layer[j,:],w)-trainlabels[j])*w[2] * (hidden_layer[j,2])*(1-hidden_layer[j,2])*train[j]


	dellW=np.array([dells, dellu, dellv])
	#print(dellW)
	
	#Update W
	W = W - eta*dellW
	
	#Recalculate objective
	hidden_layer = np.matmul(train, np.transpose(W))

	hidden_layer = np.array([sigmoid(xi) for xi in hidden_layer])

	output_layer = np.matmul(hidden_layer, np.transpose(w))

	obj = np.sum(np.square(output_layer - trainlabels))
	print("obj=",obj)
	
	i = i + 1
	

print("w:",w)
hidden_layer = np.matmul(test, np.transpose(W))
predictions = (np.matmul(sigmoid(hidden_layer),np.transpose(w)))
predictions1 = np.sign(predictions)
print("Predictions:",predictions1)

