
import numpy as np
import sys

#################
### Read data ###

f = open(sys.argv[1])
data1 = np.loadtxt(f)


onearray = np.ones((data1.shape[0],1))
data1 = np.append(data1,onearray,axis=1)

train = data1[:,1:]
trainlabels = data1[:,0]

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
k=1
k=int(sys.argv[3])
##############################
### Initialize all weights ###

w = np.random.rand(hidden_nodes)
print("w=",w)

W = np.random.rand(hidden_nodes, cols)

print("W=",W)

epochs = 200
eta = 0.001
prevobj = np.inf
i=0

###########################
### Calculate objective ###

hidden_layer = np.matmul(train, np.transpose(W))
print("hidden_layer=",hidden_layer)
print("hidden_layer shape=",hidden_layer.shape)

sigmoid = lambda x: 1/(1+np.exp(-x))
hidden_layer = np.array([sigmoid(xi) for xi in hidden_layer])
print("hidden_layer=",hidden_layer)
print("hidden_layer shape=",hidden_layer.shape)

output_layer = np.matmul(hidden_layer, np.transpose(w))
print("output_layer=",output_layer)

obj = np.sum(np.square(output_layer - trainlabels))


print("Initial Obj=",obj)
data111=np.array([i for i in range(rows)])
###############################
### Begin gradient descent ####
while(i < epochs):
	
	#Update previous objective
	prevobj = obj
	#Calculate gradient update for final layer (w)
	dellw, dells, dellu, dellv=0, 0, 0, 0
	for mk in range(0, rows):
		np.random.shuffle(data111)
		cur=data111[0]
		dellw = (np.dot(hidden_layer[cur,:],w)-trainlabels[cur])*hidden_layer[cur,:]
		for j in range(1, k):
            		cur=data111[j]
            		dellw += (np.dot(hidden_layer[cur,:],np.transpose(w))-trainlabels[cur])*hidden_layer[cur,:]

	    #Update w
		w = w - eta*dellw
	
	    #Calculate gradient update for hidden layer weights (W)

	    #Let's first calculate dells. After that we do dellu and dellv.
	    #Here s, u, and v are the three hidden nodes
		cur=data111[0]
		dells = np.sum(np.dot(hidden_layer[cur,:],w)-trainlabels[cur])*w[0] * (hidden_layer[cur,0])*(1-hidden_layer[cur,0])*train[0]
		for j in range(1, k):
                	cur=data111[j]
                	dells += np.sum(np.dot(hidden_layer[cur,:],w)-trainlabels[cur])*w[0] * (hidden_layer[cur,0])*(1-hidden_layer[cur,0])*train[j]
	

		dellu = np.sum(np.dot(hidden_layer[cur,:],w)-trainlabels[cur])*w[1] * (hidden_layer[cur,1])*(1-hidden_layer[cur,1])*train[0]
		for j in range(1, k):
			cur=data111[j]
			dellu += np.sum(np.dot(hidden_layer[cur,:],w)-trainlabels[cur])*w[1] * (hidden_layer[cur,1])*(1-hidden_layer[cur,1])*train[cur]


		dellv = np.sum(np.dot(hidden_layer[cur,:],w)-trainlabels[cur])*w[2] * (hidden_layer[cur,2])*(1-hidden_layer[cur,2])*train[cur]
		for j in range(1, k):
			cur=data111[j]
			dellv += np.sum(np.dot(hidden_layer[cur,:],w)-trainlabels[cur])*w[2] * (hidden_layer[cur,2])*(1-hidden_layer[cur,2])*train[cur]


		dellW=np.array([dells, dellu, dellv])
	
	    #Update W
		W = W - eta*dellW

	#Recalculate objective
	hidden_layer = np.matmul(train, np.transpose(W))

	hidden_layer = np.array([sigmoid(xi) for xi in hidden_layer])

	output_layer = np.matmul(hidden_layer, np.transpose(w))

	obj = np.sum(np.square(output_layer - trainlabels))
	print("obj=",obj)
	
	i = i + 1
	

hidden_layer = np.matmul(test, np.transpose(W))
predictions = (np.matmul(sigmoid(hidden_layer),np.transpose(w)))
predictions1 = np.sign(predictions)
print(predictions1)
print(w)


