from netCDF4 import Dataset
import numpy as np
import random 
import matplotlib.pyplot as plt

random.seed(3) 

# np.random.randint(5, size=(2, 4))
data = np.random.randint(100, size=(50,100,100,7))

def generat_sample(mean, var):
	x = np.random.multivariate_normal(mean, var, 100)
	x = x.astype(int)
	x = np.unique(x,axis=0)
	return x

def dataset_generator():
	out = []
	num_cluster = random.randint(1, 4)
	for i in range(num_cluster):
		t1 = random.randint(0, data.shape[0])
		t2 = random.randint(t1, data.shape[0])
		num_exp_var = random.randint(0,data.shape[3])
		index_exp_var = np.random.randint(data.shape[3], size=num_exp_var)
		x = random.randint(0,data.shape[1])
		y = random.randint(0,data.shape[2])
		mean = [x,y]
		var = np.identity(2, dtype = float)
		coordinate = generat_sample(mean,var)
		# x = coordinate[:,0]
		# y = coordinate[:,1]
		# plt.plot(x,y, 'x')
		# plt.axis('equal')
		# plt.show()
		# print ("*************\n")
		out.append([t1, t2, index_exp_var, coordinate])
	return out

def generate_mask(inp):
	out = np.zeros(data.shape)
	for i in inp:
		t1, t2, index_exp_var, coordinate = i
		for t in range(data.shape[0]):
			if (t < t1 or t > t2):
				continue
			for x in range(data.shape[1]):
				if x not in coordinate[0]:
					continue
				for y in range(data.shape[2]):
					if y not in coordinate[1]:
						continue
					for z in range(data.shape[3]):
						if z in index_exp_var:
							out[t,x,y,z] = 1
	return out


out = dataset_generator()
mask = generate_mask(out)
print (mask)
# print (np.where(mask == 1))