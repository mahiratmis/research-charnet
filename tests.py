import torch
import numpy as np
x = np.array([[[1,2],[3,4],[5,6],[7,8]],
              [[10,20],[30,40],[50,60],[70,80]],
              [[15,25],[35,45],[55,65],[75,80]]])
print(x.shape)
print(x[0,0], type(x[0,0,]), tuple(x[0,0,]), type(tuple(x[0,0])))

a = torch.from_numpy(x).unsqueeze(dim=0)
print(a)
print(a.min(dim=1))
print(a.min(dim=2))
print(a.min(dim=3))
y = x.reshape(-1,8)
print(y, y.shape)

z= np.array([1,2,3,4,5,6,7,8])
x1,x2,x3,x4,x5,x6,x7,x8 = z
print(x1,x2,x3,x4,x5,x6,x7,x8)
print(z.shape)

a = x[0]
b = np.array([3,5])
print(b*a)

c= a.reshape(8)
print("c ", c, c.shape)


k = np.array([[1,2,3,4,5,6,7,8],
              [11,21,31,41,51,61,71,81],
              [12,22,32,43,53,62,72,82,]])

print(k)
m = k[:2]
n = k[1:]
m[0][0] = -2
n[0,1] =  -1
print(k[:2])
print(k[1:])

for cbbox in k:
    x = cbbox[::2]
    y = cbbox[1::2]
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    print(x,y,xmin, xmax,ymin, ymax)


hw = np.array([50,45])

x = torch.Tensor([[1,2,3],[4,5,6]])
print(x[0,1])
print(x[0][1])

from shapely.geometry import Polygon
p1 = Polygon([(0,0), (10,0), (10,8), (0,8)])
p2 = Polygon([(2,2), (4,2), (4,5), (2,5)])
inter = p1.intersection(p2)
bbox = list(zip(*inter.exterior.coords.xy))
print(bbox[:-1])

empty_array = np.empty(0)
print(empty_array.size)

x = [[1, 2, 3], [4, 5, 6]]
y = [[ch+1 for ch in w] for w in x]
print(y)

r1 = torch.zeros((1,1,4))
r2 = torch.ones((1,1,1))
r3 = torch.cat((r1,r2), dim=2)
print(r3)

# colors = [(1,2,3),(4,5,6),(7,8,9)]
# import matplotlib.pyplot as plt
# import seaborn as sns; sns.set()
# data = torch.randn((128,128))
# fig, ax = plt.subplots(figsize=(50,50))  
# ax = sns.heatmap(data.numpy(), annot=True, ax=ax) 
# fig.savefig('class_conf.png', dpi=200)