import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def PCA(data, correlation=False, sort=True):
    # 作业1
    # 屏蔽开始
    if correlation:
        corr_data = np.corrcoef(data.T)
        eigenvectors, eigenvalues, _ = np.linalg.svd(corr_data)        
        print("using correlation")
        
    else:
        cov_data = np.cov(data.T)
        eigenvectors, eigenvalues, _ = np.linalg.svd(cov_data)
        print("using cov")
        

    eigenvalues = np.sqrt(eigenvalues)
    print("eigenvalues", eigenvalues)
    print("eigenvectors")
    print(eigenvectors)
    # 屏蔽结束

    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors


# generate data of normal distribution N(0,1) 
center = [0.0, 0.0]
cov = np.array([[100.0,100.0],[0.0,20.0]])
data = np.random.multivariate_normal(center,cov,10000)

# draw ellipse
eigenvalues, eigenvectors = np.linalg.eig(cov)
print("@@@",eigenvectors)
sort = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[sort]
eigenvectors = eigenvectors[:, sort]
angle = - np.arccos(np.dot(eigenvectors[:,1],np.array([1,0]))) * 180 / np.pi
print(angle)
ellipse = Ellipse(center,eigenvalues[0], eigenvalues[1], angle,facecolor= 'yellow', alpha=0.3)

data - np.mean(data, axis = 0)

# draw data
fig = plt.figure()
plt.scatter(data[:,0], data[:,1],color = [0,1,0])
plt.axis('equal')
ax = fig.add_subplot(111)
ax.add_patch(ellipse)
ax.set_title('Data')   
plt.xlabel('X1') #设置X轴标签 
plt.ylabel('X2') #设置Y轴标签  


w,v = PCA(data,False)
plt.scatter(center[0], center[1], marker = 'x', color = 'r', s = 30)

tmp_p1 = center + v[:,0] * w[0] 
tmp_p2 = center + v[:,1] * w[1] 

x1, y1 = [center[0], tmp_p1[0]], [center[1], tmp_p1[1]]
x2, y2 = [center[0], tmp_p2[0]], [center[1], tmp_p2[1]]
plt.plot(x1,y1,x2,y2, color = 'r')

plt.show()