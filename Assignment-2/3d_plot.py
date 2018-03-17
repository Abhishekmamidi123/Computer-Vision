from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

####### Kmeans and KNN #############

fig1 = plt.figure("Kmeans and KNN accuracies")
ax = fig1.gca(projection='3d')


zs = (29.25,35.125,39.5,46.75,47,47.625,50.375)
ys = (2,4,8,16,16,32,64)
xs = (5,5,5,8,5,5,5)


for x, y, z in zip(xs, ys, zs):
    label = '(%d, %d, %d)' % (x, y, z,)
    ax.scatter3D(x, y, z, cmap='Greens');
    #ax.text(x, y, z, (str(x)+" "+str(y)+" "+str(z)), color='red')

ax.text2D(0.05, 0.95, "(KNN, Kmeans, Accuracy) Plot",color="Green", transform=ax.transAxes)

ax.set_xlim(0, 10)
ax.set_ylim(0, 100)
ax.set_zlim(0, 100)
ax.set_xlabel('nearest neighbours')
ax.set_ylabel('no of clusters')
ax.set_zlabel('accuracy in %')

plt.show()

########## Kmeans and SVM ##############

fig2 = plt.figure("Kmeans and SVM")
plt.title("Kmeans and SVM")
x=np.array([8,20,40,60,100])
y=np.array([27,29.75,40.75,45.5,55.375])
plt.xlabel("Clusters")
plt.ylabel("Accuracy")
plt.plot(x, y, 'g*-');
plt.show()

