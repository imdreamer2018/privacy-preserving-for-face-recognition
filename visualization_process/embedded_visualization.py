import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

# load embedded
embedded = np.load('../face_embedded_data/paper/embedded_nn4_lfw_10.npy')

# load customDataset
metadata = np.load('../face_embedded_data/paper/metadata_nn4_lfw_10.npy')
print(metadata.shape)
targets = np.array([m.name for m in metadata])
print(targets)
X_embedded = TSNE(n_components=2, init='pca', method='exact').fit_transform(embedded)
# print(X_embedded.shape)
# print(X_embedded)
colors = iter(cm.rainbow(np.linspace(0, 1, len(list(enumerate(set(targets)))))))
print(X_embedded)
print(list(enumerate(set(targets))))
# try:
#     while True:
#         print(next(colors))
# except StopIteration:
#     pass

def plt3d():
    fig = plt.figure()
    ax = Axes3D(fig)
    # 添加坐标轴(顺序是Z, Y, X)
    # ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    # ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    # ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    for i, t in enumerate(set(targets)):
        idx = targets == t
        ax.scatter(X_embedded[idx, 0], X_embedded[idx, 1], X_embedded[idx, 2], colors=next(colors))


def plt2d():
    for i, t in enumerate(set(targets)):
        idx = targets == t
        plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1])


plt2d()
plt.legend(bbox_to_anchor=(1, 1))
# 添加坐标轴(顺序是Z, Y, X)
# ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
# ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
# ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
plt.title('Embedded visualization when number of people is 10 ')
plt.show()
