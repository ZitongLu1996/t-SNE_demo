from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import TSNE
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def draw_tsne(features, imgs):
    print(f">>> t-SNE fitting")
    # 初始化一个t-SNE
    tsne = TSNE(n_components=2, init='pca', perplexity=40)
    # 进行计算
    Y = tsne.fit_transform(features)
    print("fitting over")

    fig, ax = plt.subplots()
    # 设置图片尺寸
    fig.set_size_inches(18, 14.4)
    plt.axis('off')
    print("plotting images")
    # 单张图片位置
    imscatter(Y[:, 0], Y[:, 1], imgs, zoom=0.1, ax=ax)
    print("plot over")
    plt.savefig(fname='figure.jpg', dpi=600)
    plt.show()


def imscatter(x, y, images, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0, image in zip(x, y, images):
        # 读取图片
        im = cv2.imread(image)
        im = cv2.resize(im, (320, 405))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_f = OffsetImage(im, zoom=zoom)
        ab = AnnotationBbox(im_f, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

# 初始化特征
features = np.zeros([450, 162*128])

conditions = ["f", "u", "s"]
for i in range(3):
    for j in range(150):
        # 读取图片
        image = Image.open("stimuli/stimuli_" + conditions[i] + str(j+1).zfill(3) + ".bmp")
        # 平铺图片并赋值
        features[i*150+j] = np.reshape(image, [162*128])

# 获取图片地址
imgs = []
for condition in conditions:
    for i in range(150):
        imgs.append("stimuli/stimuli_" + condition + str(i+1).zfill(3) + ".bmp")

draw_tsne(features, imgs)