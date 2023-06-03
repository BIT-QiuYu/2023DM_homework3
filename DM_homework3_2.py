from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 搜索最佳得文本特征维度
num_features = [100, 500, 800, 1000, 1500, 2500, 3000, 3550, 3800, 4000, 4400, 4800, 5000, 6000, 7000, 10000]
accuracy = []
for i in range(len(num_features)):
    num_feature = num_features[i]
    # 计算每个文档的TFIDF特征值
    vectorizer_tfidf = TfidfVectorizer(max_features=num_feature)
    docs_feats_tfidf = vectorizer_tfidf.fit_transform(docs_feats).toarray()
    # k-means聚类 将文档的特征空间分为20类
    k_means = KMeans(n_clusters=20, random_state=14, init='k-means++', n_init='auto')
    # 迭代并预测
    pred = k_means.fit_predict(docs_feats_tfidf)
    # 估计分类准确度
    slice = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 15997,
             16997, 17997, 18997, 19997]
    correct = 0
    for i in range(len(slice) - 1):
        correct += np.max(np.unique(pred[slice[i]:slice[i + 1]], return_counts=True)[1])
    accuracy.append(correct / len(pred))
# 画出准确度随着而往那边特征维数的变化曲线
plt.figure()
plt.title('accuracy of cluster')
plt.xlabel('dimension of text-features')
plt.plot(num_features, accuracy)
plt.show(block=True)



# 取最好得结果算CH_score
vectorizer_tfidf = TfidfVectorizer(max_features=3550)
docs_feats_tfidf = vectorizer_tfidf.fit_transform(docs_feats).toarray()
k_means = KMeans(n_clusters=20, random_state=14, init='k-means++', n_init='auto')
pred = k_means.fit_predict(docs_feats_tfidf)
ch_score = calinski_harabasz_score(docs_feats_tfidf, pred)
print('CH-score:', ch_score)

# PCA算法降低维度方便可视化
pca = PCA(n_components=3)
visualizition = pca.fit_transform(docs_feats_tfidf)

# 将结果可视化
x = np.array(visualizition[:, 0])
y = np.array(visualizition[:, 1])
z = np.array(visualizition[:, 2])

ax = plt.subplot(projection='3d')
ax.set_title('Visualizition of cluster')
ax.scatter(x, y, z, c=[pred], cmap='magma', alpha=0.3)  # 绘制三维数据点
# 设置坐标轴
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show(block=True)
