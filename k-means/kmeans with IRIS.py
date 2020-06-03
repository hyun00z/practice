#laod data
from sklearn import datasets
import pandas as pd
iris = datasets.load_iris()
X=iris.data
y=iris.target

#split data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

#PCA
from sklearn.decomposition import PCA
pca= PCA(n_components=2)
pca.fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

#k-means
from sklearn import cluster
clf = cluster.KMeans(init='k-means++', n_clusters=3)
clf.fit(X_train_pca,y_train)

###predict
y_pred = clf.predict(X_test_pca)
##test
#predict
pred = pd.DataFrame(y_pred)
pred.columns=['pred']
feature = pd.DataFrame(X_test_pca)
p = pd.concat([feature,pred],axis=1)
p.columns=['f1','f2','pred']
print(p.head())

#real
real = pd.DataFrame(y_test)
real.columns=['real']
r = pd.concat([feature, real], axis=1)
r.columns=['f1','f2','real']
print(r.head())

#crosstable
ct = pd.crosstab(r['real'],p['pred'])
print(ct)

#무게중심 계산
centers = pd.DataFrame(clf.cluster_centers_,columns=['f1','f2'])
center_x = centers['f1']
center_y = centers['f2']

# scatter plot
import matplotlib.pyplot  as plt
plt.scatter(p['f1'],p['f2'], c=p['pred'],alpha=0.5)
plt.scatter(center_x,center_y,s=50,marker='D',c='R')
plt.figure()
plt.show()

#cluster 개수
ks = range(1, 6)
inertias = []

for k in ks:
    clf = cluster.KMeans(n_clusters=k, init='k-means++')
    clf.fit(X_train_pca, y_train)
    inertias.append(clf.inertia_)

# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()