# -*- coding: utf-8 -*-
"""
Created on May 19 21:38:20 2018

@author: Diginnos
"""
import numpy as np
np.set_printoptions(threshold=np.inf)
import sklearn
import csv

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC

from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from matplotlib import pyplot as plt


from NeuralNetMLP import NeuralNetMLP

# for sklearn 0.18's alternative syntax
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version
if Version(sklearn_version) < '0.18':
    from sklearn.grid_search import train_test_split
    from sklearn.cross_validation import StratifiedKFold#sci_kit learn 20.0以降で廃止
    from sklearn.cross_validation import cross_val_score#sci_kit learn 20.0以降で廃止
    from sklearn.learning_curve import learning_curve
    from sklearn.learning_curve import validation_curve
    from sklearn.grid_search import GridSearchCV
else:
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import learning_curve
    from sklearn.model_selection import validation_curve
    from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix
from printCMX import print_cmx

"""
The 7 activities are: 
standing (a1), 
sitting (a2), 
walking (a3), 
down and up stairs (a4 and a5), 
lying (a6) 
eating (a7).
"""
print('sklearn version: ',sklearn.__version__)
path='Feature'
allsample = 55
activity = 7#ラベルの個数
features = 35
featureReductionComponent=15
#※事前にデータセットのラベルが右端に入力されていること

bunkatu=5#k分割の分割数
AccuracyPCA, AccuracySTD =0, 0

a = np.loadtxt('Feature/Features.csv', delimiter=',', dtype='float')
X=a[:,0:features]
y=a[:,features:features+1]
#ラベルをintに変換し、配列を1次元に変換(必須)
y=np.array(y, dtype=int)
y=y.reshape(-1,)

X_train,X_test,y_train,y_test= train_test_split(X, y, test_size=0.2, stratify=y)
#※X_train,X_testはスケーリングしていないため、必ずStandardScaler()にかける事が必要


print(X_train.shape[0],X_train.shape[1])
print(X_test.shape[0],X_test.shape[1])

#############################################################################
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
#print (X_train_std)

#############################################################################
print(50 * '=')
print('Section: Total and explained variance')
print(50 * '-')
cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

f = open('PCAtest.csv', 'w')
writer = csv.writer(f, lineterminator='\n')  # writerオブジェクトを作成
writer.writerow(eigen_vals)
f.close

tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

plt.bar(range(1, 36), var_exp, alpha=0.5, align='center',
        label='individual explained variance')
plt.step(range(1, 36), cum_var_exp, where='mid',
         label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')

# plt.tight_layout()
# plt.savefig('./figures/pca1.png', dpi=300)
plt.show()

#############################################################################


#############################################################################
#Learning part
#############################################################################
#############################################################################
print(50 * '=')
print('Section: Algorithm selection with nested cross-validation')
print(50 * '-')
pipe_svc = Pipeline([#('scl', StandardScaler()),
                     ('pca', PCA(n_components=featureReductionComponent)),
                     ('clf', SVC(random_state=1))])

param_range = [0.0001, 0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]#, 1000.0]

param_grid = [{'clf__C': param_range,
               'clf__kernel': ['linear']},
              {'clf__C': param_range,
               'clf__gamma': param_range,
               'clf__kernel': ['rbf']}]


gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=2)

# Note: Optionally, you could use cv=2
# in the GridSearchCV above to produce
# the 5 x 2 nested CV that is shown in the figure.
gs = gs.fit(X_train_std, y_train)
print('Validation accuracy', gs.best_score_)
print('Best parameters', gs.best_params_)
#print(gs.best_params_['clf__C'])
scores = cross_val_score(gs, X_train_std, y_train, scoring='accuracy', cv=bunkatu)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))



print(50 * '=')
print('Section: Combining transformers and estimators in a pipeline')
print(50 * '-')

#Best parameters set found on development set: {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}
#学習器の生成
svm = SVC(kernel=gs.best_params_['clf__kernel'], random_state=1, gamma=gs.best_params_['clf__gamma'], C=gs.best_params_['clf__C'])

#スケーリングやPCA(次元削減)
pipe_lr = Pipeline([#('scl', StandardScaler()),
                    ('pca', PCA(n_components=featureReductionComponent)),
                    ('lrn', svm)])
print('scl','pca','svm')


"""
ここからSVMのパイプラインで,学習データに対する層化k分割交差検証を実行する
"""
#############################################################################
print(50 * '=')
print('Section: K-fold cross-validation')
print(50 * '-')
print('Using StratifiedKFold')

#sci_kit learn 20.0以降 
#kfold = StratifiedKFold(n_splits=5,random_state=1).split(X_train, y_train)

kfold = StratifiedKFold(n_splits=bunkatu,
                            random_state=1).split(X_train_std, y_train)
scores=[]

for k, (train,test) in enumerate(kfold):
    pipe_lr.fit(X_train_std[train],y_train[train])
    score=pipe_lr.score(X_train_std[test],y_train[test])
    scores.append(score)
    print('Fold: %s, Class dist.: %s, Acc: %.3f' %
          (k+1, np.bincount(y_train[train]), score))

print('\n学習データに対するCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

#############################################################################
"""SVMを使った場合の学習曲線でバイアスとバリアンスを診断"""
train_sizes, train_scores, test_scores =\
    learning_curve(estimator=pipe_lr,
                   X=X_train_std,
                   y=y_train,
                   train_sizes=np.linspace(0.1, 1.0, 10),
                   cv=bunkatu,
                   n_jobs=1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

#plt.axes(axisbg="white") # 背景を変更
#plt.grid(True, which = 'major', linestyle = '-', color = '#CFCFCF')
#学習曲線グラフの描画
plt.plot(train_sizes, train_mean,
         color='blue', marker='o',
         markersize=5, label='training accuracy')

#fill_between:平均±標準偏差の幅を塗りつぶす
plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='validation accuracy')

plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')

plt.xlabel('Number of training samples', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.legend(loc='lower right', fontsize=14)
plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
#plt.gca().yaxis.set_tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)
#plt.legend(fontsize=14)
#plt.tick_params(labelbottom='on')
plt.tick_params(labelsize=14)
#plt.grid()
#plt.xticks([0,50,100,150,200,250])
plt.show()

#############################################################################

"""パイプラインをSVMで一回実行"""
#学習実行
pipe_lr.fit(X_train_std, y_train)
print('Testデータに対するAccuracy: %.3f' % pipe_lr.score(X_test_std, y_test))
y_pred = pipe_lr.predict(X_test_std)
target_names = ['Standing', 'Sitting', 'Walking','DownStair', 'UpStair', 'Lying', 'Eating']
print(classification_report(y_test, y_pred, target_names=target_names))
print_cmx(y_test, y_pred)
#print('Testデータに対するprecision: %.3f' % precision_score(y_test, y_pred, average='samples') )
#print('Testデータに対するrecall: %.3f' % recall_score(y_test, y_pred, average='samples') )
#print('Testデータに対するF1: %.3f' % f1_score(y_test, y_pred, average='samples') )
