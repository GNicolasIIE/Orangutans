from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.metrics import calinski_harabasz_score as ch_score
from sklearn.cluster import KMeans as KM
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.mixture import GaussianMixture as GM
from sklearn.linear_model import SGDRegressor as SGDR
from sklearn.pipeline import Pipeline


def treadoffProbaPlot(pred_proba, y):
    treadoffProba(pred_proba, y).plot(xlabel='seuil de proba', ylabel='taux', grid=True)
    plt.show()
    
def treadoffProbaChoose(pred_proba, y):    
    return treadoffProba(pred_proba, y).F1.idxmax()

def treadoffProba(pred_proba, y):
    values = pd.DataFrame([
    [f1_score(y==1, pred_proba>i*.01),
     precision_score(y==1, pred_proba>i*.01),
     recall_score(y==1, pred_proba>i*.01),
     accuracy_score(y==1, pred_proba>i*.01),
    ] for i in range(0,101)], 
        index = [i*.01 for i in range(0,101)],
        columns=['F1','Precision', 'Recall', 'Accuracy'])
    values.plot(xlabel='seuil de proba', ylabel='taux', grid=True)
    plt.show()
    return values.F1.idxmax()
    
def significantCoeff(series, alpha=1.5):
    series_ = series.abs()
    return series_.max()-alpha*((((series_.max()-series_)**2).mean())**.5)

def plot3D(X, alpha=.2, c=None):
    fig = plt.figure(figsize=(6,8))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(X[:,0],X[:,1],X[:,2],alpha=alpha,c=c)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
    
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
len_max_colors = len(colors)
    
    
class ModelDF:
    def __init__(self, model, x_columns):
        self.model = model
        self.x_columns = x_columns
    def fit(self, X, Y):
        self.model.fit(X[self.x_columns],Y)
    def predict(self, X):
        return self.model.predict(X[self.x_columns])
    def score(self, X, Y):
        return self.model.score(X[self.x_columns], Y)
    def transform(self, X, Y=None):
        return self.model.transform(self, X[self.x_columns], Y=None)


class LogisticPipeline:
    def __init__(self):
        self.lr = LogisticRegression(class_weight='balanced', random_state=100)
        self.ss = StandardScaler()
        self.pca = PCA()
        self.threshold = .5

    def fit(self, X, Y, cuts=False):
        m = max([(1-Y).sum(),Y.sum()]) #sur-échantillonnage => sous-échantillonnage des non-churneurs
        X_eq = pd.concat([X[Y].sample(m, replace=True),X[~Y].sample(m, replace=True)], axis=0)
        Y_eq = np.concatenate((np.repeat(True,m),np.repeat(False,m)))
        
        self.differences = dict()
        for col in X_eq.columns:
            var = X_eq[col].var()
            if var != 0:
                self.differences.update({col :(
                ((X_eq[col][Y_eq].mean()-X_eq[col][~Y_eq].mean())**2)/var
                )})
        self.differences = pd.Series(self.differences).sort_values(ascending=False)

        if cuts:
            self.setCutDiff()
            
        self.pca.fit(
            self.ss.fit_transform(
                X_eq[self.differences.index])*self.differences.values)

        if cuts:
            self.setCutPCA()
            self.pca = PCA(n_components = self.pca.n_components_)
            self.pca.fit(
                self.ss.fit_transform(
                    X_eq[self.differences.index])*self.differences.values)
            
        
        self.lr.fit(self.transform(X), Y)
    
    def predict(self, X):
        return self.predict_proba(X)>=self.threshold
    
    def predict_proba(self, X):
        return self.lr.predict_proba(
                    self.pca.transform(
                        self.ss.transform(X[self.differences.index])*self.differences.values))[:,1]
    
    def transform(self, X):
        return self.pca.transform(
                        self.ss.transform(X[self.differences.index])*self.differences.values)
    def inverse_transform(self, X):
        return pd.DataFrame(self.ss.inverse_transform(
                        self.pca.inverse_transform(X))/self.differences.values, columns=self.differences.index)
    
    def thresholdPlot(self, X, Y):
        pred_proba = self.predict_proba(X)
        treadoffProba(pred_proba, Y)
        plt.show()
        self.threshold = input("Seuil ? 'Enter'->Non Int->Oui ")
        if self.threshold!='': 
            self.threshold = float(self.threshold)
        else: 
            self.threshold =.5
        
    def setCutDiff(self):
        plt.grid()
        diff = self.differences.reset_index(drop=True)
        diff.index = 1+diff.index
        diff.plot()
        plt.hlines(xmin=1,xmax=diff.index.max(),y=0,color='green')
        plt.show()
        cut = input("Couper ? 'Enter'->Non Int->Oui ")
        if cut!='': 
            self.differences = self.differences[:int(cut)]
    def setCutPCA(self): 
        plt.grid()
        plt.plot(range(1,self.pca.n_components_+1), self.pca.explained_variance_ratio_.cumsum())
        plt.hlines(xmin=1,xmax=self.pca.n_components_,y=1,color='green')
        plt.vlines(x=3,ymin=0,ymax=1,color='red')
        plt.show()
        cut = input("Couper ? 'Enter'->Non Int->Oui ") 
        if cut!='':
            self.pca.n_components_=int(cut)            
           
    def components(self):
        return pd.DataFrame(self.pca.components_*self.differences.values.reshape(1,-1), columns=self.differences.index)
    def componentsPlot(self, N=3):
        mat = self.components()
        for m in range(N):
            coeffs = mat.iloc[m,:].sort_values()
            sc = significantCoeff(coeffs)
            coeffs[lambda x: abs(x)>sc].plot(kind='barh',color=colors[m%len_max_colors], figsize=(5,10))
            plt.show()
            
    def coeffs(self):
        return pd.Series(self.pca.inverse_transform(self.lr.coef_)[0,:]*self.differences.values, index=self.differences.index)
    def coeffsPlot(self, alpha=1.5, figsize=(5,10)):
        coeffs = self.coeffs().sort_values()
        sc = significantCoeff(coeffs, alpha)
        coeffs = coeffs[lambda x: abs(x)>sc]
        coeffs.plot(kind='barh',color=['lightgreen']*(coeffs<0).sum()+['coral']*(coeffs>=0).sum(), figsize=figsize)
    
    def plot3D(self, X, Y, alpha=.4):
        plot3D(self.transform(X), alpha=alpha, c=Y.apply(lambda x: 'coral' if x else 'lightgreen'))
       
    def clusterize(self, X, Y, path, pca_n=None, min_n=2, max_n=10):
        if pca_n is None:
            self.pca_n_cluster = self.pca.n_components_
        else:
            self.pca_n_cluster = pca_n
        self.clusterizer = Clusterizer(min_n=min_n, max_n=max_n)
        X_pca = self.transform(X)[:,:self.pca_n_cluster]
        pred = self.clusterizer.fit(X_pca)
        fig = plt.figure(figsize=(6,8))
        ax = fig.add_subplot(projection='3d')

        for i in range(self.clusterizer.n):
            ax.scatter(X_pca[pred==i,0],X_pca[pred==i,1],X_pca[pred==i,2],alpha=0.2)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.legend(range(self.clusterizer.n))
        plt.show()
        
        order_metric = X.var(axis=0)/pd.concat([X.loc[pred==i,:].var(axis=0) for i in range(self.clusterizer.n)], axis=1).mean(axis=1).values
        order_metric = order_metric.fillna(order_metric.min()/2).values.argsort()[::-1]
        pred = pd.Series(pred, index=Y.index, name='Volumes')
        counts = pred.value_counts()
        out = pd.concat([
              Y, 
             pred.apply(lambda x: counts[x]), 
            X.iloc[:,order_metric]],axis=1)
        pd.concat([out[pred==i].mean(axis=0) for i in range(self.clusterizer.n)], axis=1).to_excel('output/3_model_'+path+'.xlsx')
    def predict_labels(self, X):
        return self.clusterizer.predict(self.transform(X)[:,:self.pca_n_cluster])
    def predict_labels_proba(self, X):
        return self.clusterizer.predict_proba(self.transform(X)[:,:self.pca_n_cluster])
    def plotDensity(self, X, Y):
        X_pca = self.transform(X)
        plt.grid()
        sb.kdeplot(X_pca[~Y,0],X_pca[~Y,1],shade=True,cmap="Greens_d",cbar=True)
        plt.show()
        plt.grid()
        sb.kdeplot(X_pca[Y,0],X_pca[Y,1],shade=True,cmap="Reds_d",cbar=True)
        plt.show()
        
    def save(self, path):
        dump(self, 'data/3_model/model/'+path+'/LRP.joblib')

    def load(path):
        return load('data/3_model/model/'+path+'/LRP.joblib')
    
    
class Clusterizer:
    def __init__(self, min_n=2, max_n=10):
        self.inter_max_train=150
        self.inter_max_eval=200
        self.min_n, self.max_n = min_n, max_n
    def fit(self, X):
        self.n = self.min_n
        gm_ = [GM(n_components=n, max_iter=self.inter_max_train, covariance_type='tied', random_state=n)
              for n in range(self.min_n, self.max_n)]
        score_ = [ch_score(X, gm.fit_predict(X)) for gm in gm_]
        self.n = np.array(score_).argmax()+self.min_n
        plt.scatter(range(self.min_n, self.max_n),score_)
        plt.show()
        #self.mms = MinMaxScaler()
        self.gm = GM(n_components=self.n, max_iter=self.inter_max_eval, covariance_type='tied',
                      random_state=100) 
        labeled = self.gm.fit_predict(X)
            #self.mms.fit_transform(X))
        self.discrete = np.concatenate(tuple([ 
                np.concatenate((X[(labeled==center),:].min(axis=0).reshape(1,-1),
                          X[(labeled==center),:].max(axis=0).reshape(1,-1)), axis=0)
             for center in range(self.n)]), axis=0)
        self.discrete  = [set(self.discrete[:,i])
            for i in range(self.discrete.shape[1])]
        return labeled
    def predict(self, X): 
        return self.gm.predict(X)
            #self.mms.transform(X))
    def predict_proba(self, X):
        return self.gm.predict_proba(X)
            #self.mms.transform(X))
    def discretize(self, X):
        return np.concatenate(tuple([sum([X[:,i]>d for d in self.discrete[i]]).reshape(-1,1)
            for i in range(X.shape[1])]), axis=1)    
    def save(self, path):
        dump(self, 'data/3_model/GM_'+path+'.joblib')
    def load(path):
        return load('data/3_model/GM_'+path+'.joblib')

class MahanobisClassifier:
    def __init__(self):
        pass
    
    def fit(self, X, Y, y_values=None, centers=None):
        if y_values is None:
            self.y_values = set(Y)
        else :
            self.y_values = y_values
        
        if centers is None:
            self.mean_val = {y_val: X[Y==y_val,:].mean(axis=0).reshape(1,-1) for y_val in self.y_values}
        else:
            self.mean_val = centers
        
        self.cov_val = {y_val: pd.DataFrame(X[Y==y_val,:]).astype(float).cov().values for y_val in self.y_values}
        
        epsilon = np.random.normal(0,.000001, (X.shape[1], X.shape[1]))
        self.cov_val = {y_val: np.linalg.inv(self.cov_val[y_val]+epsilon)
                        for y_val in self.y_values}
        
    def predict(self, X):        
        cluster_dist = np.exp(-((np.concatenate(tuple([             
            (np.dot(X-self.mean_val[y_val], self.cov_val[y_val])*(X-self.mean_val[y_val])).sum(axis=1).reshape(-1,1)
                        for y_val in self.y_values
                        ]), axis=1)**2)**.25).astype(float))
        return cluster_dist/cluster_dist.sum(axis=1).reshape(-1,1)
    
class KNearestCentroides(BaseEstimator, ClassifierMixin) :
    def __init__(self):
        self.inter_max_train=[80, 50]
        self.inter_max_eval=[200, 150]
    def fit(self, X, y):
        self.n = [2,2]
        for label in [0,1]:
            score = -np.inf
            asc = True
            while asc:
                km = KM(n_clusters=self.n[label], max_iter=self.inter_max_train[label], random_state=self.n[label])
                km.fit(X[y==label])
                new_score = ch_score(X[y==label], km.labels_)
                if score<=new_score:
                    score = new_score
                    self.n[label]+=1
                else:
                    asc=False
        self.km = [KM(n_clusters=self.n[label], max_iter=self.inter_max_eval[label], random_state=100) for label in [0,1]]
        for label in [0,1]:
            self.km[label].fit(X[y==label])
        labeled = self.predict_labels_lin(X)
        self.mahanobis = MahanobisClassifier()
        self.mahanobis.fit(X.values, labeled)
        labeled = self.predict_labels_lin_prob(X)
        self.discrete = np.concatenate(tuple([ 
                np.concatenate((X.loc[(labeled==center+label*self.n[0]),:].min(axis=0).values.reshape(1,-1),
                          X.loc[(labeled==center+label*self.n[0]),:].max(axis=0).values.reshape(1,-1)), axis=0)
            for label in [0,1] for center in range(self.n[label])]), axis=0)
        self.discrete  = [set(self.discrete[:,i])
            for i in range(self.discrete.shape[1])]
    def distToCenters(self, x, label):
        return ((x.values-self.km[label].cluster_centers_)**2).sum(axis=1)
    def predict(self, X):
        return np.concatenate((
            pd.DataFrame(X).apply(lambda x: self.distToCenters(x, label=0).min(), axis=1).values.reshape(-1,1),
            pd.DataFrame(X).apply(lambda x: self.distToCenters(x, label=1).min(), axis=1).values.reshape(-1,1)),
            axis=1).argmin(axis=1)
    def predict_proba(self, X):
        return self.mahanobis.predict(X.values)
    def predict_labels(self, X):
        predicted = self.predict(X)
        labels_ = np.concatenate((self.km[0].predict(X).reshape(-1,1), self.km[1].predict(X).reshape(-1,1)),axis=1)
        return np.concatenate((predicted.reshape(-1,1), (labels_[:,0]*(1-predicted)+labels_[:,1]*predicted).reshape(-1,1)),axis=1)
    def predict_labels_lin(self, X):
        labeled = self.predict_labels(X)
        labeled[labeled[:,0]==1,1] = labeled[labeled[:,0]==1,1]+self.n[0]
        return labeled[:,1]
    def predict_labels_lin_prob(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def transform(self, X):
        return np.concatenate((
            list(pd.DataFrame(X).apply(lambda x: self.distToCenters(x, 0).tolist(), axis=1).values),
            list(pd.DataFrame(X).apply(lambda x: self.distToCenters(x, 1).tolist(), axis=1).values)),
            axis=1)
    def discretize(self, X):
        return np.concatenate(tuple([sum([X.values[:,i]>d for d in self.discrete[i]]).reshape(-1,1)
            for i in range(X.shape[1])]), axis=1)    
    def save(self, path):
        dump(self, 'data/3_model/KNC_'+path+'.joblib')

    def load(path):
        return load('data/3_model/KNC_'+path+'.joblib')
    
    
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score

# Function to find the best number of components for PCA comsbined with optimal KMeans clusters
def find_optimal_pca_kmeans(X, max_components=10, max_clusters=10):
    best_score = -1
    best_pca_components = None
    best_kmeans_clusters = None

    for n_components in range(1, max_components + 1):
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)

        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X_pca)
            silhouette_avg = calinski_harabasz_score(X_pca, cluster_labels)
            
            if silhouette_avg > best_score:
                best_score = silhouette_avg
                best_pca_components = n_components
                best_kmeans_clusters = n_clusters

    return best_pca_components, best_kmeans_clusters

# Find the best number of components for PCA combined with optimal KMeans clusters
best_pca_components, best_kmeans_clusters = find_optimal_pca_kmeans(X)

print("Best number of PCA components:", best_pca_components)
print("Best number of KMeans clusters:", best_kmeans_clusters)

# Apply PCA with the best number of components
pca = PCA(n_components=best_pca_components)
X_pca = pca.fit_transform(X)

# Apply KMeans clustering with the best number of clusters
kmeans = KMeans(n_clusters=best_kmeans_clusters, random_state=42)
kmeans.fit(X_pca)
"""


class SGDR_time_series:
    """Linear Regression model for Time-Series with exogene factors

    Uses **Linear Regression model **optimized by Gradient Descent** and fitted with exogene factors to predict a Time-Serie output. For that, the model focuses on a fixed time-*window* and *slides* to the next (given a nomber of time points to do this transposition).
    This choice allows a linear modelisation for further problem optimization or for a non-complexe factors' analysis. Also permites an automated actualisation of the model across time. On this last benefit, we entroduce a predict-fit function that permites that dynamic actualisation at any time with fresh data, and to test parameter in real condictions.
    
    .. WARNING::

            BE HAPPY
            
    :param str text: The text to translate.

    :returns: the translated text.
    :rtype: str

    :raises <Exception>: <Description de l'exception>.

    >>> translate("Hello!")
    'sssss!'
    >>> translate("I am a sssnake!")
    's ss s sssssss!'
    """
    def __init__(self, combi, x_columns, y_column):
        """Init

        Instanciate SGDR_time_series object with its params.
        
        :param str text: The text to translate.

        :returns: Instanciated SGDR_time_series object.
        :rtype: SGDR_time_series

        :raises <Exception>: <Description de l'exception>.

        >>> translate("Hello!")
        'sssss!'
        >>> translate("I am a sssnake!")
        's ss s sssssss!'"""
        self.window_fit, self.slide_fit, self.sample_fit, self.eta = combi
        self.columns = x_columns
        self.y = y_column
        self.model = SGDR(learning_rate='constant', eta0=self.eta)
        self.ite = int(self.window_fit/self.sample_fit)
        self.x_rest_to_fit = pd.DataFrame([], columns= x_columns)
        self.scores = []

    def fit_sample_xy(self, X):
        for sampling in range(self.ite):
            yield X.sample(int(self.window_fit*self.sample_fit), replace=True)
 
    def fit_decouple_xy(self, gen):
        for x in gen:
            yield x[self.columns], x[self.y]

    def fit_param(self, X):
        return self.fit_decouple_xy(self.fit_sample_xy(X))
    
    def fit_ite(self, X):
        for x in self.fit_param(X):
            self.model.partial_fit(*x)
    
    def fit_ite_last(self, X, last_index):
        if last_index < X.shape[0]:
            self.x_rest_to_fit = X.iloc[last_index:,:]
        else:
            self.x_rest_to_fit = pd.DataFrame([], columns= self.columns)
    
    def fit(self, X):  
        pf = 0
        for pf in range(0, X.shape[0]-self.window_fit+1, self.slide_fit):
            self.fit_ite(X.iloc[pf:pf+self.window_fit,:])
            self.scores.append(self.score(X.iloc[pf:pf+self.window_fit,:]))
        self.fit_ite_last(X, last_index=pf+self.window_fit)
                               
        return self
                                                              
    def predict_sample(self, X):
        res = self.model.predict(X[self.columns])
        self.scores.append(self.score_calc(res, X[self.y]))   
        self.fit_ite(X)
        return res
        
    def predict_fit_ite(self, X): 
        X = pd.concat([self.x_rest_to_fit,X], axis=0)
        
        logic = (X.shape[0]>=self.window_fit)
        if logic:
            fitted_size = -self.x_rest_to_fit.shape[0]
        
        while X.shape[0]>=self.window_fit:
            res = self.predict_sample(X.iloc[:self.window_fit])[-fitted_size:]
            fitted_size = self.slide_fit
            X = X.iloc[fitted_size:]
            yield res
        else:
            if logic:
                fitted_size = self.window_fit-self.slide_fit
            else:
                fitted_size =self.x_rest_to_fit.shape[0]
                        
        self.x_rest_to_fit = X
        yield self.predict(X.iloc[fitted_size:])

    def predict_fit(self, X): 
        return np.concatenate(tuple([x for x in self.predict_fit_ite(X)]))
    
    def predict(self, X): 
        return self.model.predict(X[self.columns]) if X.shape[0]!=0 else np.array([])

    def score(self, X): return self.score_calc(self.model.predict(X[self.columns]),X[self.y])

    def score_calc(self, y_hat, y): return ((y_hat - y)**2).sum()/((y.mean() - y)**2).sum()
    
"""
Avec Scaler dans le Pipeline
class OnlinePipeline(Pipeline):
    def partial_fit(self, X, y=None):
        for i, step in enumerate(self.steps):
            name, est = step
            est.partial_fit(X, y)
            if i < len(self.steps) - 1:
                X = est.transform(X)
        return self   
"""