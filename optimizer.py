import pandas as pd
import numpy as np
from sklearn.exceptions import NotFittedError
from .general import estim_normal_parameters, replace_keys_in_string
from .general import Chronometer
import collections
import seaborn as sb
        
class Optimizer:
    
    def __init__(self, Model, x_columns, y, X_train, X_test, param=None, 
                 condition='slide_fit>window_fit or slide_fit<=0 or sample_fit*window_fit<1',
                nbr_ite=100, quantile=.2):       
        """Optimizer - find the best parameters combinaison through normal and uniform distribution convergence
        
        :Model sklearn.base.Estimator: The model to fit.
        :x_columns list[str] features: Named columns for Dataframes used to train/test model.
        :y str target: Target column name in Dataframes to be estimated using x_columns and trained model.
        :param dict[dict] parameters_domain: Structured information for hyperparameters' repartition to optimize {'name_1':{'type_':int, 'min_'=0, 'max_':10},...}.
        :condition str problem_constrains: Logical expression string implying hyperparameters to optimize 'name_1<name_2'.

        :returns: Instanciated Optimizer object.
        :rtype: Optimizer
        '"""
        ##input parameters
        self.Model = Model
        self.x_columns, self.y = x_columns, y
        self.X_train, self.X_test = X_train, X_test
        self.param = dict()
        if param:
            for p in param:
                self.set_param(p, **param[p])
        self.condition = condition
        self.nbr_ite = nbr_ite
        self.quantile = quantile
            
        ##initialization parameters 
        #error we search for a minimum
        self.error = np.inf
        self.param_actuel = {}
        self.combi_errors = {}
                              
    def set_param(self, name_:str, type_:type, min_, max_)->None: 
        """set_param
        Sets model's initialization parameter's domain
        :name_ str: Parameter's name.
        :type_ type: Parameter's type in [int, float].
        :min_ int|float: Parameter's domain minimum.
        :max_ int|float: Parameter's domain maximum.
       
        :returns: None.
        """
        self.param.update({name_:{'func':'np.random.randint(*{})' if type_==int else 'np.random.uniform(*{})',
                                   'param':(min_, max_)}})

    def fit_combi(self, combi):  
        """fits a model with the params contained in combi argument"""
        self.model = self.Model(combi, self.x_columns, self.y).fit(self.X_train)

    def score_combi(self): 
        """computs fitted model errors with X_test"""
        index_test = [13,39]
        scores = [self.model.score(self.X_test)*self.X_test.shape[0]]
        div = self.X_test.shape[0]
        i_1 = 0
        for i in index_test:
            self.model.fit(self.X_test.iloc[i_1:i])
            i_1 = i
            scores.append(self.model.score(self.X_test.iloc[i:])*(self.X_test.shape[0]-i))
            div += self.X_test.shape[0]-i
        return sum(scores)/div
    
    def minimize_error(self, new_error):
        """error actualization"""
        if self.error > new_error:
            self.error = new_error
            print(self.error)
            
    def generate_valid_combi(self):
        """loops until generating valid combinasion of parameters, given constraints string condition"""
        condition = 'True'
        while eval(condition):
            combi = {k: eval(self.param[k]['func'].format(self.param[k]['param'])) for k in self.param}  
            condition = replace_keys_in_string(self.condition, combi)
        return combi
        
    def optim_param(self):
        """for nbr_ite iterations generates param, fits a model with the late, scores it and checks if its error is a min"""
        for ite in range(self.nbr_ite):
            combi = tuple(self.generate_valid_combi().values())
            
            self.fit_combi(combi)
            new_score = self.score_combi()
            self.combi_errors.update({combi: new_score})
            self.minimize_error(new_score)
    
    def actualize_param(self, strenght):
        """updates distribition parameters with given set of combinaision of params"""
        for name_ in self.param:
            self.param[name_]['param']=estim_normal_parameters(self.histo_fit[name_], 
                                                               weights=1-self.histo_fit['error'],
                                                               center=self.histo_fit[name_].loc[self.histo_fit['error'].idxmin()],
                                                               convergence=pow(1+strenght,-self.i))
            
            if self.param[name_]['func'] in ['np.random.randint(*{})', 'np.random.uniform(*{})']:
                self.param[name_]['func']=('int' if self.param[name_]['func']=='np.random.randint(*{})' 
                                       else '') + '(abs(np.random.normal(*{})))'
        
    def start(self, strenght=.8):
        """while discovering new parameters or having no keyboard interrupt, precises best params' generation domain"""
        self.i=0
        while self.param_actuel != self.param:
            print([self.param[k]['param'] for k in self.param])
            
            try:
                self.optim_param()
            except KeyboardInterrupt:
                self.histo_fit = pd.DataFrame(
                                [list(k)+[self.combi_errors[k]] for k in self.combi_errors],
                        columns=list(self.param.keys())+['error'])[lambda x: x.error<x.error.quantile(self.quantile)]
                self.param_actuel = {p:{copy:self.param[p][copy] for copy in self.param[p]} for p in self.param}
                self.actualize_param(strenght)
                self.i+=1
                input('Continue ?')
            except NotFittedError:
                continue

            self.histo_fit = pd.DataFrame(
                            [list(k)+[self.combi_errors[k]] for k in self.combi_errors],
                    columns=list(self.param.keys())+['error'])[lambda x: x.error<x.error.quantile(self.quantile)]
            self.param_actuel = {p:{copy:self.param[p][copy] for copy in self.param[p]} for p in self.param}
            self.actualize_param(strenght)
            self.i+=1
        return self.histo_fit[self.histo_fit.error.min()==self.histo_fit.error].iloc[0]
    
#recherche de solutions locales pour un produit de deux modèles linéaires, sous contraintes
#implémentation d'une méthode itérative de résolution du système de Cramer
class ProductOptimization:
    def __init__(self, model1_coef, model1_intercept, max1, min1, 
                       model2_coef, model2_intercept, max2, min2, x_constraint):
        """ProductOptimization - Helps finding solutions to a product of two linear outputs' maximisation problem
        
        :Model sklearn.base.Estimator: The model to fit.
        :x_columns list[str] features: Named columns for Dataframes used to train/test model.
        :y str target: Target column name in Dataframes to be estimated using x_columns and trained model.
        :param dict[dict] parameters_domain: Structured information for hyperparameters' repartition to optimize {'name_1':{'type_':int, 'min_'=0, 'max_':10},...}.
        :condition str problem_constrains: Logical expression string implying hyperparameters to optimize 'name_1<name_2'.

        :returns: Instanciated Optimizer object.
        :rtype: Optimizer
        '"""
        #enregistrement des coefficients des deux modèles, filtre des nuls (=>vecteurs constants / colinéarité)
        v1 = model1_coef
        v2 = model2_coef
        self.logic = (v1!=0)&(v2!=0)

        self.v1 = v1[self.logic]
        self.inter1 = model1_intercept
        self.v2 = v2[self.logic]
        self.inter2 = model2_intercept

        #calcul de la matrice produit A dans le sys. de Cramer Ax=b <=> x extremum de xT.(inter1,v1)T.(inter2,v2).x
        self.alpha = self.v1.reshape(-1,1)*self.v2.reshape(1,-1)
        self.n = self.alpha.shape[0]
        #b dans le sys. de Cramer Ax=b
        self.b = -self.inter1*v2-self.inter2*v1
        
        #contraintes pour les résultats de chacun des deux modèles  
        self.constraints = ((max1, min1), (max2, min2))
        #contraintes plus directes sur x
        self.x_constraint = x_constraint

    def calcul_product_with_constraints(self, x, short_path=True):
        calc1 = np.dot(x,self.v1)+self.inter1
        calc2 = np.dot(self.v2,x)+self.inter2
        return (calc1*calc2 - self.apply(self.x_init))*(
            calc1<=self.constraints[0][0])*(calc1>=self.constraints[0][1])*(
            calc2<=self.constraints[1][0])*(calc2>=self.constraints[1][1])*(
            self.x_constraint(x))/(abs(x-self.x_init).max() if short_path else 1)

    def apply(self, x):
        return (np.dot(x,self.v1)+self.inter1)*(np.dot(self.v2,x.T)+self.inter2)
    
    def norm_mat(self, x): return np.sqrt((x**2).sum())/(x.shape[0]**2)
        
    #méthode itérative de résolution du système de Cramer: calcul itératif
    def resolution_ite(self, x, M, N):
        return np.dot(np.linalg.inv(M), np.dot(N,x) + self.b)

    #méthode itérative de résolution du système de Cramer: initialisation des matrices de calcul
    def gen_M_N(self, n_alpha, dist_M):
        if dist_M:
            M = np.random.normal(0,n_alpha,(self.n,self.n))
        else :
            M = np.random.uniform(-n_alpha,n_alpha,(self.n,self.n))            
        return M, M - (self.alpha+self.alpha.T)
        
    # voir shrinkage, ajustement de la matrice M (M tend vers Idn avec des oscilations aléatoires fn des perf.) ou d'n_alpha (formule des normes - conditions initiales) ?  ou même chose ... ?
    def maximize(self, x_init, ite_rech = 1000, ite_sol = 8, dist_M = 0, short_path=True, n_alpha=None):
        self.x_init = x_init[self.logic]
        prod_min = self.calcul_product_with_constraints(self.x_init, short_path=False)
        if n_alpha is None:
            n_alpha = .5/self.norm_mat(self.alpha+self.alpha.T) 

        for rech in range(ite_rech):
            prod_max = prod_min
            self.x_ite = self.x_init

            M,N = self.gen_M_N(n_alpha, dist_M)
            while self.norm_mat(np.dot(np.linalg.inv(M), N))>=1:
                M,N = self.gen_M_N(n_alpha, dist_M)

            for i in range(ite_sol):
                self.x_ite = self.resolution_ite(self.x_ite, M, N)
                
                prod = self.calcul_product_with_constraints(self.x_ite, short_path)
                if prod > prod_max:
                    x_max = self.x_ite
                    prod_max = prod
            
            if prod_max > prod_min:
                yield x_max, prod_max
                
                
class GeneticalOptimization:
    def __init__(self, 
                 search_domains=[{'name_':'good', 'func_':np.random.randint, 'param_':(0,1)}], 
                 eval_score=sum,
                 mutation=(lambda x: x)):
        #liste des paramètres à optimiser avec leurs fonctions génératrices
        self.search_domains=search_domains
        #fonction cible à maximiser
        self.eval_score=eval_score
        #fonction pour transformer un ou pls paramètres
        self.mutation=mutation
        
    def start(self, num=1000, ite=100):
        chrono = Chronometer()
        #initialisation
        self.num = num
        self.generate()
        self.score()
        sb.kdeplot(self.scores)
        
        for i in range(ite):
            chrono.start()
            sb.kdeplot(self.scores)
            #les moins bons jeux de paramètres sont combinés avec les meilleurs puis mutés           
            self.score()
            if np.any(~np.isfinite(self.scores)):
                break
            self.crossover() 
            self.mutate()
            chrono.end()
            left = (ite-i+1)*chrono.estimate()
            print("ite {}/{} : restant § {} min {} sec. <=> {} s".format(i,ite,
                                                                            int(left//60),int(left%60), 
                                                                            left))
            
    #initialisation à l'aide des fonctions génératrices
    def generate(self): 
        if isinstance(self.search_domains, collections.Callable):
            self.population = np.array([
                [sd['func_'](*sd['param_']) for sd in self.search_domains()]
            for i in range(self.num)])
        else:
            self.population = np.array([
                [sd['func_'](*sd['param_']) for sd in self.search_domains]
            for i in range(self.num)])
    
    #calcul de la fonction à maximiser (avant mutation et crossover)
    #normalistion pour l'interpréter sous forme de probabilité de séléction
    def score(self):
        self.scores = self.eval_score(self.population)
        print("score max: ",self.scores.max())
        self.scores = (self.scores-self.scores.min())/(self.scores.max()-self.scores.min())
        
    #les individus les moins bien classés ont plus de chance de subir une mutation 
    def mutate(self): self.population = np.array([p.tolist() if np.random.binomial(1, s)==1 else self.mutation(np.array(p)).tolist()
                                         for p,s in zip(self.population, self.scores)])
        
    #les moins bien classés ont plus de chance de recevoir les valeurs de paramètres des autres via "combine"
    def crossover(self):
        self.population = self.population.tolist()
        self.crossed = np.array(range(self.num))[np.random.binomial(1, self.scores)==0].tolist()
        self.crosser = np.random.permutation(np.array(range(self.num))[np.random.binomial(1, self.scores)==1])
        k=0
        while self.crossed!=[]:
            i=self.crossed.pop()
            j=self.crosser[k]
            self.population[i]=self.combine(self.population[i],self.scores[i],
                                            self.population[j],self.scores[j])
            k+=1
            if k==len(self.crosser):
                k=0
        self.population = np.array(self.population)
        
    #combinaison des valeurs de deux jeux de paramètres p1/p2 suivant prédominance d'un score s1/s2
    def combine(self, p1, s1, p2, s2): return np.array(p1)-(np.array(p1)-np.array(p2))*np.random.uniform(0,.1)
        #return [p_1 if np.random.binomial(1, s1/(s1+s2))==1 else p_2 for p_1,p_2 in zip(p1,p2)