# Structure de dossiers et fichiers
"""
data_analysis_lib/
│
├── __init__.py
├── database/
│   ├── __init__.py
│   ├── connector.py        # Connexion aux bases de données
│   └── file_importer.py    # Import de fichiers (CSV, Excel, etc.)
│
├── univariate/
│   ├── __init__.py
│   ├── numerical.py        # Analyses pour variables numériques
│   ├── categorical.py      # Analyses pour variables catégorielles
│   ├── temporal.py         # Analyses pour séries temporelles
│   └── visualization.py    # Visualisations univariées
│
├── multivariate/
│   ├── __init__.py
│   ├── correlation.py      # Analyse de corrélations
│   ├── clustering.py       # Méthodes de clustering
│   ├── dim_reduction.py    # Réduction de dimensionnalité (PCA, t-SNE, UMAP)
│   ├── causal.py           # Inférence causale
│   └── visualization.py    # Visualisations multivariées
│
└── modeling/
    ├── __init__.py
    ├── cross_validation.py # Validation croisée
    ├── regression.py       # Modèles de régression
    ├── classification.py   # Modèles de classification
    ├── hypertuning.py      # Optimisation d'hyperparamètres
    └── statistics.py       # Métriques et intervalles de confiance
"""

# Module 1: Connexion et import de données (database/connector.py)
import pandas as pd
import sqlalchemy as sa
from typing import Dict, Optional, Union, List

class DatabaseConnector:
    """Classe pour la connexion et les requêtes de base de données."""
    
    def __init__(self, connection_string: str):
        """
        Initialise la connexion à la base de données.
        
        Args:
            connection_string: Chaîne de connexion à la base de données
        """
        self.engine = sa.create_engine(connection_string)
        
    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Exécute une requête SQL et retourne un DataFrame.
        
        Args:
            query: Requête SQL à exécuter
            
        Returns:
            DataFrame contenant le résultat de la requête
        """
        return pd.read_sql_query(query, self.engine)
    
    def get_table(self, table_name: str, schema: Optional[str] = None) -> pd.DataFrame:
        """
        Récupère une table complète.
        
        Args:
            table_name: Nom de la table
            schema: Schéma de la table (optionnel)
            
        Returns:
            DataFrame contenant la table
        """
        if schema:
            query = f"SELECT * FROM {schema}.{table_name}"
        else:
            query = f"SELECT * FROM {table_name}"
        return self.execute_query(query)


# Module 1: Import de fichiers (database/file_importer.py)
class FileImporter:
    """Classe pour l'import de données depuis différents formats de fichiers."""
    
    @staticmethod
    def from_csv(filepath: str, **kwargs) -> pd.DataFrame:
        """
        Importe des données depuis un fichier CSV.
        
        Args:
            filepath: Chemin vers le fichier CSV
            **kwargs: Arguments supplémentaires pour pd.read_csv
            
        Returns:
            DataFrame contenant les données du fichier
        """
        return pd.read_csv(filepath, **kwargs)
    
    @staticmethod
    def from_excel(filepath: str, sheet_name: Union[str, int, List, None] = 0, **kwargs) -> pd.DataFrame:
        """
        Importe des données depuis un fichier Excel.
        
        Args:
            filepath: Chemin vers le fichier Excel
            sheet_name: Nom ou index de la feuille à importer
            **kwargs: Arguments supplémentaires pour pd.read_excel
            
        Returns:
            DataFrame contenant les données du fichier
        """
        return pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)
    
    @staticmethod
    def from_json(filepath: str, **kwargs) -> pd.DataFrame:
        """
        Importe des données depuis un fichier JSON.
        
        Args:
            filepath: Chemin vers le fichier JSON
            **kwargs: Arguments supplémentaires pour pd.read_json
            
        Returns:
            DataFrame contenant les données du fichier
        """
        return pd.read_json(filepath, **kwargs)


# Module 2: Analyses univariées - Variables numériques (univariate/numerical.py)
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class NumericalAnalysis:
    """Classe pour l'analyse univariée de variables numériques."""
    
    @staticmethod
    def describe(data: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Génère des statistiques descriptives pour les colonnes numériques.
        
        Args:
            data: DataFrame contenant les données
            columns: Liste des colonnes à analyser (si None, toutes les colonnes numériques)
            
        Returns:
            DataFrame contenant les statistiques descriptives
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        stats_df = data[columns].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])
        
        # Ajouter des métriques supplémentaires
        stats_df.loc['skewness'] = data[columns].skew()
        stats_df.loc['kurtosis'] = data[columns].kurtosis()
        stats_df.loc['missing'] = data[columns].isnull().sum()
        stats_df.loc['missing_pct'] = data[columns].isnull().mean() * 100
        
        return stats_df
    
    @staticmethod
    def plot_distribution(data: pd.DataFrame, column: str, bins: int = 30, figsize: tuple = (10, 6)):
        """
        Visualise la distribution d'une variable numérique.
        
        Args:
            data: DataFrame contenant les données
            column: Nom de la colonne à visualiser
            bins: Nombre de bins pour l'histogramme
            figsize: Taille de la figure
        """
        plt.figure(figsize=figsize)
        
        # Distribution avec KDE
        sns.histplot(data[column].dropna(), kde=True, bins=bins)
        
        # Ajouter des lignes verticales pour les statistiques clés
        mean_val = data[column].mean()
        median_val = data[column].median()
        
        plt.axvline(mean_val, color='r', linestyle='--', label=f'Moyenne: {mean_val:.2f}')
        plt.axvline(median_val, color='g', linestyle='-.', label=f'Médiane: {median_val:.2f}')
        
        plt.title(f'Distribution de {column}')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def outlier_analysis(data: pd.DataFrame, column: str, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        Identifie les valeurs aberrantes pour une variable numérique.
        
        Args:
            data: DataFrame contenant les données
            column: Nom de la colonne à analyser
            method: Méthode de détection ('iqr' ou 'zscore')
            threshold: Seuil pour la méthode (1.5 pour IQR, généralement 3 pour zscore)
            
        Returns:
            DataFrame contenant les valeurs aberrantes
        """
        if method == 'iqr':
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data[column].dropna()))
            outliers = data[data[column].notna()][z_scores > threshold]
        
        else:
            raise ValueError("La méthode doit être 'iqr' ou 'zscore'")
        
        return outliers


# Module 2: Analyses univariées - Variables catégorielles (univariate/categorical.py)
class CategoricalAnalysis:
    """Classe pour l'analyse univariée de variables catégorielles."""
    
    @staticmethod
    def frequency_table(data: pd.DataFrame, column: str, normalize: bool = True) -> pd.DataFrame:
        """
        Génère une table de fréquence pour une variable catégorielle.
        
        Args:
            data: DataFrame contenant les données
            column: Nom de la colonne à analyser
            normalize: Si True, inclut les fréquences relatives
            
        Returns:
            DataFrame contenant la table de fréquence
        """
        value_counts = data[column].value_counts(dropna=False)
        
        freq_table = pd.DataFrame({'count': value_counts})
        if normalize:
            freq_table['percentage'] = value_counts / len(data) * 100
            freq_table['cumulative_percentage'] = freq_table['percentage'].cumsum()
        
        return freq_table
    
    @staticmethod
    def plot_distribution(data: pd.DataFrame, column: str, top_n: Optional[int] = None, 
                         figsize: tuple = (10, 6), horizontal: bool = False):
        """
        Visualise la distribution d'une variable catégorielle.
        
        Args:
            data: DataFrame contenant les données
            column: Nom de la colonne à visualiser
            top_n: Nombre de catégories les plus fréquentes à afficher (si None, toutes)
            figsize: Taille de la figure
            horizontal: Si True, affiche un barplot horizontal
        """
        plt.figure(figsize=figsize)
        
        # Préparation des données
        value_counts = data[column].value_counts()
        if top_n is not None and len(value_counts) > top_n:
            # Garder les top_n catégories les plus fréquentes et regrouper le reste
            others = value_counts.iloc[top_n:].sum()
            value_counts = value_counts.iloc[:top_n]
            value_counts['Autres'] = others
        
        # Plot
        if horizontal:
            ax = sns.barplot(y=value_counts.index, x=value_counts.values)
            ax.set(xlabel='Count', ylabel=column)
            # Ajouter les valeurs sur les barres
            for i, v in enumerate(value_counts.values):
                ax.text(v + 0.1, i, f"{v}", va='center')
        else:
            ax = sns.barplot(x=value_counts.index, y=value_counts.values)
            ax.set(xlabel=column, ylabel='Count')
            plt.xticks(rotation=45, ha='right')
            # Ajouter les valeurs sur les barres
            for i, v in enumerate(value_counts.values):
                ax.text(i, v + 0.1, f"{v}", ha='center')
        
        plt.title(f'Distribution de {column}')
        plt.tight_layout()
        plt.show()


# Module 3: Analyses multivariées - Corrélations (multivariate/correlation.py)
class CorrelationAnalysis:
    """Classe pour l'analyse de corrélations entre variables."""
    
    @staticmethod
    def correlation_matrix(data: pd.DataFrame, method: str = 'pearson', min_periods: int = 1) -> pd.DataFrame:
        """
        Calcule la matrice de corrélation pour les variables numériques.
        
        Args:
            data: DataFrame contenant les données
            method: Méthode de corrélation ('pearson', 'kendall', 'spearman')
            min_periods: Nombre minimum de paires de valeurs non-NA requises
            
        Returns:
            DataFrame contenant la matrice de corrélation
        """
        numeric_data = data.select_dtypes(include=[np.number])
        return numeric_data.corr(method=method, min_periods=min_periods)
    
    @staticmethod
    def plot_correlation_matrix(corr_matrix: pd.DataFrame, figsize: tuple = (12, 10), 
                               mask_upper: bool = True, cmap: str = 'coolwarm'):
        """
        Visualise une matrice de corrélation sous forme de heatmap.
        
        Args:
            corr_matrix: Matrice de corrélation (DataFrame)
            figsize: Taille de la figure
            mask_upper: Si True, masque le triangle supérieur de la matrice
            cmap: Palette de couleurs pour la heatmap
        """
        plt.figure(figsize=figsize)
        
        # Créer un masque pour le triangle supérieur
        mask = None
        if mask_upper:
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Heatmap
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
                   annot=True, fmt='.2f', square=True, linewidths=.5)
        
        plt.title('Matrice de Corrélation')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def cramer_v(data: pd.DataFrame, cat_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calcule le V de Cramer pour les paires de variables catégorielles.
        
        Args:
            data: DataFrame contenant les données
            cat_columns: Liste des colonnes catégorielles (si None, utilise les colonnes de type 'object' et 'category')
            
        Returns:
            DataFrame contenant les valeurs du V de Cramer
        """
        if cat_columns is None:
            cat_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        n = len(cat_columns)
        cramer_matrix = pd.DataFrame(np.zeros((n, n)), index=cat_columns, columns=cat_columns)
        
        for i, col1 in enumerate(cat_columns):
            for j, col2 in enumerate(cat_columns):
                if i == j:
                    cramer_matrix.iloc[i, j] = 1.0
                elif i < j:
                    contingency = pd.crosstab(data[col1], data[col2])
                    chi2 = stats.chi2_contingency(contingency)[0]
                    n = contingency.sum().sum()
                    phi2 = chi2 / n
                    r, k = contingency.shape
                    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
                    rcorr = r - ((r-1)**2)/(n-1)
                    kcorr = k - ((k-1)**2)/(n-1)
                    cramer = np.sqrt(phi2corr / min(kcorr-1, rcorr-1))
                    cramer_matrix.iloc[i, j] = cramer
                    cramer_matrix.iloc[j, i] = cramer
        
        return cramer_matrix


# Module 3: Analyses multivariées - Réduction de dimension (multivariate/dim_reduction.py)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

class DimensionReduction:
    """Classe pour les méthodes de réduction de dimensionnalité."""
    
    @staticmethod
    def apply_pca(data: pd.DataFrame, n_components: Optional[int] = None, 
                 variance_threshold: float = 0.95) -> tuple:
        """
        Applique une ACP (Analyse en Composantes Principales).
        
        Args:
            data: DataFrame contenant les données numériques
            n_components: Nombre de composantes à conserver (si None, utilise variance_threshold)
            variance_threshold: Seuil de variance expliquée cumulée
            
        Returns:
            tuple (transformed_data, pca_model, explained_variance_ratio)
        """
        # Si n_components n'est pas spécifié, utiliser tous les composants
        if n_components is None:
            n_components = min(data.shape)
        
        # Initialiser et appliquer PCA
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(data)
        
        # Si un seuil de variance est spécifié et n_components n'est pas spécifié
        if n_components is None and variance_threshold is not None:
            # Déterminer le nombre de composantes à conserver
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
            
            # Réappliquer PCA avec le nombre optimal de composantes
            pca = PCA(n_components=n_components)
            transformed = pca.fit_transform(data)
        
        # Créer un DataFrame avec les composantes
        columns = [f'PC{i+1}' for i in range(transformed.shape[1])]
        transformed_df = pd.DataFrame(transformed, columns=columns, index=data.index)
        
        return transformed_df, pca, pca.explained_variance_ratio_
    
    @staticmethod
    def plot_pca_variance(explained_variance_ratio: np.ndarray, figsize: tuple = (10, 6)):
        """
        Visualise la variance expliquée par les composantes principales.
        
        Args:
            explained_variance_ratio: Ratio de variance expliquée par composante
            figsize: Taille de la figure
        """
        plt.figure(figsize=figsize)
        
        # Variance expliquée par composante
        plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
        plt.plot(range(1, len(explained_variance_ratio) + 1), 
                np.cumsum(explained_variance_ratio), 'r-o', linewidth=2)
        
        plt.xlabel('Composante Principale')
        plt.ylabel('Proportion de Variance Expliquée')
        plt.title('Variance Expliquée par les Composantes Principales')
        plt.xticks(range(1, len(explained_variance_ratio) + 1))
        plt.grid(True)
        
        # Ajouter une ligne horizontale à 95% de variance expliquée
        plt.axhline(y=0.95, color='g', linestyle='--', label='Seuil à 95%')
        
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def apply_tsne(data: pd.DataFrame, n_components: int = 2, perplexity: float = 30.0, 
                  random_state: int = 42) -> pd.DataFrame:
        """
        Applique t-SNE pour la visualisation.
        
        Args:
            data: DataFrame contenant les données numériques
            n_components: Nombre de composantes (généralement 2 ou 3 pour la visualisation)
            perplexity: Paramètre de perplexité pour t-SNE
            random_state: Graine aléatoire pour la reproductibilité
            
        Returns:
            DataFrame contenant les coordonnées t-SNE
        """
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
        tsne_result = tsne.fit_transform(data)
        
        columns = [f'TSNE{i+1}' for i in range(n_components)]
        tsne_df = pd.DataFrame(tsne_result, columns=columns, index=data.index)
        
        return tsne_df
    
    @staticmethod
    def apply_umap(data: pd.DataFrame, n_components: int = 2, n_neighbors: int = 15, 
                  min_dist: float = 0.1, random_state: int = 42) -> pd.DataFrame:
        """
        Applique UMAP pour la visualisation et la réduction de dimension.
        
        Args:
            data: DataFrame contenant les données numériques
            n_components: Nombre de composantes
            n_neighbors: Nombre de voisins à considérer
            min_dist: Distance minimale entre les points
            random_state: Graine aléatoire pour la reproductibilité
            
        Returns:
            DataFrame contenant les coordonnées UMAP
        """
        reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, 
                           min_dist=min_dist, random_state=random_state)
        umap_result = reducer.fit_transform(data)
        
        columns = [f'UMAP{i+1}' for i in range(n_components)]
        umap_df = pd.DataFrame(umap_result, columns=columns, index=data.index)
        
        return umap_df


# Module 4: Modélisation - Validation croisée (modeling/cross_validation.py)
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.base import BaseEstimator
from typing import Tuple, Dict, Any, List, Callable

class CrossValidation:
    """Classe pour la validation croisée des modèles."""
    
    @staticmethod
    def train_test_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, 
                        random_state: int = 42) -> Tuple:
        """
        Divise les données en ensembles d'entraînement et de test.
        
        Args:
            X: Features
            y: Variable cible
            test_size: Proportion de l'ensemble de test
            random_state: Graine aléatoire pour la reproductibilité
            
        Returns:
            Tuple (X_train, X_test, y_train, y_test)
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    @staticmethod
    def k_fold_cross_validation(model: BaseEstimator, X: pd.DataFrame, y: pd.Series, 
                              n_splits: int = 5, scoring: str = 'accuracy', 
                              random_state: int = 42) -> Dict[str, Any]:
        """
        Effectue une validation croisée k-fold.
        
        Args:
            model: Modèle à évaluer
            X: Features
            y: Variable cible
            n_splits: Nombre de plis
            scoring: Métrique d'évaluation
            random_state: Graine aléatoire pour la reproductibilité
            
        Returns:
            Dictionnaire contenant les scores et statistiques
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        # Calculer les scores de validation croisée
        cv_scores = cross_val_score(model, X, y, cv=kf, scoring=scoring)
        
        # Résultats
        results = {
            'cv_scores': cv_scores,
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores),
            'min_score': np.min(cv_scores),
            'max_score': np.max(cv_scores),
            'confidence_interval': (
                np.mean(cv_scores) - 1.96 * np.std(cv_scores) / np.sqrt(n_splits),
                np.mean(cv_scores) + 1.96 * np.std(cv_scores) / np.sqrt(n_splits)
            )
        }
        
        return results
    
    @staticmethod
    def grid_search(model: BaseEstimator, param_grid: Dict[str, List], X: pd.DataFrame, 
                   y: pd.Series, cv: int = 5, scoring: str = None, 
                   n_jobs: int = -1) -> GridSearchCV:
        """
        Effectue une recherche par grille pour les hyperparamètres.
        
        Args:
            model: Modèle à optimiser
            param_grid: Grille de paramètres à explorer
            X: Features
            y: Variable cible
            cv: Nombre de plis pour la validation croisée
            scoring: Métrique d'évaluation
            n_jobs: Nombre de jobs à exécuter en parallèle
            
        Returns:
            Objet GridSearchCV ajusté
        """
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs)
        grid_search.fit(X, y)
        
        return grid_search


# Module 4: Modélisation - Régressions (modeling/regression.py)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

class RegressionModels:
    """Classe pour les modèles de régression."""
    
    @staticmethod
    def get_model(model_type: str, **kwargs) -> BaseEstimator:
        """
        Crée une instance d'un modèle de régression.
        
        Args:
            model_type: Type de modèle ('linear', 'ridge', 'lasso', 'elastic_net', 
                                      'random_forest', 'gradient_boosting', 'svr',
                                      'xgboost', 'lightgbm')
            **kwargs: Paramètres spécifiques au modèle
            
        Returns:
            Instance du modèle
        """
        models = {
            'linear': LinearRegression,
            'ridge': Ridge,
            'lasso': Lasso,
            'elastic_net': ElasticNet,
            'random_forest': RandomForestRegressor,
            'gradient_boosting': GradientBoostingRegressor,
            'svr': SVR,
            'xgboost': xgb.XGBRegressor,
            'lightgbm': lgb.LGBMRegressor
        }
        
        if model_type not in models:
            raise ValueError(f"Type de modèle '{model_type}' non reconnu.")
        
        return models[model_type](**kwargs)
    
    @staticmethod
    def evaluate(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        """
        Évalue les performances d'un modèle de régression.
        
        Args:
            y_true: Valeurs réelles
            y_pred: Prédictions
            
        Returns:
            Dictionnaire contenant les métriques d'évaluation
        """
        mse = mean_squared_error(y_true, y_pred)
        
        metrics = {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'adjusted_r2': 1 - (1 - r2_score(y_true, y_pred)) * (len(y_true) - 1) / (len(y_true) - len(y_pred) - 1)
        }
        
        return metrics
    
    @staticmethod
    def get_feature_importance(model: BaseEstimator, feature_names: List[str]) -> pd.DataFrame:
        """
        Extrait l'importance des features à partir d'un modèle.
        
        Args:
            model: Modèle entraîné
            feature_names: Noms des features
            
        Returns:
            DataFrame contenant l'importance des features
        """
        # Vérifier si le modèle a un attribut feature_importances_
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        # Vérifier si c'est un modèle linéaire avec des coefficients
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
            if importances.ndim > 1:
                importances = importances.mean(axis=0)
        else:
            raise ValueError("Le modèle ne fournit pas d'importance des features.")
        
        # Créer un DataFrame avec les importances
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        # Trier par importance décroissante
        return importance_df.sort_values('importance', ascending=False)


# Module 4: Modélisation - Classifications (modeling/classification.py)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           roc_auc_score, confusion_matrix, classification_report)

class ClassificationModels:
    """Classe pour les modèles de classification."""
    
    @staticmethod
    def get_model(model_type: str, **kwargs) -> BaseEstimator:
        """
        Crée une instance d'un modèle de classification.
        
        Args:
            model_type: Type de modèle ('logistic', 'random_forest', 'gradient_boosting', 
                                       'svc', 'knn', 'naive_bayes', 'decision_tree',
                                       'xgboost', 'lightgbm')
            **kwargs: Paramètres spécifiques au modèle
            
        Returns:
            Instance du modèle
        """
        models = {
            'logistic': LogisticRegression,
            'random_forest': RandomForestClassifier,
            'gradient_boosting': GradientBoostingClassifier,
            'svc': SVC,
            'knn': KNeighborsClassifier,
            'naive_bayes': GaussianNB,
            'decision_tree': DecisionTreeClassifier,
            'xgbo