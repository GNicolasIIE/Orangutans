import pandas as pd
from typing import Dict, Optional, Union, List

# Analyses univariées - Variables numériques 
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


# Analyses univariées - Variables catégorielles 
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


# Analyses multivariées - Corrélations 
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
