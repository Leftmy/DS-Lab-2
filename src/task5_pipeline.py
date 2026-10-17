"""
Task 5: Integrated Data Preprocessing Pipeline with Cross-Validation

This module implements an end-to-end data preprocessing pipeline that combines
cleaning, scaling, and feature engineering with proper cross-validation to
prevent data leakage.

Author: MLDS Course
Date: December 2025
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split, cross_val_score, learning_curve, 
    GridSearchCV, StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score, confusion_matrix, 
    roc_curve, auc, roc_auc_score
)
from sklearn.datasets import make_classification
from typing import Tuple, Dict, List, Optional, Any
import warnings
import joblib

warnings.filterwarnings('ignore')


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Custom transformer for outlier detection and handling.
    
    Parameters:
        method (str): Method for outlier detection
        strategy (str): Strategy for handling outliers
        threshold (float): Threshold for detection
    """
    
    def __init__(self, method: str = 'iqr', strategy: str = 'cap', threshold: float = 1.5):
        self.method = method
        self.strategy = strategy
        self.threshold = threshold
        self.bounds_ = {}
        
    def fit(self, X, y=None):
        """
        Fit the outlier handler by computing bounds.
        
        Parameters:
            X: Feature matrix
            y: Target vector (ignored)
            
        Returns:
            self: Fitted transformer
        """
        if self.method not in {'iqr', 'zscore'}:
            raise ValueError("method must be 'iqr' or 'zscore'.")
        if self.strategy not in {'cap', 'remove'}:
            raise ValueError("strategy must be 'cap' or 'remove'.")
        if not np.isfinite(self.threshold) or self.threshold < 0:
            raise ValueError("threshold must be a finite non-negative number.")

        if isinstance(X, pd.DataFrame):
            X = X.copy()
            feature_names = np.asarray(X.columns, dtype=object)
        else:
            raw = np.asarray(X)
            feature_names = np.asarray([f'x{i}' for i in range(raw.shape[1])], dtype=object)
            X = pd.DataFrame(raw, columns=feature_names)
        self.feature_names_in_ = feature_names
        self.n_features_in_ = X.shape[1]
        self.bounds_ = {}
        
        for col in X.columns:
            values = pd.to_numeric(X[col], errors='coerce').replace(
                [np.inf, -np.inf], np.nan
            )
            if values.notna().sum() == 0:
                self.bounds_[col] = (np.nan, np.nan)
                continue
            if self.method == 'iqr':
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - self.threshold * IQR
                upper = Q3 + self.threshold * IQR
                self.bounds_[col] = (lower, upper)
            elif self.method == 'zscore':
                mean = values.mean()
                std = values.std(ddof=0)
                lower = mean - self.threshold * std
                upper = mean + self.threshold * std
                self.bounds_[col] = (lower, upper)
                
        return self
    
    def transform(self, X):
        """
        Transform data by handling outliers.
        
        Parameters:
            X: Feature matrix
            
        Returns:
            np.ndarray: Transformed feature matrix
        """
        if isinstance(X, pd.DataFrame):
            X = X.copy()
        else:
            X = pd.DataFrame(X, columns=getattr(self, 'feature_names_in_', None))
        if hasattr(self, 'n_features_in_') and X.shape[1] != self.n_features_in_:
            raise ValueError("Input feature count differs from the fitted data.")
        
        if self.strategy == 'cap':
            for col in X.columns:
                if col in self.bounds_ and np.isfinite(self.bounds_[col]).all():
                    lower, upper = self.bounds_[col]
                    X[col] = X[col].clip(lower=lower, upper=upper)
        elif self.strategy == 'remove':
            # Mark outliers as NaN for imputation
            for col in X.columns:
                if col in self.bounds_ and np.isfinite(self.bounds_[col]).all():
                    lower, upper = self.bounds_[col]
                    mask = (X[col] < lower) | (X[col] > upper) | ~np.isfinite(X[col])
                    X.loc[mask, col] = np.nan
        return X.values


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for feature engineering.
    
    Parameters:
        interactions (bool): Whether to create interaction features
        polynomials (bool): Whether to create polynomial features
        transformations (bool): Whether to create mathematical transformations
    """
    
    def __init__(self, interactions: bool = True, polynomials: bool = False, 
                 transformations: bool = True):
        self.interactions = interactions
        self.polynomials = polynomials
        self.transformations = transformations
        self.feature_names_ = []
        self.n_features_in_ = 0
        
    def fit(self, X, y=None):
        """
        Fit the feature engineer.
        
        Parameters:
            X: Feature matrix
            y: Target vector (ignored)
            
        Returns:
            self: Fitted transformer
        """
        X_array = np.asarray(X)
        if X_array.ndim != 2:
            raise ValueError("X must be a two-dimensional feature matrix.")
        self.n_features_in_ = X_array.shape[1]
        self.feature_names_in_ = np.asarray(
            X.columns if isinstance(X, pd.DataFrame)
            else [f'x{i}' for i in range(X_array.shape[1])],
            dtype=object,
        )
        names = list(self.feature_names_in_)
        generated_names = names.copy()
        if self.transformations:
            generated_names.extend(f'{name}_log' for name in names)
            generated_names.extend(f'{name}_sqrt' for name in names)
        if self.interactions and self.n_features_in_ <= 10:
            generated_names.extend(
                f'{names[i]}_x_{names[j]}'
                for i in range(min(3, self.n_features_in_))
                for j in range(i + 1, min(3, self.n_features_in_))
            )
        if self.polynomials and self.n_features_in_ <= 5:
            generated_names.extend(f'{name}_square' for name in names)
        self.feature_names_ = np.asarray(generated_names, dtype=object)
        return self
    
    def transform(self, X):
        """
        Transform data by creating new features.
        
        Parameters:
            X: Feature matrix
            
        Returns:
            np.ndarray: Transformed feature matrix with new features
        """
        X_array = np.asarray(X, dtype=float)
        if X_array.ndim != 2:
            raise ValueError("X must be a two-dimensional feature matrix.")
        if X_array.shape[1] != self.n_features_in_:
            raise ValueError("Input feature count differs from the fitted data.")
        X_array = np.where(np.isfinite(X_array), X_array, 0.0)
        features = [X_array]
        
        # Mathematical transformations
        if self.transformations:
            # Log transform (add small constant to avoid log(0))
            log_features = np.log1p(np.abs(X_array) + 1e-8)
            features.append(log_features)
            
            # Square root transform
            sqrt_features = np.sqrt(np.abs(X_array))
            features.append(sqrt_features)
        
        # Interaction features (limited to avoid explosion)
        if self.interactions and X_array.shape[1] <= 10:
            n_features = X_array.shape[1]
            for i in range(min(3, n_features)):
                for j in range(i + 1, min(3, n_features)):
                    interaction = X_array[:, i] * X_array[:, j]
                    features.append(interaction.reshape(-1, 1))
        
        # Polynomial features (degree 2, limited features)
        if self.polynomials and X_array.shape[1] <= 5:
            square_features = X_array ** 2
            features.append(square_features)
        
        return np.hstack(features)

    def get_feature_names_out(self, input_features=None):
        if not self.feature_names_.size:
            raise ValueError("FeatureEngineer must be fitted before requesting feature names.")
        if input_features is not None and len(input_features) != self.n_features_in_:
            raise ValueError("input_features length does not match the fitted feature count.")
        return self.feature_names_.copy()


class AdaptiveSelectKBest(SelectKBest):
    """Select no more features than are available after feature generation."""

    def fit(self, X, y=None):
        if isinstance(self.k, (int, np.integer)):
            self.k = min(int(self.k), X.shape[1])
        return super().fit(X, y)


class PreprocessingPipeline:
    """
    Integrated preprocessing pipeline combining multiple preprocessing steps
    with proper cross-validation to prevent data leakage.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the preprocessing pipeline.
        
        Parameters:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.pipeline = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.cv_results = {}
        self.feature_names = []
        
    def create_pipeline(self, imputer_strategy: str = 'mean',
                       outlier_method: str = 'iqr',
                       scaler_type: str = 'standard',
                       feature_engineering: bool = True,
                       feature_selection: bool = True,
                       n_features: int = 20,
                       outlier_strategy: str = 'cap',
                       outlier_threshold: Optional[float] = None) -> Pipeline:
        """
        Create a sklearn pipeline with specified preprocessing steps.
        
        Parameters:
            imputer_strategy (str): Strategy for imputation
            outlier_method (str): Method for outlier detection
            scaler_type (str): Type of scaler to use
            feature_engineering (bool): Whether to apply feature engineering
            feature_selection (bool): Whether to apply feature selection
            n_features (int): Number of features to select
            
        Returns:
            Pipeline: Configured sklearn pipeline
        """
        if imputer_strategy not in {'mean', 'median', 'knn'}:
            raise ValueError("imputer_strategy must be 'mean', 'median', or 'knn'.")
        if outlier_method not in {'iqr', 'zscore'}:
            raise ValueError("outlier_method must be 'iqr' or 'zscore'.")
        if outlier_strategy not in {'cap', 'remove'}:
            raise ValueError("outlier_strategy must be 'cap' or 'remove'.")
        if scaler_type not in {'standard', 'robust', 'minmax'}:
            raise ValueError("scaler_type must be 'standard', 'robust', or 'minmax'.")
        if n_features < 1:
            raise ValueError("n_features must be at least 1.")
        if outlier_threshold is None:
            outlier_threshold = 1.5 if outlier_method == 'iqr' else 3.0
        if not np.isfinite(outlier_threshold) or outlier_threshold < 0:
            raise ValueError("outlier_threshold must be finite and non-negative.")

        imputer = (
            KNNImputer(n_neighbors=5)
            if imputer_strategy == 'knn'
            else SimpleImputer(strategy=imputer_strategy)
        )
        scaler = {
            'standard': StandardScaler,
            'robust': RobustScaler,
            'minmax': MinMaxScaler,
        }[scaler_type]()
        steps = [
            ('outlier_handler', OutlierHandler(
                method=outlier_method,
                strategy=outlier_strategy,
                threshold=outlier_threshold,
            )),
            ('imputer', imputer),
        ]
        if feature_engineering:
            steps.append(('feature_engineer', FeatureEngineer()))
        steps.append(('scaler', scaler))
        if feature_selection:
            steps.append(('feature_selector', AdaptiveSelectKBest(
                score_func=f_classif, k=n_features
            )))
        steps.append((
            'classifier',
            LogisticRegression(max_iter=1000, random_state=self.random_state),
        ))
        
        self.pipeline = Pipeline(steps)
        return self.pipeline
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series, 
                    test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, 
                                                      np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets.
        
        Parameters:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target vector
            test_size (float): Proportion of test set
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of rows.")
        if len(X) < 2:
            raise ValueError("At least two rows are required for a train/test split.")
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1.")
        target = pd.Series(y)
        if target.isna().any():
            raise ValueError("Target values must not contain missing values.")
        class_counts = target.value_counts()
        if len(class_counts) < 2 or class_counts.min() < 2:
            raise ValueError("Stratified splitting requires at least two rows per class.")
        test_rows = int(np.ceil(len(X) * test_size))
        train_rows = len(X) - test_rows
        if test_rows < len(class_counts) or train_rows < len(class_counts):
            raise ValueError("Both train and test sets must be large enough to contain every class.")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        self.feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else []
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def fit_pipeline(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Fit the pipeline on training data.
        
        Parameters:
            X_train: Training feature matrix
            y_train: Training target vector
            
        Returns:
            self: Fitted pipeline
        """
        if self.pipeline is None:
            self.create_pipeline()
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must contain the same number of rows.")
        if len(X_train) == 0:
            raise ValueError("Cannot fit the pipeline on an empty training set.")
        self.pipeline.fit(X_train, y_train)
        return self
    
    def evaluate_pipeline(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the fitted pipeline on test data.
        
        Parameters:
            X_test: Test feature matrix
            y_test: Test target vector
            
        Returns:
            Dict: Evaluation metrics
        """
        if self.pipeline is None or not hasattr(self.pipeline, 'classes_'):
            raise ValueError("Fit the pipeline before evaluating it.")
        if len(X_test) != len(y_test):
            raise ValueError("X_test and y_test must contain the same number of rows.")
        y_pred = self.pipeline.predict(X_test)
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
        }
        if hasattr(self.pipeline, 'predict_proba') and len(self.pipeline.classes_) == 2:
            positive_index = 1
            y_pred_proba = self.pipeline.predict_proba(X_test)[:, positive_index]
            if pd.Series(y_test).nunique() != 2:
                raise ValueError("ROC-AUC evaluation requires both target classes in y_test.")
            metrics['roc_auc'] = float(roc_auc_score(
                y_test, y_pred_proba, labels=self.pipeline.classes_
            ))
        else:
            metrics['roc_auc'] = float('nan')
        
        return metrics
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                      cv: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation on the pipeline.
        
        Parameters:
            X: Feature matrix
            y: Target vector
            cv (int): Number of cross-validation folds
            
        Returns:
            Dict: Cross-validation results
        """
        if cv < 2:
            raise ValueError("cv must be at least 2.")
        target = pd.Series(y)
        if len(X) != len(target):
            raise ValueError("X and y must contain the same number of rows.")
        if target.isna().any():
            raise ValueError("Target values must not contain missing values.")
        class_counts = target.value_counts()
        n_splits = min(cv, int(class_counts.min())) if len(class_counts) >= 2 else 0
        if n_splits < 2:
            raise ValueError("Stratified cross-validation requires at least two rows per class.")
        if self.pipeline is None:
            self.create_pipeline()
        
        cv_splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, 
                                     random_state=self.random_state)
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.pipeline, X, y, cv=cv_splitter, 
                                   scoring='accuracy')
        
        self.cv_results = {
            'scores': cv_scores,
            'mean': float(cv_scores.mean()),
            'std': float(cv_scores.std()),
            'min': float(cv_scores.min()),
            'max': float(cv_scores.max())
        }
        
        return self.cv_results
    
    def compare_configurations(self, X: np.ndarray, y: np.ndarray, 
                              configurations: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare different pipeline configurations.
        
        Parameters:
            X: Feature matrix
            y: Target vector
            configurations (List[Dict]): List of configuration dictionaries
            
        Returns:
            pd.DataFrame: Comparison results
        """
        results = []
        
        for i, config in enumerate(configurations):
            print(f"Evaluating configuration {i+1}/{len(configurations)}...")
            
            # Extract config name and create pipeline config without it
            config_name = config.get('name', f'Config_{i+1}')
            pipeline_config = {k: v for k, v in config.items() if k != 'name'}
            
            # Create pipeline with this configuration
            self.create_pipeline(**pipeline_config)
            
            # Cross-validate
            target = pd.Series(y)
            fold_count = min(5, int(target.value_counts().min()))
            cv_results = self.cross_validate(X, y, cv=fold_count)
            
            result = {
                'config_name': config_name,
                'mean_accuracy': cv_results['mean'],
                'std_accuracy': cv_results['std'],
                **pipeline_config
            }
            results.append(result)
        
        return pd.DataFrame(results)
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray,
                                param_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Optimize pipeline hyperparameters using GridSearchCV.
        
        Parameters:
            X: Feature matrix
            y: Target vector
            param_grid (Dict): Parameter grid for search
            
        Returns:
            Dict: Best parameters and scores
        """
        if self.pipeline is None:
            self.create_pipeline()
        
        grid_search = GridSearchCV(
            self.pipeline, param_grid, cv=5, scoring='accuracy',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X, y)
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def get_learning_curves(self, X: np.ndarray, y: np.ndarray,
                           train_sizes: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        Generate learning curves for the pipeline.
        
        Parameters:
            X: Feature matrix
            y: Target vector
            train_sizes: Training set sizes to evaluate
            
        Returns:
            Dict: Learning curve data
        """
        if self.pipeline is None:
            self.create_pipeline()
        
        if train_sizes is not None:
            train_sizes = np.asarray(train_sizes)
            if train_sizes.ndim != 1 or train_sizes.size == 0:
                raise ValueError("train_sizes must be a non-empty one-dimensional array.")
            if np.any(train_sizes <= 0):
                raise ValueError("train_sizes values must be greater than zero.")
        target = pd.Series(y)
        if len(target) != len(X):
            raise ValueError("X and y must contain the same number of rows.")
        if target.isna().any() or len(target.value_counts()) < 2:
            raise ValueError("Learning curves require valid labels from at least two classes.")
        n_splits = min(5, int(target.value_counts().min()))
        if n_splits < 2:
            raise ValueError("Each class must contain at least two rows for learning curves.")
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        train_sizes_abs, train_scores, val_scores = learning_curve(
            self.pipeline, X, y, train_sizes=train_sizes,
            cv=StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=self.random_state
            ),
            scoring='accuracy', n_jobs=-1,
        )
        
        return {
            'train_sizes': train_sizes_abs,
            'train_scores_mean': train_scores.mean(axis=1),
            'train_scores_std': train_scores.std(axis=1),
            'val_scores_mean': val_scores.mean(axis=1),
            'val_scores_std': val_scores.std(axis=1)
        }
    
    def save_pipeline(self, filepath: str):
        """
        Save the fitted pipeline to disk.
        
        Parameters:
            filepath (str): Path to save the pipeline
        """
        if self.pipeline is None or not hasattr(self.pipeline, 'classes_'):
            raise ValueError("No pipeline to save. Create and fit a pipeline first.")
        
        joblib.dump(self.pipeline, filepath)
        print(f"Pipeline saved to {filepath}")
    
    def load_pipeline(self, filepath: str):
        """
        Load a pipeline from disk.
        
        Parameters:
            filepath (str): Path to load the pipeline from
        """
        self.pipeline = joblib.load(filepath)
        print(f"Pipeline loaded from {filepath}")
    
    def visualize_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   title: str = 'Confusion Matrix',
                                   figsize: Tuple[int, int] = (8, 6)):
        """
        Visualize confusion matrix.
        
        Parameters:
            y_true: True labels
            y_pred: Predicted labels
            title (str): Plot title
            figsize (Tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted Label', fontweight='bold')
        ax.set_ylabel('True Label', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return fig
    
    def visualize_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                           title: str = 'ROC Curve',
                           figsize: Tuple[int, int] = (8, 6)):
        """
        Visualize ROC curve.
        
        Parameters:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            title (str): Plot title
            figsize (Tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
               label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
               label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=20)
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def visualize_learning_curves(self, learning_curve_data: Dict[str, np.ndarray],
                                  title: str = 'Learning Curves',
                                  figsize: Tuple[int, int] = (10, 6)):
        """
        Visualize learning curves.
        
        Parameters:
            learning_curve_data (Dict): Data from get_learning_curves
            title (str): Plot title
            figsize (Tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        train_sizes = learning_curve_data['train_sizes']
        train_mean = learning_curve_data['train_scores_mean']
        train_std = learning_curve_data['train_scores_std']
        val_mean = learning_curve_data['val_scores_mean']
        val_std = learning_curve_data['val_scores_std']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Training scores
        ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                       alpha=0.2, color='blue')
        
        # Validation scores
        ax.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                       alpha=0.2, color='red')
        
        ax.set_xlabel('Training Set Size', fontweight='bold')
        ax.set_ylabel('Accuracy Score', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=20)
        ax.legend(loc='best')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def visualize_cv_scores(self, cv_scores: np.ndarray,
                           title: str = 'Cross-Validation Scores',
                           figsize: Tuple[int, int] = (10, 6)):
        """
        Visualize cross-validation scores as box plot.
        
        Parameters:
            cv_scores: Array of cross-validation scores
            title (str): Plot title
            figsize (Tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        bp = ax.boxplot([cv_scores], labels=['CV Scores'], patch_artist=True,
                       boxprops=dict(facecolor='lightblue'),
                       medianprops=dict(color='red', linewidth=2),
                       widths=0.5)
        
        # Add individual points
        ax.scatter([1] * len(cv_scores), cv_scores, color='darkblue', 
                  alpha=0.5, s=50, zorder=3)
        
        ax.set_ylabel('Accuracy Score', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)
        
        # Add statistics text
        mean_score = cv_scores.mean()
        std_score = cv_scores.std()
        ax.text(1.2, mean_score, f'Mean: {mean_score:.4f}\nStd: {std_score:.4f}',
               verticalalignment='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    def visualize_pipeline_comparison(self, comparison_df: pd.DataFrame,
                                     title: str = 'Pipeline Configuration Comparison',
                                     figsize: Tuple[int, int] = (12, 6)):
        """
        Visualize comparison of different pipeline configurations.
        
        Parameters:
            comparison_df (pd.DataFrame): Results from compare_configurations
            title (str): Plot title
            figsize (Tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        config_names = comparison_df['config_name'].values
        means = comparison_df['mean_accuracy'].values
        stds = comparison_df['std_accuracy'].values
        
        x_pos = np.arange(len(config_names))
        
        bars = ax.bar(x_pos, means, yerr=stds, align='center', alpha=0.7,
                     capsize=10, color='steelblue', edgecolor='black')
        
        # Color the best performing bar differently
        best_idx = np.argmax(means)
        bars[best_idx].set_color('gold')
        bars[best_idx].set_edgecolor('darkgoldenrod')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(config_names, rotation=45, ha='right')
        ax.set_ylabel('Mean Accuracy', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std + 0.01, f'{mean:.4f}', 
                   ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        return fig


def create_sample_dataset_with_issues(n_samples: int = 1000, 
                                      n_features: int = 20,
                                      missing_rate: float = 0.1,
                                      outlier_rate: float = 0.05,
                                      random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create a sample classification dataset with data quality issues.
    
    Parameters:
        n_samples (int): Number of samples
        n_features (int): Number of features
        missing_rate (float): Proportion of missing values
        outlier_rate (float): Proportion of outliers
        random_state (int): Random seed
        
    Returns:
        Tuple: (X, y) feature matrix and target vector
    """
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2.")
    if n_features < 3:
        raise ValueError("n_features must be at least 3.")
    if not np.isfinite(missing_rate) or not 0 <= missing_rate <= 1:
        raise ValueError("missing_rate must be between 0 and 1.")
    if not np.isfinite(outlier_rate) or not 0 <= outlier_rate <= 1:
        raise ValueError("outlier_rate must be between 0 and 1.")

    np.random.seed(random_state)
    
    # Create classification dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.7),
        n_redundant=int(n_features * 0.2),
        n_classes=2,
        class_sep=1.0,
        random_state=random_state
    )
    
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    
    # Inject missing values
    n_missing = int(n_samples * n_features * missing_rate)
    missing_indices = np.random.choice(n_samples * n_features, size=n_missing, replace=False)
    for idx in missing_indices:
        row = idx // n_features
        col = idx % n_features
        X_df.iloc[row, col] = np.nan
    
    # Inject outliers
    n_outliers = int(n_samples * outlier_rate)
    outlier_indices = np.random.choice(n_samples, size=n_outliers, replace=False)
    for idx in outlier_indices:
        col = np.random.randint(0, n_features)
        X_df.iloc[idx, col] = X_df.iloc[idx, col] * np.random.uniform(5, 10)
    
    return X_df, pd.Series(y, name='target')


def main():
    """
    Main function demonstrating the complete Task 5 workflow.
    
    Returns:
        PreprocessingPipeline: Configured pipeline object
    """
    print("=" * 80)
    print("Task 5: Integrated Data Preprocessing Pipeline with Cross-Validation")
    print("=" * 80)
    print()
    
    # Create sample dataset
    print("Creating sample dataset with quality issues...")
    X, y = create_sample_dataset_with_issues(
        n_samples=1000, n_features=20, missing_rate=0.1, outlier_rate=0.05
    )
    print(f"Dataset created: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Missing values: {X.isna().sum().sum()}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    print()
    
    # Initialize pipeline
    pipeline_manager = PreprocessingPipeline(random_state=42)
    
    # Prepare data
    print("1. DATA SPLITTING")
    print("-" * 80)
    X_train, X_test, y_train, y_test = pipeline_manager.prepare_data(X, y, test_size=0.2)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print()
    
    # Create and evaluate baseline pipeline
    print("2. BASELINE PIPELINE (No Feature Engineering)")
    print("-" * 80)
    pipeline_manager.create_pipeline(
        imputer_strategy='mean',
        outlier_method='iqr',
        scaler_type='standard',
        feature_engineering=False,
        feature_selection=False
    )
    
    baseline_cv = pipeline_manager.cross_validate(X_train, y_train, cv=5)
    print(f"Cross-validation accuracy: {baseline_cv['mean']:.4f} (±{baseline_cv['std']:.4f})")
    print()
    
    # Fit baseline and evaluate on test set
    pipeline_manager.fit_pipeline(X_train, y_train)
    baseline_test = pipeline_manager.evaluate_pipeline(X_test, y_test)
    print(f"Test set accuracy: {baseline_test['accuracy']:.4f}")
    print(f"Test set ROC-AUC: {baseline_test['roc_auc']:.4f}")
    print()
    
    # Compare different configurations
    print("3. COMPARING PIPELINE CONFIGURATIONS")
    print("-" * 80)
    
    configurations = [
        {
            'name': 'Baseline',
            'imputer_strategy': 'mean',
            'outlier_method': 'iqr',
            'scaler_type': 'standard',
            'feature_engineering': False,
            'feature_selection': False
        },
        {
            'name': 'With Feature Eng',
            'imputer_strategy': 'mean',
            'outlier_method': 'iqr',
            'scaler_type': 'standard',
            'feature_engineering': True,
            'feature_selection': False
        },
        {
            'name': 'With Selection',
            'imputer_strategy': 'mean',
            'outlier_method': 'iqr',
            'scaler_type': 'standard',
            'feature_engineering': False,
            'feature_selection': True,
            'n_features': 15
        },
        {
            'name': 'Full Pipeline',
            'imputer_strategy': 'knn',
            'outlier_method': 'iqr',
            'scaler_type': 'robust',
            'feature_engineering': True,
            'feature_selection': True,
            'n_features': 20
        }
    ]
    
    comparison_df = pipeline_manager.compare_configurations(X_train, y_train, configurations)
    print("\nConfiguration Comparison:")
    print(comparison_df[['config_name', 'mean_accuracy', 'std_accuracy']].to_string(index=False))
    print()
    
    # Generate learning curves for best configuration
    print("4. GENERATING LEARNING CURVES")
    print("-" * 80)
    best_config = comparison_df.loc[comparison_df['mean_accuracy'].idxmax()]
    print(f"Best configuration: {best_config['config_name']}")
    
    # Recreate pipeline with best config
    pipeline_manager.create_pipeline(
        imputer_strategy=best_config.get('imputer_strategy', 'mean'),
        outlier_method=best_config.get('outlier_method', 'iqr'),
        scaler_type=best_config.get('scaler_type', 'standard'),
        feature_engineering=best_config.get('feature_engineering', False),
        feature_selection=best_config.get('feature_selection', False),
        n_features=best_config.get('n_features', 20)
    )
    
    lc_data = pipeline_manager.get_learning_curves(X_train, y_train)
    print("Learning curves generated")
    print()
    
    # Fit final pipeline and evaluate
    print("5. FINAL PIPELINE EVALUATION")
    print("-" * 80)
    pipeline_manager.fit_pipeline(X_train, y_train)
    final_metrics = pipeline_manager.evaluate_pipeline(X_test, y_test)
    print(f"Final test accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Final test ROC-AUC: {final_metrics['roc_auc']:.4f}")
    print()
    
    # Generate predictions for visualizations
    y_pred = pipeline_manager.pipeline.predict(X_test)
    y_pred_proba = pipeline_manager.pipeline.predict_proba(X_test)[:, 1]
    
    # Create visualizations
    print("6. GENERATING VISUALIZATIONS")
    print("-" * 80)
    
    print("  Creating confusion matrix...")
    pipeline_manager.visualize_confusion_matrix(y_test, y_pred, 
                                               title='Confusion Matrix - Final Pipeline')
    plt.savefig('task5_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("  Saved: task5_confusion_matrix.png")
    
    print("  Creating ROC curve...")
    pipeline_manager.visualize_roc_curve(y_test, y_pred_proba,
                                        title='ROC Curve - Final Pipeline')
    plt.savefig('task5_roc_curve.png', dpi=300, bbox_inches='tight')
    print("  Saved: task5_roc_curve.png")
    
    print("  Creating learning curves...")
    pipeline_manager.visualize_learning_curves(lc_data,
                                              title='Learning Curves - Best Configuration')
    plt.savefig('task5_learning_curves.png', dpi=300, bbox_inches='tight')
    print("  Saved: task5_learning_curves.png")
    
    print("  Creating CV scores visualization...")
    cv_scores = baseline_cv['scores']
    pipeline_manager.visualize_cv_scores(cv_scores,
                                        title='Cross-Validation Score Distribution')
    plt.savefig('task5_cv_scores.png', dpi=300, bbox_inches='tight')
    print("  Saved: task5_cv_scores.png")
    
    print("  Creating pipeline comparison...")
    pipeline_manager.visualize_pipeline_comparison(comparison_df,
                                                   title='Pipeline Configuration Performance')
    plt.savefig('task5_pipeline_comparison.png', dpi=300, bbox_inches='tight')
    print("  Saved: task5_pipeline_comparison.png")
    print()
    
    # Save pipeline
    print("7. SAVING PIPELINE")
    print("-" * 80)
    pipeline_manager.save_pipeline('task5_preprocessing_pipeline.pkl')
    print()
    
    print("=" * 80)
    print("Task 5 completed successfully!")
    print("=" * 80)
    print()
    print("Key Findings:")
    print(f"  - Baseline accuracy: {baseline_test['accuracy']:.4f}")
    print(f"  - Final accuracy: {final_metrics['accuracy']:.4f}")
    print(f"  - Improvement: {(final_metrics['accuracy'] - baseline_test['accuracy']):.4f}")
    print(f"  - Best configuration: {best_config['config_name']}")
    
    return pipeline_manager


if __name__ == "__main__":
    pipeline = main()
    plt.show()
