"""
Task 6: Real-World Case Study - Complete Data Preprocessing Challenge

This module implements a complete end-to-end data preprocessing workflow
for a complex synthetic dataset with multiple data quality issues.
Demonstrates best practices for EDA, preprocessing, and model evaluation.

Author: MLDS Course
Date: December 2025
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.utils import resample
from typing import Tuple, Dict, List, Any
from datetime import datetime, timedelta
import warnings
import json

warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Comprehensive data preprocessor for real-world case study.
    Handles EDA, data quality assessment, and preprocessing.
    """
    
    def __init__(self, data: pd.DataFrame, target_col: str):
        """
        Initialize the data preprocessor.
        
        Parameters:
            data (pd.DataFrame): Input dataset
            target_col (str): Name of target column
        """
        self.original_data = data.copy()
        self.data = data.copy()
        self.target_col = target_col
        self.preprocessed_data = None
        self.quality_report = {}
        self.preprocessing_steps = []

    def _feature_columns(self) -> Tuple[List[str], List[str], List[str]]:
        feature_data = self.data.drop(
            columns=[self.target_col], errors='ignore'
        )
        datetime_cols = feature_data.select_dtypes(
            include=['datetime', 'datetimetz']
        ).columns.tolist()
        for column in feature_data.select_dtypes(include=['object']).columns:
            non_missing = feature_data[column].dropna()
            if not non_missing.empty and non_missing.map(
                lambda value: isinstance(value, (datetime, pd.Timestamp, np.datetime64))
            ).all():
                datetime_cols.append(column)
            elif any(token in str(column).lower() for token in ('date', 'time')):
                parsed = pd.to_datetime(non_missing, errors='coerce')
                if not non_missing.empty and parsed.notna().mean() >= 0.9:
                    datetime_cols.append(column)
        numeric_cols = feature_data.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        categorical_cols = [
            column for column in feature_data.columns
            if column not in numeric_cols and column not in datetime_cols
        ]
        return numeric_cols, categorical_cols, datetime_cols

    def _ensure_quality_report(self) -> None:
        if not self.quality_report:
            self.assess_data_quality()
        
    def assess_data_quality(self) -> Dict[str, Any]:
        """
        Perform comprehensive data quality assessment.
        
        Returns:
            Dict: Quality assessment report
        """
        report = {
            'shape': self.data.shape,
            'total_samples': len(self.data),
            'total_features': len(self.data.columns),
            'memory_usage_mb': float(self.data.memory_usage(deep=True).sum() / 1024**2),
            'duplicates': int(self.data.duplicated().sum()),
            'missing_values': {},
            'data_types': {},
            'numeric_features': [],
            'categorical_features': [],
            'datetime_features': [],
            'target_distribution': {}
        }
        
        missing_by_column = self.data.isnull().sum()
        missing_by_column = missing_by_column[missing_by_column > 0]
        report['missing_values'] = {
            'total': int(self.data.isnull().sum().sum()),
            'by_column': missing_by_column.astype(int).to_dict(),
            'percentage_by_column': (
                (missing_by_column / len(self.data) * 100).to_dict()
                if len(self.data) else {}
            ),
        }

        numeric_cols, categorical_cols, datetime_cols = self._feature_columns()
        report['numeric_features'] = numeric_cols
        report['categorical_features'] = categorical_cols
        report['datetime_features'] = datetime_cols
        report['data_types'] = self.data.dtypes.astype(str).to_dict()
        if self.target_col in self.data.columns:
            target_counts = self.data[self.target_col].value_counts(dropna=False)
            report['target_distribution'] = {
                str(class_value): int(count)
                for class_value, count in target_counts.items()
            }
            nonzero_counts = target_counts[target_counts > 0]
            report['class_balance'] = (
                float(nonzero_counts.min() / nonzero_counts.max())
                if len(nonzero_counts) > 1 else 1.0
            )
        self.quality_report = report
        return report

    def print_quality_summary(self):
        """Print the current data quality report using the README's method name."""
        self._ensure_quality_report()
        return self.print_quality_report()
    
    def print_quality_report(self):
        """
        Print formatted data quality report.
        """
        self._ensure_quality_report()
        print("=" * 80)
        print("DATA QUALITY ASSESSMENT REPORT")
        print("=" * 80)
        print()
        
        print(f"Dataset Shape: {self.quality_report['shape']}")
        print(f"Total Samples: {self.quality_report['total_samples']:,}")
        print(f"Total Features: {self.quality_report['total_features']}")
        print(f"Memory Usage: {self.quality_report['memory_usage_mb']:.2f} MB")
        print(f"Duplicate Rows: {self.quality_report['duplicates']}")
        print()
        
        print("MISSING VALUES:")
        print(f"  Total Missing: {self.quality_report['missing_values']['total']:,}")
        if self.quality_report['missing_values']['by_column']:
            print("  Columns with Missing Values:")
            for col, count in self.quality_report['missing_values']['by_column'].items():
                pct = self.quality_report['missing_values']['percentage_by_column'][col]
                print(f"    {col}: {count} ({pct:.2f}%)")
        print()
        
        print("FEATURE TYPES:")
        print(f"  Numeric: {len(self.quality_report['numeric_features'])}")
        print(f"  Categorical: {len(self.quality_report['categorical_features'])}")
        print(f"  Datetime: {len(self.quality_report['datetime_features'])}")
        print()
        
        if self.quality_report['target_distribution']:
            print("TARGET DISTRIBUTION:")
            for cls, count in self.quality_report['target_distribution'].items():
                pct = (count / self.quality_report['total_samples']) * 100
                print(f"  Class {cls}: {count} ({pct:.2f}%)")
            print(f"  Class Balance Ratio: {self.quality_report['class_balance']:.3f}")
        print()
    
    def handle_missing_values(self, strategy: str = 'auto') -> pd.DataFrame:
        """
        Handle missing values with specified strategy.
        
        Parameters:
            strategy (str): Imputation strategy
            
        Returns:
            pd.DataFrame: Data with missing values handled
        """
        self._ensure_quality_report()
        if strategy not in {'auto', 'simple', 'knn', 'iterative'}:
            raise ValueError("strategy must be 'auto', 'simple', 'knn', or 'iterative'.")
        data = self.data.copy()
        if self.target_col in data.columns and data[self.target_col].isna().any():
            raise ValueError("Target column contains missing values; impute or remove targets separately.")
        numeric_cols, categorical_cols, datetime_cols = self._feature_columns()
        numeric_cols = [
            column for column in numeric_cols if column != self.target_col
        ]
        data.loc[:, numeric_cols] = data[numeric_cols].replace(
            [np.inf, -np.inf], np.nan
        )
        numeric_cols = [
            column for column in numeric_cols
            if data[column].notna().any()
        ]
        all_missing_numeric = [
            column for column in data.select_dtypes(include=[np.number]).columns
            if column != self.target_col and data[column].isna().all()
        ]
        data.loc[:, all_missing_numeric] = 0.0

        if strategy in {'auto', 'simple'}:
            if numeric_cols:
                imputer = SimpleImputer(strategy='median')
                data.loc[:, numeric_cols] = imputer.fit_transform(data[numeric_cols])
            for column in categorical_cols:
                missing = data[column].isna()
                if missing.any():
                    non_missing = data[column].dropna()
                    fill_value = non_missing.mode().iloc[0] if not non_missing.empty else 'Missing'
                    data.loc[missing, column] = fill_value
            for column in datetime_cols:
                if data[column].isna().any():
                    non_missing = data[column].dropna()
                    if not non_missing.empty:
                        data[column] = data[column].fillna(non_missing.mode().iloc[0])
                    else:
                        data[column] = pd.Timestamp('1970-01-01')

        elif strategy == 'knn':
            if numeric_cols:
                imputer = KNNImputer(n_neighbors=min(5, max(1, len(data) - 1)))
                data.loc[:, numeric_cols] = imputer.fit_transform(data[numeric_cols])
            for column in categorical_cols:
                missing = data[column].isna()
                if missing.any():
                    non_missing = data[column].dropna()
                    fill_value = non_missing.mode().iloc[0] if not non_missing.empty else 'Missing'
                    data.loc[missing, column] = fill_value
            for column in datetime_cols:
                if data[column].isna().any():
                    non_missing = data[column].dropna()
                    if not non_missing.empty:
                        data[column] = data[column].fillna(non_missing.mode().iloc[0])
                    else:
                        data[column] = pd.Timestamp('1970-01-01')

        elif strategy == 'iterative':
            if numeric_cols:
                imputer = IterativeImputer(max_iter=10, random_state=42)
                data.loc[:, numeric_cols] = imputer.fit_transform(data[numeric_cols])
            for column in categorical_cols:
                missing = data[column].isna()
                if missing.any():
                    non_missing = data[column].dropna()
                    fill_value = non_missing.mode().iloc[0] if not non_missing.empty else 'Missing'
                    data.loc[missing, column] = fill_value
            for column in datetime_cols:
                if data[column].isna().any():
                    non_missing = data[column].dropna()
                    if not non_missing.empty:
                        data[column] = data[column].fillna(non_missing.mode().iloc[0])
                    else:
                        data[column] = pd.Timestamp('1970-01-01')

        self.data = data
        self.preprocessing_steps.append(f"Missing values handled: {strategy}")
        return data
    
    def detect_and_handle_outliers(self, method: str = 'iqr', 
                                   action: str = 'cap') -> pd.DataFrame:
        """
        Detect and handle outliers.
        
        Parameters:
            method (str): Detection method
            action (str): Action to take on outliers
            
        Returns:
            pd.DataFrame: Data with outliers handled
        """
        self._ensure_quality_report()
        if method not in {'iqr', 'zscore'}:
            raise ValueError("method must be 'iqr' or 'zscore'.")
        if action not in {'cap', 'remove'}:
            raise ValueError("action must be 'cap' or 'remove'.")
        data = self.data.copy()
        numeric_cols = [
            column for column in data.select_dtypes(include=[np.number]).columns
            if column != self.target_col
        ]
        outlier_mask = pd.DataFrame(False, index=data.index, columns=numeric_cols)
        if method == 'iqr':
            for column in numeric_cols:
                values = data[column].replace([np.inf, -np.inf], np.nan)
                q1, q3 = values.quantile([0.25, 0.75])
                spread = q3 - q1
                lower, upper = q1 - 1.5 * spread, q3 + 1.5 * spread
                outlier_mask[column] = (
                    data[column].lt(lower) | data[column].gt(upper)
                    | ~np.isfinite(data[column])
                ).fillna(False)
        else:
            for column in numeric_cols:
                values = data[column].replace([np.inf, -np.inf], np.nan)
                mean, std = values.mean(), values.std(ddof=0)
                if pd.isna(std) or std == 0:
                    outlier_mask[column] = ~np.isfinite(data[column])
                else:
                    outlier_mask[column] = (
                        ((data[column] - mean).abs() > 3.0 * std)
                        | ~np.isfinite(data[column])
                    ).fillna(False)

        outliers_count = int(outlier_mask.any(axis=1).sum()) if numeric_cols else 0
        if action == 'cap':
            for column in numeric_cols:
                values = data[column].replace([np.inf, -np.inf], np.nan)
                if method == 'iqr':
                    q1, q3 = values.quantile([0.25, 0.75])
                    spread = q3 - q1
                    lower, upper = q1 - 1.5 * spread, q3 + 1.5 * spread
                else:
                    mean, std = values.mean(), values.std(ddof=0)
                    lower, upper = mean - 3.0 * std, mean + 3.0 * std
                if pd.notna(lower) and pd.notna(upper):
                    data[column] = values.clip(lower=lower, upper=upper)
        elif action == 'remove' and numeric_cols:
            data = data.loc[~outlier_mask.any(axis=1)].copy()

        self.data = data
        self.preprocessing_steps.append(
            f"Outliers handled: {method} method, {action} action, {outliers_count} outliers"
        )
        return data
    
    def encode_categorical_features(self, method: str = 'auto') -> pd.DataFrame:
        """
        Encode categorical features.
        
        Parameters:
            method (str): Encoding method
            
        Returns:
            pd.DataFrame: Data with encoded categorical features
        """
        if method not in {'auto', 'onehot', 'label'}:
            raise ValueError("method must be 'auto', 'onehot', or 'label'.")
        data = self.data.copy()
        _, _, datetime_cols = self._feature_columns()
        for column in datetime_cols:
            if column == self.target_col:
                continue
            dates = pd.to_datetime(data[column], errors='coerce')
            data[f'{column}_year'] = dates.dt.year
            data[f'{column}_month'] = dates.dt.month
            data[f'{column}_day'] = dates.dt.day
            data[f'{column}_dayofweek'] = dates.dt.dayofweek
            data = data.drop(columns=[column])

        categorical_cols = [
            column for column in data.select_dtypes(include=['object', 'category', 'bool']).columns
            if column != self.target_col
        ]
        for column in categorical_cols:
            values = data[column].astype('object').where(data[column].notna(), 'Missing').astype(str)
            cardinality = values.nunique()
            selected_method = (
                'onehot' if method == 'auto' and cardinality <= 10
                else 'label' if method == 'auto'
                else method
            )
            if selected_method == 'onehot':
                encoded = pd.get_dummies(values, prefix=column, dtype=int)
                data = data.drop(columns=[column]).join(encoded)
            else:
                encoder = LabelEncoder()
                data[f'{column}_encoded'] = encoder.fit_transform(values)
                data = data.drop(columns=[column])

        self.data = data
        self.preprocessing_steps.append(f"Categorical encoding: {method}, datetime features extracted")
        return data
    
    def engineer_features(self, domain_features: bool = True) -> pd.DataFrame:
        """
        Engineer new features.
        
        Parameters:
            domain_features (bool): Whether to create domain-specific features
            
        Returns:
            pd.DataFrame: Data with engineered features
        """
        data = self.data.copy()
        numeric_cols = [
            col for col in data.select_dtypes(include=[np.number]).columns
            if col != self.target_col
        ]
        
        new_features = []
        
        # Statistical features
        if len(numeric_cols) >= 2:
            numeric_data = data[numeric_cols]
            data['mean_all_features'] = numeric_data.mean(axis=1)
            data['std_all_features'] = numeric_data.std(axis=1)
            data['min_all_features'] = numeric_data.min(axis=1)
            data['max_all_features'] = numeric_data.max(axis=1)
            new_features.extend([
                'mean_all_features', 'std_all_features',
                'min_all_features', 'max_all_features',
            ])
        
        # Interaction features (limited to avoid explosion)
        if domain_features and len(numeric_cols) >= 2:
            for first, second in zip(numeric_cols[:3], numeric_cols[1:4]):
                product_name = f'{first}_x_{second}'
                ratio_name = f'{first}_ratio_{second}'
                data[product_name] = data[first] * data[second]
                denominator = data[second].replace(0, np.nan)
                data[ratio_name] = (data[first] / denominator).replace(
                    [np.inf, -np.inf], np.nan
                ).fillna(0.0)
                new_features.extend([product_name, ratio_name])
        
        self.data = data
        self.preprocessing_steps.append(f"Feature engineering: {len(new_features)} new features created")
        return data
    
    def scale_features(self, method: str = 'standard') -> pd.DataFrame:
        """
        Scale numerical features.
        
        Parameters:
            method (str): Scaling method
            
        Returns:
            pd.DataFrame: Data with scaled features
        """
        if method not in {'standard', 'robust'}:
            raise ValueError("method must be 'standard' or 'robust'.")
        data = self.data.copy()
        numeric_cols = [
            column for column in data.select_dtypes(include=[np.number]).columns
            if column != self.target_col
        ]
        if numeric_cols:
            finite_data = data[numeric_cols].replace([np.inf, -np.inf], np.nan)
            empty_columns = finite_data.columns[finite_data.isna().all()].tolist()
            finite_data.loc[:, empty_columns] = 0.0
            imputer = SimpleImputer(strategy='median')
            imputed = imputer.fit_transform(finite_data)
            scaler = StandardScaler() if method == 'standard' else RobustScaler()
            data.loc[:, numeric_cols] = scaler.fit_transform(imputed)
        
        self.data = data
        self.preprocessing_steps.append(f"Feature scaling: {method}")
        return data
    
    def select_features(self, method: str = 'kbest', n_features: int = 20) -> pd.DataFrame:
        """
        Select most important features.
        
        Parameters:
            method (str): Feature selection method
            n_features (int): Number of features to select
            
        Returns:
            pd.DataFrame: Data with selected features
        """
        data = self.data.copy()
        
        if self.target_col not in data.columns:
            return data
        if n_features < 1:
            raise ValueError("n_features must be at least 1.")
        if method not in {'kbest', 'mutual_info'}:
            raise ValueError("method must be 'kbest' or 'mutual_info'.")

        X = data.drop(columns=[self.target_col])
        y = data[self.target_col]
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("Feature selection requires numeric feature columns.")
        X = X[numeric_cols].replace([np.inf, -np.inf], np.nan)
        nonempty_cols = X.columns[X.notna().any()].tolist()
        if not nonempty_cols:
            raise ValueError("Feature selection requires at least one non-empty feature.")
        X = X[nonempty_cols].fillna(X[nonempty_cols].median()).fillna(0.0)
        valid_target = y.notna()
        X = X.loc[valid_target]
        y = y.loc[valid_target]
        if y.nunique() < 2:
            raise ValueError("Feature selection requires at least two target classes.")
        n_features = min(n_features, X.shape[1])

        if method == 'kbest':
            selector = SelectKBest(score_func=f_classif, k=n_features)
        elif method == 'mutual_info':
            selector = SelectKBest(
                score_func=lambda features, target: mutual_info_classif(
                    features, target, random_state=42
                ),
                k=n_features,
            )
        
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        data = data.loc[valid_target, selected_features].copy()
        data[self.target_col] = y
        
        self.data = data
        self.preprocessing_steps.append(
            f"Feature selection: {method}, {len(selected_features)} features selected"
        )
        return data
    
    def balance_classes(self, method: str = 'undersample') -> pd.DataFrame:
        """
        Balance class distribution.
        
        Parameters:
            method (str): Balancing method
            
        Returns:
            pd.DataFrame: Balanced dataset
        """
        data = self.data.copy()
        
        if self.target_col not in data.columns:
            return data
        if method not in {'undersample', 'oversample'}:
            raise ValueError("method must be 'undersample' or 'oversample'.")
        if data[self.target_col].isna().any():
            raise ValueError("Class balancing requires non-missing target values.")
        class_counts = data[self.target_col].value_counts()
        if len(class_counts) < 2:
            raise ValueError("Class balancing requires at least two target classes.")
        target_count = int(class_counts.min() if method == 'undersample' else class_counts.max())
        class_frames = [
            data.loc[data[self.target_col] == class_value]
            for class_value in class_counts.index
        ]

        if method == 'undersample':
            balanced_frames = [
                resample(frame, replace=False, n_samples=target_count, random_state=42)
                for frame in class_frames
            ]
        else:
            balanced_frames = [
                resample(frame, replace=True, n_samples=target_count, random_state=42)
                for frame in class_frames
            ]
        data = pd.concat(balanced_frames).sample(frac=1, random_state=42)
        
        self.data = data
        self.preprocessing_steps.append(f"Class balancing: {method}")
        return data
    
    def preprocess_pipeline(self, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Execute complete preprocessing pipeline.
        
        Parameters:
            config (Dict): Configuration for preprocessing steps
            
        Returns:
            pd.DataFrame: Fully preprocessed data
        """
        if not isinstance(config, dict):
            raise TypeError("config must be a dictionary.")
        self._ensure_quality_report()
        print("Starting preprocessing pipeline...")
        print("-" * 80)
        
        # Step 1: Handle missing values
        if config.get('handle_missing', True):
            print("Step 1: Handling missing values...")
            self.data = self.handle_missing_values(config.get('missing_strategy', 'auto'))
        
        # Step 2: Handle outliers
        if config.get('handle_outliers', True):
            print("Step 2: Handling outliers...")
            self.data = self.detect_and_handle_outliers(
                method=config.get('outlier_method', 'iqr'),
                action=config.get('outlier_action', 'cap')
            )
        
        # Step 3: Encode categorical features
        if config.get('encode_categorical', True):
            print("Step 3: Encoding categorical features...")
            self.data = self.encode_categorical_features(config.get('encoding_method', 'auto'))
        
        # Step 4: Engineer features
        if config.get('engineer_features', True):
            print("Step 4: Engineering features...")
            self.data = self.engineer_features()
        
        # Step 5: Scale features
        if config.get('scale_features', True):
            print("Step 5: Scaling features...")
            self.data = self.scale_features(config.get('scaling_method', 'standard'))
        
        # Step 6: Select features
        if config.get('select_features', False):
            print("Step 6: Selecting features...")
            self.data = self.select_features(
                method=config.get(
                    'feature_selection_method',
                    config.get('selection_method', 'kbest'),
                ),
                n_features=config.get('n_features', 20)
            )
        
        # Step 7: Balance classes
        if config.get('balance_classes', False):
            print("Step 7: Balancing classes...")
            self.data = self.balance_classes(config.get('balance_method', 'undersample'))
        
        print("-" * 80)
        print(f"Preprocessing complete! Final shape: {self.data.shape}")
        print()
        
        self.preprocessed_data = self.data.copy()
        return self.data
    
    
    def visualize_eda(self, figsize: Tuple[int, int] = (20, 15)):
        """
        Create comprehensive EDA visualizations.
        
        Parameters:
            figsize (Tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        self._ensure_quality_report()
        numeric_cols, categorical_cols, _ = self._feature_columns()
        
        # Create subplot grid
        n_rows = 3
        n_cols = 3
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(n_rows, n_cols, hspace=0.3, wspace=0.3)
        
        # Numeric distributions
        for i, col in enumerate(numeric_cols[:6]):
            row = i // 3
            col_idx = i % 3
            ax = fig.add_subplot(gs[row, col_idx])
            self.original_data[col].hist(bins=30, ax=ax, edgecolor='black', alpha=0.7)
            ax.set_title(f'Distribution: {col}', fontweight='bold')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.grid(alpha=0.3)
        
        # Categorical distributions
        for i, col in enumerate(categorical_cols[:3]):
            if i + 6 < 9:
                ax = fig.add_subplot(gs[2, i])
                value_counts = self.original_data[col].value_counts().head(10)
                value_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
                ax.set_title(f'Distribution: {col}', fontweight='bold')
                ax.set_xlabel('Category')
                ax.set_ylabel('Count')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(alpha=0.3, axis='y')
        
        plt.suptitle('Exploratory Data Analysis - Feature Distributions', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        return fig
    
    def visualize_correlations(self, figsize: Tuple[int, int] = (12, 10)):
        """
        Visualize feature correlations.
        
        Parameters:
            figsize (Tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        numeric_data = self.original_data.select_dtypes(include=[np.number])
        numeric_data = numeric_data.replace([np.inf, -np.inf], np.nan)
        numeric_data = numeric_data.dropna(axis=1, how='all')
        
        # Limit to reasonable number of features
        if numeric_data.shape[1] > 20:
            numeric_data = numeric_data.iloc[:, :20]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if numeric_data.empty:
            ax.text(0.5, 0.5, 'No numeric features available', ha='center', va='center')
            ax.axis('off')
        else:
            corr_matrix = numeric_data.corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation'})
        
        ax.set_title('Feature Correlation Matrix', fontweight='bold', pad=20, fontsize=14)
        plt.tight_layout()
        
        return fig
    
    def visualize_missing_patterns(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Visualize missing value patterns.
        
        Parameters:
            figsize (Tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Missing values heatmap
        missing_matrix = self.original_data.isnull()
        
        # Sample if too many rows
        if len(missing_matrix) > 500:
            missing_matrix = missing_matrix.sample(500, random_state=42)
        
        sns.heatmap(missing_matrix, cbar=True, yticklabels=False, 
                   cmap='viridis', ax=ax1)
        ax1.set_title('Missing Values Heatmap\n(Yellow = Missing)', 
                     fontweight='bold', pad=10)
        ax1.set_xlabel('Features')
        ax1.set_ylabel('Samples')
        
        # Missing values bar plot
        missing_counts = self.original_data.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
        
        if len(missing_counts) > 0:
            missing_counts.plot(kind='barh', ax=ax2, color='coral', edgecolor='black')
            ax2.set_title('Missing Values by Feature', fontweight='bold', pad=10)
            ax2.set_xlabel('Count')
            ax2.set_ylabel('Feature')
            ax2.grid(alpha=0.3, axis='x')
        else:
            ax2.text(0.5, 0.5, 'No Missing Values', 
                    ha='center', va='center', fontsize=14)
            ax2.set_title('Missing Values by Feature', fontweight='bold', pad=10)
        
        plt.tight_layout()
        return fig
    
    
    def visualize_pca(self, n_components: int = 2, figsize: Tuple[int, int] = (10, 8)):
        """
        Visualize PCA dimensionality reduction.
        
        Parameters:
            n_components (int): Number of PCA components
            figsize (Tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        if self.target_col not in self.data.columns:
            raise ValueError("Target column not found")
        if n_components < 2:
            raise ValueError("n_components must be at least 2 for a 2D PCA plot.")

        X = self.data.drop(columns=[self.target_col]).select_dtypes(include=[np.number])
        y = self.data[self.target_col]
        X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='all')
        if X.empty:
            raise ValueError("PCA requires at least one non-empty numeric feature.")
        X = X.fillna(X.median()).fillna(0.0)
        actual_components = min(n_components, X.shape[1], X.shape[0])
        pca = PCA(n_components=actual_components)
        transformed = pca.fit_transform(X)
        if actual_components == 1:
            X_pca = np.column_stack([transformed[:, 0], np.zeros(len(transformed))])
            variance = np.array([pca.explained_variance_ratio_[0], 0.0])
        else:
            X_pca = transformed[:, :2]
            variance = pca.explained_variance_ratio_[:2]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot by class
        for class_val in y.unique():
            mask = y == class_val
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                      label=f'Class {class_val}', alpha=0.6, s=50)
        
        ax.set_xlabel(f'PC1 ({variance[0]:.2%} variance)', 
                     fontweight='bold')
        ax.set_ylabel(f'PC2 ({variance[1]:.2%} variance)', 
                     fontweight='bold')
        ax.set_title('PCA - Feature Space Visualization', fontweight='bold', 
                    pad=20, fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_report(self, filepath: str = 'preprocessing_report.json'):
        """
        Generate comprehensive preprocessing report.
        
        Parameters:
            filepath (str): Path to save report
        """
        report = {
            'data_quality': self.quality_report,
            'preprocessing_steps': self.preprocessing_steps,
            'final_shape': self.data.shape,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Report saved to {filepath}")
        return report


def create_complex_synthetic_dataset(n_samples: int = 2000,
                                     n_features: int = 25,
                                     missing_rate: float = 0.15,
                                     outlier_rate: float = 0.08,
                                     imbalance_ratio: float = 0.3,
                                     random_state: int = 42) -> pd.DataFrame:
    """
    Create a complex synthetic dataset with multiple data quality issues.
    
    Parameters:
        n_samples (int): Number of samples
        n_features (int): Number of features
        missing_rate (float): Proportion of missing values
        outlier_rate (float): Proportion of outliers
        imbalance_ratio (float): Class imbalance ratio
        random_state (int): Random seed
        
    Returns:
        pd.DataFrame: Complex synthetic dataset
    """
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2.")
    if n_features < 3:
        raise ValueError("n_features must be at least 3.")
    if not np.isfinite(missing_rate) or not 0 <= missing_rate <= 1:
        raise ValueError("missing_rate must be between 0 and 1.")
    if not np.isfinite(outlier_rate) or not 0 <= outlier_rate <= 1:
        raise ValueError("outlier_rate must be between 0 and 1.")
    if not np.isfinite(imbalance_ratio) or not 0 < imbalance_ratio < 1:
        raise ValueError("imbalance_ratio must be between 0 and 1.")

    np.random.seed(random_state)
    
    # Generate numerical features
    numeric_features = {}
    for i in range(n_features):
        if i % 3 == 0:
            # Normal distribution
            numeric_features[f'feature_{i}'] = np.random.randn(n_samples) * 10 + 50
        elif i % 3 == 1:
            # Exponential distribution
            numeric_features[f'feature_{i}'] = np.random.exponential(scale=20, size=n_samples)
        else:
            # Uniform distribution
            numeric_features[f'feature_{i}'] = np.random.uniform(0, 100, n_samples)
    
    df = pd.DataFrame(numeric_features)
    
    # Add categorical features
    categories = ['A', 'B', 'C', 'D', 'E']
    df['category_1'] = np.random.choice(categories, size=n_samples)
    df['category_2'] = np.random.choice(['Type1', 'Type2', 'Type3'], size=n_samples)
    df['category_3'] = np.random.choice(['Low', 'Medium', 'High', 'Very High'], size=n_samples)
    
    # Add datetime feature
    start_date = datetime(2020, 1, 1)
    df['date'] = [start_date + timedelta(days=int(x)) 
                  for x in np.random.uniform(0, 1000, n_samples)]
    
    # Create target variable with imbalance
    # Based on some features
    target_score = (
        df['feature_0'] * 0.3 + 
        df['feature_1'] * 0.2 + 
        df['feature_2'] * 0.1 +
        np.random.randn(n_samples) * 10
    )
    
    threshold = np.percentile(target_score, imbalance_ratio * 100)
    df['target'] = (target_score > threshold).astype(int)
    
    # Inject missing values
    n_missing = int(n_samples * len(df.columns) * missing_rate)
    missing_indices = np.random.choice(n_samples * len(df.columns), 
                                      size=n_missing, replace=False)
    for idx in missing_indices:
        row = idx // len(df.columns)
        col = idx % len(df.columns)
        if col < len(numeric_features):  # Only numeric features
            df.iloc[row, col] = np.nan
    
    # Inject outliers
    n_outliers = int(n_samples * outlier_rate)
    numeric_cols = list(numeric_features.keys())
    for _ in range(n_outliers):
        row = np.random.randint(0, n_samples)
        col = np.random.choice(numeric_cols)
        df.loc[row, col] = df[col].mean() + np.random.choice([-1, 1]) * df[col].std() * 5
    
    # Add some duplicate rows
    n_duplicates = int(n_samples * 0.02)
    duplicate_indices = np.random.choice(n_samples, size=n_duplicates, replace=False)
    df = pd.concat([df, df.iloc[duplicate_indices]], ignore_index=True)
    
    return df


def main():
    """
    Main function demonstrating the complete Task 6 workflow.
    
    Returns:
        DataPreprocessor: Configured preprocessor object
    """
    print("=" * 80)
    print("Task 6: Real-World Case Study - Complete Data Preprocessing Challenge")
    print("=" * 80)
    print()
    
    # Create complex synthetic dataset
    print("Creating complex synthetic dataset...")
    df = create_complex_synthetic_dataset(
        n_samples=2000,
        n_features=25,
        missing_rate=0.15,
        outlier_rate=0.08,
        imbalance_ratio=0.3
    )
    print(f"Dataset created: {df.shape[0]} samples, {df.shape[1]} features")
    print()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(df, target_col='target')
    
    # Step 1: Data Quality Assessment
    print("STEP 1: DATA QUALITY ASSESSMENT")
    print("=" * 80)
    preprocessor.assess_data_quality()
    preprocessor.print_quality_report()
    
    # Step 2: EDA Visualizations
    print("STEP 2: EXPLORATORY DATA ANALYSIS")
    print("=" * 80)
    print("Generating EDA visualizations...")
    
    preprocessor.visualize_eda()
    plt.savefig('task6_eda_distributions.png', dpi=300, bbox_inches='tight')
    print("  Saved: task6_eda_distributions.png")
    
    preprocessor.visualize_correlations()
    plt.savefig('task6_correlations.png', dpi=300, bbox_inches='tight')
    print("  Saved: task6_correlations.png")
    
    preprocessor.visualize_missing_patterns()
    plt.savefig('task6_missing_patterns.png', dpi=300, bbox_inches='tight')
    print("  Saved: task6_missing_patterns.png")
    print()
    
    # Step 3: Preprocessing Pipeline
    print("STEP 3: PREPROCESSING PIPELINE")
    print("=" * 80)
    
    config = {
        'handle_missing': True,
        'missing_strategy': 'auto',
        'handle_outliers': True,
        'outlier_method': 'iqr',
        'outlier_action': 'cap',
        'encode_categorical': True,
        'encoding_method': 'auto',
        'engineer_features': True,
        'scale_features': True,
        'scaling_method': 'robust',
        'select_features': True,
        'selection_method': 'kbest',
        'n_features': 30,
        'balance_classes': True,
        'balance_method': 'undersample'
    }
    
    preprocessor.preprocess_pipeline(config)
    print()
    
    # Step 4: Additional Visualizations
    print("STEP 4: GENERATING ADDITIONAL VISUALIZATIONS")
    print("=" * 80)
    
    print("  Creating PCA visualization...")
    preprocessor.visualize_pca()
    plt.savefig('task6_pca_visualization.png', dpi=300, bbox_inches='tight')
    print("  Saved: task6_pca_visualization.png")
    print()
    
    # Step 5: Generate Report
    print("STEP 5: GENERATING REPORT")
    print("=" * 80)
    preprocessor.generate_report('task6_preprocessing_report.json')
    print()
    
    # Summary
    print("=" * 80)
    print("TASK 6 COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print()
    print("SUMMARY:")
    print(f"  Original shape: {preprocessor.original_data.shape}")
    print(f"  Final shape: {preprocessor.data.shape}")
    print(f"  Preprocessing steps: {len(preprocessor.preprocessing_steps)}")
    print()
    print("FILES GENERATED:")
    print("  - task6_eda_distributions.png")
    print("  - task6_correlations.png")
    print("  - task6_missing_patterns.png")
    print("  - task6_pca_visualization.png")
    print("  - task6_preprocessing_report.json")
    print()
    
    return preprocessor


if __name__ == "__main__":
    preprocessor = main()
    plt.show()
