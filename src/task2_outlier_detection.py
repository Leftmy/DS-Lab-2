"""
Task 2: Outlier Detection and Treatment with Advanced Visualization

This module implements various outlier detection and treatment techniques,
including statistical methods, machine learning-based detection, and 
comprehensive visualization of outliers and their impact.

Author: MLDS Course
Date: December 2025
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from scipy import stats
from typing import Tuple, Dict
import warnings

warnings.filterwarnings('ignore')


class OutlierAnalyzer:
    """
    A comprehensive outlier analyzer that detects and treats outliers using
    multiple methods and provides detailed visualizations.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the analyzer with a dataset.
        
        Parameters:
            data (pd.DataFrame): Input DataFrame to analyze
        """
        self.data = data.copy()
        self.original_data = data.copy()
        self.outlier_masks = {}
        self.treated_datasets = {}

    def _get_numeric_data(self) -> pd.DataFrame:
        return self.data.select_dtypes(include=[np.number])

    def _prepare_detection_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        numeric = self._get_numeric_data().replace([np.inf, -np.inf], np.nan)
        numeric = numeric.dropna(axis=1, how='all')
        valid_rows = numeric.notna().any(axis=1)
        if numeric.empty:
            return numeric, valid_rows
        prepared = numeric.fillna(numeric.median()).fillna(0.0)
        return prepared, valid_rows
    
    def detect_outliers_zscore(self, threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect outliers using Z-score method.
        
        Parameters:
            threshold (float): Z-score threshold for outlier detection
            
        Returns:
            pd.DataFrame: Boolean mask indicating outliers
        """
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("threshold must be a finite number greater than zero.")

        numeric = self._get_numeric_data()
        if numeric.empty:
            mask = pd.DataFrame(index=self.data.index)
        else:
            infinite_values = numeric.isin([np.inf, -np.inf])
            finite_numeric = numeric.replace([np.inf, -np.inf], np.nan)
            standard_deviation = finite_numeric.std(ddof=0).replace(0, np.nan)
            z_scores = (finite_numeric - finite_numeric.mean()) / standard_deviation
            mask = (z_scores.abs().gt(threshold) | infinite_values).fillna(False).astype(bool)
        self.outlier_masks['zscore'] = mask.copy()
        return mask.copy()
    
    def detect_outliers_iqr(self, multiplier: float = 1.5) -> pd.DataFrame:
        """
        Detect outliers using Interquartile Range (IQR) method.
        
        Parameters:
            multiplier (float): IQR multiplier for boundary calculation
            
        Returns:
            pd.DataFrame: Boolean mask indicating outliers
        """
        if not np.isfinite(multiplier) or multiplier < 0:
            raise ValueError("multiplier must be a finite non-negative number.")

        numeric = self._get_numeric_data()
        if numeric.empty:
            mask = pd.DataFrame(index=self.data.index)
        else:
            infinite_values = numeric.isin([np.inf, -np.inf])
            finite_numeric = numeric.replace([np.inf, -np.inf], np.nan)
            first_quartile = finite_numeric.quantile(0.25)
            third_quartile = finite_numeric.quantile(0.75)
            spread = third_quartile - first_quartile
            lower_bound = first_quartile - multiplier * spread
            upper_bound = third_quartile + multiplier * spread
            mask = (finite_numeric.lt(lower_bound, axis='columns')
                    | finite_numeric.gt(upper_bound, axis='columns') | infinite_values)
            mask = mask.fillna(False).astype(bool)
        self.outlier_masks['iqr'] = mask.copy()
        return mask.copy()
    
    def detect_outliers_isolation_forest(self, contamination: float = 0.1, 
                                        random_state: int = 42) -> pd.DataFrame:
        """
        Detect outliers using Isolation Forest algorithm.
        
        Parameters:
            contamination (float): Expected proportion of outliers in the dataset
            random_state (int): Random seed for reproducibility
            
        Returns:
            pd.DataFrame: Boolean mask indicating outliers
        """
        if not 0 < contamination <= 0.5:
            raise ValueError("contamination must be in the interval (0, 0.5].")

        prepared, valid_rows = self._prepare_detection_data()
        numeric_columns = self._get_numeric_data().columns
        mask = pd.DataFrame(False, index=self.data.index, columns=numeric_columns, dtype=bool)
        if prepared.shape[1] and int(valid_rows.sum()) >= 2:
            model = IsolationForest(contamination=contamination, random_state=random_state)
            predictions = model.fit_predict(prepared.loc[valid_rows])
            flags = np.repeat((predictions == -1)[:, np.newaxis], prepared.shape[1], axis=1)
            mask.loc[valid_rows, prepared.columns] = flags
        infinite_rows = self._get_numeric_data().isin([np.inf, -np.inf]).any(axis=1)
        mask.loc[infinite_rows, :] = True
        self.outlier_masks['isolation_forest'] = mask.copy()
        return mask.copy()
    
    def detect_outliers_lof(self, n_neighbors: int = 20, 
                           contamination: float = 0.1) -> pd.DataFrame:
        """
        Detect outliers using Local Outlier Factor (LOF) algorithm.
        
        Parameters:
            n_neighbors (int): Number of neighbors to consider
            contamination (float): Expected proportion of outliers
            
        Returns:
            pd.DataFrame: Boolean mask indicating outliers
        """
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1.")
        if not 0 < contamination <= 0.5:
            raise ValueError("contamination must be in the interval (0, 0.5].")

        prepared, valid_rows = self._prepare_detection_data()
        numeric_columns = self._get_numeric_data().columns
        mask = pd.DataFrame(False, index=self.data.index, columns=numeric_columns, dtype=bool)
        n_samples = int(valid_rows.sum())
        if prepared.shape[1] and n_samples >= 3:
            features = prepared.loc[valid_rows]
            model = LocalOutlierFactor(
                n_neighbors=min(n_neighbors, n_samples - 1),
                contamination=contamination,
            )
            predictions = model.fit_predict(features)
            flags = np.repeat((predictions == -1)[:, np.newaxis], prepared.shape[1], axis=1)
            mask.loc[valid_rows, prepared.columns] = flags
        infinite_rows = self._get_numeric_data().isin([np.inf, -np.inf]).any(axis=1)
        mask.loc[infinite_rows, :] = True
        self.outlier_masks['lof'] = mask.copy()
        return mask.copy()
    
    def get_outlier_summary(self) -> pd.DataFrame:
        """
        Get summary statistics of outliers detected by different methods.
        
        Returns:
            DataFrame with outlier counts for each method
        """
        if not self.outlier_masks:
            return pd.DataFrame()

        summary = {}
        row_count = len(self.data)
        for method, mask in self.outlier_masks.items():
            count = int(mask.any(axis=1).sum()) if not mask.empty else 0
            summary[method] = {
                'outlier_count': count,
                'outlier_percentage': count / row_count * 100 if row_count else 0.0,
            }
        return pd.DataFrame(summary)
    
    def remove_outliers(self, method: str = 'iqr') -> pd.DataFrame:
        """
        Remove outliers from the dataset.
        
        Parameters:
            method (str): Detection method to use
            
        Returns:
            pd.DataFrame: Dataset with outliers removed
        """
        detector_names = {
            'zscore': 'detect_outliers_zscore',
            'iqr': 'detect_outliers_iqr',
            'isolation_forest': 'detect_outliers_isolation_forest',
            'lof': 'detect_outliers_lof',
        }
        if method not in detector_names:
            raise ValueError(f"Unsupported outlier detection method: {method}")
        if method not in self.outlier_masks:
            getattr(self, detector_names[method])()

        outliers = self.outlier_masks[method].any(axis=1)
        result = self.data.loc[~outliers].copy()
        self.treated_datasets[f'removed_{method}'] = result.copy()
        return result.copy()
    
    def cap_outliers(self, method: str = 'iqr', multiplier: float = 1.5) -> pd.DataFrame:
        """
        Cap outliers using winsorization.
        
        Parameters:
            method (str): Detection method to use
            multiplier (float): IQR multiplier for bounds
            
        Returns:
            pd.DataFrame: Dataset with capped outliers
        """
        detector_names = {
            'zscore': 'detect_outliers_zscore',
            'iqr': 'detect_outliers_iqr',
            'isolation_forest': 'detect_outliers_isolation_forest',
            'lof': 'detect_outliers_lof',
        }
        if method not in detector_names:
            raise ValueError(f"Unsupported outlier detection method: {method}")
        if not np.isfinite(multiplier) or multiplier < 0:
            raise ValueError("multiplier must be a finite non-negative number.")
        if method == 'iqr':
            self.detect_outliers_iqr(multiplier=multiplier)
        elif method not in self.outlier_masks:
            getattr(self, detector_names[method])()

        result = self.data.copy()
        numeric_cols = self._get_numeric_data().columns
        if len(numeric_cols):
            finite_numeric = result[numeric_cols].replace([np.inf, -np.inf], np.nan)
            first_quartile = finite_numeric.quantile(0.25)
            third_quartile = finite_numeric.quantile(0.75)
            spread = third_quartile - first_quartile
            lower_bound = first_quartile - multiplier * spread
            upper_bound = third_quartile + multiplier * spread
            outlier_mask = self.outlier_masks[method].reindex(
                index=result.index, columns=numeric_cols, fill_value=False
            )
            for column in numeric_cols:
                below = outlier_mask[column] & result[column].lt(lower_bound[column])
                above = outlier_mask[column] & result[column].gt(upper_bound[column])
                result.loc[below, column] = lower_bound[column]
                result.loc[above, column] = upper_bound[column]
        self.treated_datasets['capped'] = result.copy()
        return result.copy()
    
    def transform_outliers_log(self) -> pd.DataFrame:
        """
        Transform data using log transformation.
        
        Returns:
            pd.DataFrame: Dataset with log-transformed values
        """
        result = self.data.copy()
        for column in self._get_numeric_data().columns:
            values = result[column]
            finite_values = values[np.isfinite(values)]
            if finite_values.empty:
                continue
            shift = max(0.0, -float(finite_values.min()))
            finite = np.isfinite(values)
            result.loc[finite, column] = np.log1p(values.loc[finite] + shift)
        self.treated_datasets['log_transform'] = result.copy()
        return result.copy()
    
    def transform_outliers_boxcox(self) -> pd.DataFrame:
        """
        Transform data using Box-Cox/Yeo-Johnson transformation.
        
        Returns:
            pd.DataFrame: Dataset with power-transformed values
        """
        result = self.data.copy()
        for column in self._get_numeric_data().columns:
            values = result[column].to_numpy(dtype=float, na_value=np.nan)
            finite = np.isfinite(values)
            if finite.sum() < 2 or np.all(values[finite] == values[finite][0]):
                continue
            transformer = PowerTransformer(method='yeo-johnson')
            values[finite] = transformer.fit_transform(values[finite].reshape(-1, 1)).ravel()
            result[column] = values
        self.treated_datasets['boxcox_transform'] = result.copy()
        return result.copy()
    
    def evaluate_model_performance(self, target_col: str, 
                                   dataset_name: str = 'original') -> Dict[str, float]:
        """
        Evaluate model performance with and without outliers.
        
        Args:
            target_col: Name of target column
            dataset_name: Dataset to evaluate ('original' or treated dataset name)
            
        Returns:
            Dictionary with performance metrics
        """
        if dataset_name == 'original':
            dataset = self.original_data.copy()
        elif dataset_name in self.treated_datasets:
            dataset = self.treated_datasets[dataset_name].copy()
        else:
            raise ValueError(f"Dataset '{dataset_name}' is not available.")

        if target_col not in dataset.columns:
            raise ValueError(f"Target column '{target_col}' not found.")
        if not pd.api.types.is_numeric_dtype(dataset[target_col]):
            raise ValueError(f"Target column '{target_col}' must be numeric.")

        numeric = dataset.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
        if target_col not in numeric.columns:
            raise ValueError(f"Target column '{target_col}' must be numeric.")
        predictors = numeric.drop(columns=[target_col])
        if predictors.empty:
            raise ValueError("At least one numeric predictor is required.")

        model_data = predictors.assign(**{target_col: numeric[target_col]}).dropna()
        if len(model_data) < 4:
            raise ValueError("At least four complete rows are required for cross-validation.")
        features = model_data.drop(columns=[target_col])
        target = model_data[target_col]

        cv_splits = min(5, len(model_data) // 2)
        cross_validator = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
        scores = cross_val_score(
            LinearRegression(), features, target, cv=cross_validator, scoring='r2'
        )
        model = LinearRegression().fit(features, target)
        return {
            'r2_score': float(model.score(features, target)),
            'r2_std': float(np.std(scores)),
            'mean_cv_score': float(np.mean(scores)),
        }
    
    def visualize_outliers_boxplot(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Create box plots showing outliers.
        
        Parameters:
            figsize (Tuple[int, int]): Figure size for plots
            
        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
        numeric_cols = self._get_numeric_data().columns[:6]
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()

        for index, column in enumerate(numeric_cols):
            sns.boxplot(y=self.data[column], ax=axes[index], color='skyblue')
            axes[index].set_title(f'Box Plot: {column}', fontweight='bold')
            axes[index].set_ylabel(column)
        for index in range(len(numeric_cols), len(axes)):
            axes[index].axis('off')

        if len(numeric_cols) == 0:
            axes[0].text(0.5, 0.5, 'No numeric columns', ha='center', va='center')
            axes[0].axis('off')
        plt.suptitle('Outlier Detection: Box Plots', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def visualize_outliers_violin(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Create violin plots showing distribution with outliers.
        
        Args:
            figsize: Figure size for plots
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns[:6]
        
        if len(numeric_cols) == 0:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No numeric columns', ha='center', va='center')
            ax.axis('off')
            return fig

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        for index, column in enumerate(numeric_cols):
            values = self.data[column].dropna()
            if not values.empty:
                sns.violinplot(y=values, ax=axes[index], color='lightcoral', inner='quartile')
                axes[index].axhline(values.mean(), color='blue', linestyle='--', label='Mean')
                axes[index].axhline(values.median(), color='green', linestyle=':', label='Median')
                axes[index].legend()
            axes[index].set_title(f'Violin Plot: {column}', fontweight='bold')
            axes[index].set_ylabel(column)
        for index in range(len(numeric_cols), len(axes)):
            axes[index].axis('off')
        plt.suptitle('Outlier Detection: Violin Plots', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def visualize_outliers_scatter(self, x_col: str, y_col: str, 
                                   figsize: Tuple[int, int] = (15, 5)):
        """
        Create scatter plots showing outliers detected by different methods.
        
        Args:
            x_col: Column for x-axis
            y_col: Column for y-axis
            figsize: Figure size for plots
        """
        if x_col not in self.data.columns or y_col not in self.data.columns:
            print(f"Column '{x_col if x_col not in self.data.columns else y_col}' not found")
            return
        if not pd.api.types.is_numeric_dtype(self.data[x_col]) or not pd.api.types.is_numeric_dtype(self.data[y_col]):
            print("Scatter plot columns must be numeric")
            return

        detectors = {
            'zscore': self.detect_outliers_zscore,
            'iqr': self.detect_outliers_iqr,
            'isolation_forest': self.detect_outliers_isolation_forest,
        }
        for method, detector in detectors.items():
            if method not in self.outlier_masks:
                detector()

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        for ax, method in zip(axes, detectors):
            mask = self.outlier_masks[method].any(axis=1)
            valid = self.data[[x_col, y_col]].notna().all(axis=1)
            regular = valid & ~mask
            outliers = valid & mask
            ax.scatter(self.data.loc[regular, x_col], self.data.loc[regular, y_col],
                       color='steelblue', alpha=0.65, label='Inliers')
            ax.scatter(self.data.loc[outliers, x_col], self.data.loc[outliers, y_col],
                       color='crimson', alpha=0.8, label='Outliers')
            ax.set_title(f'{method.replace("_", " ").title()}')
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.legend()
            ax.grid(alpha=0.25)
        plt.tight_layout()
        return fig
    
    def visualize_3d_outliers(self, x_col: str, y_col: str, z_col: str, 
                             method: str = 'isolation_forest',
                             figsize: Tuple[int, int] = (12, 8)):
        """
        Create three-dimensional scatter plot showing multivariate outliers.
        
        Args:
            x_col, y_col, z_col: Columns for three-dimensional axes
            method: Detection method to use
            figsize: Figure size for plot
        """
        from mpl_toolkits.mplot3d import Axes3D
        
        columns = [x_col, y_col, z_col]
        if any(column not in self.data.columns for column in columns):
            missing = next(column for column in columns if column not in self.data.columns)
            print(f"Column '{missing}' not found")
            return
        if method not in self.outlier_masks:
            print(f"Outlier detection method '{method}' not found")
            return
        if any(not pd.api.types.is_numeric_dtype(self.data[column]) for column in columns):
            print("3D scatter plot columns must be numeric")
            return

        values = self.data[columns]
        valid = values.notna().all(axis=1)
        outliers = self.outlier_masks[method].any(axis=1) & valid
        inliers = valid & ~outliers
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(values.loc[inliers, x_col], values.loc[inliers, y_col],
                   values.loc[inliers, z_col], color='steelblue', alpha=0.65, label='Inliers')
        ax.scatter(values.loc[outliers, x_col], values.loc[outliers, y_col],
                   values.loc[outliers, z_col], color='crimson', alpha=0.85, label='Outliers')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_zlabel(z_col)
        ax.set_title(f'3D Outliers: {method.replace("_", " ").title()}')
        ax.legend()
        return fig
    
    def visualize_treatment_comparison(self, column: str, 
                                      figsize: Tuple[int, int] = (15, 10)):
        """
        Compare distributions before and after different treatments.
        
        Args:
            column: Column to visualize
            figsize: Figure size for plots
        """
        if column not in self.data.columns:
            print(f"Column '{column}' not found")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        # Original data
        ax = axes[0]
        self.original_data[column].hist(bins=30, ax=ax, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(self.original_data[column].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
        ax.set_title('Original Data', fontweight='bold')
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Treated datasets
        treatment_names = list(self.treated_datasets.keys())[:5]
        for idx, name in enumerate(treatment_names, start=1):
            ax = axes[idx]
            treated_data = self.treated_datasets[name]
            if column in treated_data.columns:
                treated_data[column].hist(bins=30, ax=ax, color='lightcoral', 
                                        edgecolor='black', alpha=0.7)
                ax.axvline(treated_data[column].mean(), color='red', 
                          linestyle='--', linewidth=2, label='Mean')
                ax.set_title(f'{name.replace("_", " ").title()}', fontweight='bold')
                ax.set_xlabel(column)
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(treatment_names) + 1, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Treatment Comparison: {column}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig
    
    def visualize_qq_plots(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Create Q-Q plots to assess normality before and after treatment.
        
        Args:
            figsize: Figure size for plots
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns[:6]
        
        if len(numeric_cols) == 0:
            print("No numeric columns to visualize")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for idx, col in enumerate(numeric_cols):
            ax = axes[idx]
            
            stats.probplot(self.data[col].dropna(), dist="norm", plot=ax)
            ax.set_title(f'Q-Q Plot: {col}', fontweight='bold')
            ax.grid(alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Q-Q Plots for Normality Assessment', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig


def create_sample_dataset_with_outliers() -> pd.DataFrame:
    """
    Create a sample dataset with outliers for demonstration.
    
    Returns:
        pd.DataFrame: Dataset with outliers
    """
    np.random.seed(42)
    n_samples = 300
    
    # Create base data
    data = {
        'feature1': np.random.normal(100, 15, n_samples),
        'feature2': np.random.exponential(50, n_samples),
        'feature3': np.random.normal(500, 100, n_samples),
        'feature4': np.random.uniform(0, 100, n_samples),
        'target': np.random.normal(200, 30, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Add synthetic target relationship
    df['target'] = 0.5 * df['feature1'] + 0.3 * df['feature2'] + np.random.normal(0, 10, n_samples)
    
    # Inject outliers
    # Extreme high values in feature1
    outlier_indices = np.random.choice(df.index, size=15, replace=False)
    df.loc[outlier_indices, 'feature1'] = np.random.uniform(200, 300, len(outlier_indices))
    
    # Extreme low values in feature2
    outlier_indices = np.random.choice(df.index, size=10, replace=False)
    df.loc[outlier_indices, 'feature2'] = np.random.uniform(200, 400, len(outlier_indices))
    
    # Random extreme values in feature3
    outlier_indices = np.random.choice(df.index, size=20, replace=False)
    # Split outliers into low and high
    low_outliers = outlier_indices[:len(outlier_indices)//2]
    high_outliers = outlier_indices[len(outlier_indices)//2:]
    df.loc[low_outliers, 'feature3'] = np.random.uniform(0, 100, len(low_outliers))
    df.loc[high_outliers, 'feature3'] = np.random.uniform(1000, 1500, len(high_outliers))
    
    return df


def main():
    """
    Main function demonstrating the complete Task 2 workflow.
    
    Returns:
        OutlierAnalyzer: Configured analyzer object
    """
    print("=" * 80)
    print("Task 2: Outlier Detection and Treatment with Advanced Visualization")
    print("=" * 80)
    print()
    
    # Create sample dataset
    print("Creating sample dataset with outliers...")
    df = create_sample_dataset_with_outliers()
    print(f"Dataset created: {df.shape[0]} rows, {df.shape[1]} columns")
    print()
    
    # Initialize analyzer
    analyzer = OutlierAnalyzer(df)
    
    # Detect outliers using different methods
    print("1. OUTLIER DETECTION")
    print("-" * 80)
    
    print("  a) Z-Score Detection...")
    zscore_outliers = analyzer.detect_outliers_zscore(threshold=3.0)
    print(f"     Outliers detected: {zscore_outliers.sum().sum()}")
    
    print("  b) IQR Detection...")
    iqr_outliers = analyzer.detect_outliers_iqr(multiplier=1.5)
    print(f"     Outliers detected: {iqr_outliers.sum().sum()}")
    
    print("  c) Isolation Forest Detection...")
    iso_outliers = analyzer.detect_outliers_isolation_forest(contamination=0.1)
    print(f"     Outliers detected: {iso_outliers.sum().sum()}")
    
    print("  d) Local Outlier Factor Detection...")
    lof_outliers = analyzer.detect_outliers_lof(n_neighbors=20, contamination=0.1)
    print(f"     Outliers detected: {lof_outliers.sum().sum()}")
    print()
    
    # Get outlier summary
    print("2. OUTLIER SUMMARY")
    print("-" * 80)
    summary = analyzer.get_outlier_summary()
    print(summary)
    print()
    
    # Apply different treatment strategies
    print("3. OUTLIER TREATMENT")
    print("-" * 80)
    
    print("  a) Removing outliers (IQR method)...")
    removed_data = analyzer.remove_outliers(method='iqr')
    print(f"     Rows after removal: {len(removed_data)} (removed {len(df) - len(removed_data)})")
    
    print("  b) Capping outliers (Winsorization)...")
    capped_data = analyzer.cap_outliers(multiplier=1.5)
    print(f"     Data capped at IQR bounds")
    
    print("  c) Log transformation...")
    log_data = analyzer.transform_outliers_log()
    print(f"     Log transformation applied")
    
    print("  d) Box-Cox transformation...")
    boxcox_data = analyzer.transform_outliers_boxcox()
    print(f"     Box-Cox/Yeo-Johnson transformation applied")
    print()
    
    # Evaluate model performance
    print("4. MODEL PERFORMANCE COMPARISON")
    print("-" * 80)
    
    datasets_to_evaluate = ['original', 'removed_iqr', 'capped', 'log_transform']
    for dataset_name in datasets_to_evaluate:
        try:
            performance = analyzer.evaluate_model_performance('target', dataset_name)
            print(f"  {dataset_name.replace('_', ' ').title()}:")
            print(f"    R² Score: {performance['r2_score']:.4f} (±{performance['r2_std']:.4f})")
        except Exception as e:
            print(f"  {dataset_name}: Could not evaluate - {str(e)}")
    print()
    
    # Generate visualizations
    print("5. GENERATING VISUALIZATIONS")
    print("-" * 80)
    
    print("  Creating box plots...")
    analyzer.visualize_outliers_boxplot()
    plt.savefig('task2_boxplots.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_boxplots.png")
    
    print("  Creating violin plots...")
    analyzer.visualize_outliers_violin()
    plt.savefig('task2_violin_plots.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_violin_plots.png")
    
    print("  Creating scatter plots with outlier detection...")
    analyzer.visualize_outliers_scatter('feature1', 'feature2')
    plt.savefig('task2_scatter_outliers.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_scatter_outliers.png")
    
    print("  Creating 3D outlier visualization...")
    analyzer.visualize_3d_outliers('feature1', 'feature2', 'feature3', method='isolation_forest')
    plt.savefig('task2_3d_outliers.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_3d_outliers.png")
    
    print("  Creating treatment comparison plots...")
    analyzer.visualize_treatment_comparison('feature1')
    plt.savefig('task2_treatment_comparison.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_treatment_comparison.png")
    
    print("  Creating Q-Q plots...")
    analyzer.visualize_qq_plots()
    plt.savefig('task2_qq_plots.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_qq_plots.png")
    print()
    
    print("=" * 80)
    print("Task 2 completed successfully!")
    print("=" * 80)
    
    return analyzer


if __name__ == "__main__":
    analyzer = main()
    plt.show()
