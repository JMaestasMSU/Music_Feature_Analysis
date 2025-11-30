import numpy as np
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
import pandas as pd

try:
    from bayes_opt import BayesianOptimization
    BAYES_OPT_AVAILABLE = True
except ImportError:
    BAYES_OPT_AVAILABLE = False
    print("Warning: bayes_opt not installed. Install with: pip install bayesian-optimization")


class BayesianHyperparameterOptimizer:
    """
    Bayesian optimization for hyperparameter tuning.
    Uses Gaussian Processes to model the objective function.
    """
    
    def __init__(self, X_train, y_train, model_type='random_forest', cv_folds=5):
        """
        Initialize optimizer.
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_type: 'random_forest' or 'gradient_boosting'
            cv_folds: Number of cross-validation folds
        """
        self.X_train = X_train
        self.y_train = y_train
        self.model_type = model_type
        self.cv_folds = cv_folds
        self.cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        self.optimizer_history = []
    
    def _objective_random_forest(self, n_estimators, max_depth, min_samples_split,
                                 min_samples_leaf, max_features):
        """Objective function for Random Forest."""
        params = {
            'n_estimators': int(n_estimators),
            'max_depth': int(max_depth),
            'min_samples_split': int(min_samples_split),
            'min_samples_leaf': int(min_samples_leaf),
            'max_features': max_features,
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = RandomForestClassifier(**params)
        scores = cross_validate(model, self.X_train, self.y_train, cv=self.cv,
                               scoring='f1_weighted', return_train_score=False)
        
        mean_f1 = scores['test_score'].mean()
        self.optimizer_history.append({
            'iteration': len(self.optimizer_history),
            'params': params.copy(),
            'f1_score': mean_f1,
            'std': scores['test_score'].std()
        })
        
        return mean_f1
    
    def _objective_gradient_boosting(self, n_estimators, learning_rate, max_depth,
                                     min_samples_split, subsample):
        """Objective function for Gradient Boosting."""
        params = {
            'n_estimators': int(n_estimators),
            'learning_rate': learning_rate,
            'max_depth': int(max_depth),
            'min_samples_split': int(min_samples_split),
            'subsample': subsample,
            'random_state': 42
        }
        
        model = GradientBoostingClassifier(**params)
        scores = cross_validate(model, self.X_train, self.y_train, cv=self.cv,
                               scoring='f1_weighted', return_train_score=False)
        
        mean_f1 = scores['test_score'].mean()
        self.optimizer_history.append({
            'iteration': len(self.optimizer_history),
            'params': params.copy(),
            'f1_score': mean_f1,
            'std': scores['test_score'].std()
        })
        
        return mean_f1
    
    def optimize_random_forest(self, n_init_points=5, n_iter=20):
        """
        Optimize Random Forest hyperparameters.
        
        Args:
            n_init_points: Random exploration iterations
            n_iter: Bayesian optimization iterations
        
        Returns:
            Best parameters dictionary
        """
        if not BAYES_OPT_AVAILABLE:
            return self._grid_search_random_forest()
        
        print("="*70)
        print("BAYESIAN OPTIMIZATION: RANDOM FOREST")
        print("="*70)
        
        pbounds = {
            'n_estimators': (50, 300),
            'max_depth': (5, 30),
            'min_samples_split': (2, 20),
            'min_samples_leaf': (1, 10),
            'max_features': (0.3, 1.0)
        }
        
        optimizer = BayesianOptimization(
            f=self._objective_random_forest,
            pbounds=pbounds,
            random_state=42,
            verbose=2
        )
        
        optimizer.maximize(init_points=n_init_points, n_iter=n_iter)
        
        # Extract best parameters
        best_params = optimizer.max['params']
        best_f1 = optimizer.max['target']
        
        best_params = {
            'n_estimators': int(best_params['n_estimators']),
            'max_depth': int(best_params['max_depth']),
            'min_samples_split': int(best_params['min_samples_split']),
            'min_samples_leaf': int(best_params['min_samples_leaf']),
            'max_features': best_params['max_features'],
            'random_state': 42,
            'n_jobs': -1
        }
        
        print(f"\nBest F1-Score: {best_f1:.4f}")
        print(f"Best Parameters: {best_params}")
        
        return best_params
    
    def optimize_gradient_boosting(self, n_init_points=5, n_iter=20):
        """
        Optimize Gradient Boosting hyperparameters.
        
        Args:
            n_init_points: Random exploration iterations
            n_iter: Bayesian optimization iterations
        
        Returns:
            Best parameters dictionary
        """
        if not BAYES_OPT_AVAILABLE:
            return self._grid_search_gradient_boosting()
        
        print("="*70)
        print("BAYESIAN OPTIMIZATION: GRADIENT BOOSTING")
        print("="*70)
        
        pbounds = {
            'n_estimators': (50, 300),
            'learning_rate': (0.001, 0.3),
            'max_depth': (3, 10),
            'min_samples_split': (2, 20),
            'subsample': (0.5, 1.0)
        }
        
        optimizer = BayesianOptimization(
            f=self._objective_gradient_boosting,
            pbounds=pbounds,
            random_state=42,
            verbose=2
        )
        
        optimizer.maximize(init_points=n_init_points, n_iter=n_iter)
        
        best_params = optimizer.max['params']
        best_f1 = optimizer.max['target']
        
        best_params = {
            'n_estimators': int(best_params['n_estimators']),
            'learning_rate': best_params['learning_rate'],
            'max_depth': int(best_params['max_depth']),
            'min_samples_split': int(best_params['min_samples_split']),
            'subsample': best_params['subsample'],
            'random_state': 42
        }
        
        print(f"\nBest F1-Score: {best_f1:.4f}")
        print(f"Best Parameters: {best_params}")
        
        return best_params
    
    def _grid_search_random_forest(self):
        """Fallback grid search for Random Forest."""
        print("Bayesian optimization unavailable. Running grid search...")
        
        param_grid = {
            'n_estimators': [100, 150, 200],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 5, 10],
        }
        
        best_score = 0
        best_params = None
        
        for n_est in param_grid['n_estimators']:
            for max_d in param_grid['max_depth']:
                for min_samp in param_grid['min_samples_split']:
                    params = {
                        'n_estimators': n_est,
                        'max_depth': max_d,
                        'min_samples_split': min_samp,
                        'random_state': 42,
                        'n_jobs': -1
                    }
                    model = RandomForestClassifier(**params)
                    scores = cross_validate(model, self.X_train, self.y_train,
                                          cv=self.cv, scoring='f1_weighted')
                    mean_score = scores['test_score'].mean()
                    
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = params
        
        print(f"Best F1-Score: {best_score:.4f}")
        return best_params
    
    def _grid_search_gradient_boosting(self):
        """Fallback grid search for Gradient Boosting."""
        print("Bayesian optimization unavailable. Running grid search...")
        
        param_grid = {
            'n_estimators': [100, 150, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
        }
        
        best_score = 0
        best_params = None
        
        for n_est in param_grid['n_estimators']:
            for lr in param_grid['learning_rate']:
                for max_d in param_grid['max_depth']:
                    params = {
                        'n_estimators': n_est,
                        'learning_rate': lr,
                        'max_depth': max_d,
                        'random_state': 42
                    }
                    model = GradientBoostingClassifier(**params)
                    scores = cross_validate(model, self.X_train, self.y_train,
                                          cv=self.cv, scoring='f1_weighted')
                    mean_score = scores['test_score'].mean()
                    
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = params
        
        print(f"Best F1-Score: {best_score:.4f}")
        return best_params
    
    def plot_optimization_history(self):
        """Plot Bayesian optimization history."""
        if not self.optimizer_history:
            print("No optimization history to plot.")
            return
        
        history_df = pd.DataFrame(self.optimizer_history)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(history_df['iteration'], history_df['f1_score'], 'bo-', linewidth=2, markersize=6)
        axes[0].fill_between(history_df['iteration'],
                            history_df['f1_score'] - history_df['std'],
                            history_df['f1_score'] + history_df['std'],
                            alpha=0.3)
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('F1-Score')
        axes[0].set_title('Bayesian Optimization: Objective Function')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].scatter(range(len(history_df)), history_df['f1_score'],
                       s=100, alpha=0.6, c=history_df['f1_score'], cmap='viridis')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('F1-Score')
        axes[1].set_title('F1-Score Progression')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/bayesian_optimization_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nTop 5 Configurations:")
        top_5 = history_df.nlargest(5, 'f1_score')
        for idx, row in top_5.iterrows():
            print(f"  {idx+1}. F1-Score: {row['f1_score']:.4f} (±{row['std']:.4f})")
