import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, f1_score
import xgboost as xgb

class ModelTrainer:
    """
    Handles data splitting, model training (Regression & Classification), 
    evaluation, and visualization.
    """
    
    def split_data(self, X, y, test_size=0.2, task='regression'):
        """Divides data into train and test sets."""
        if task == 'classification':
            return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        return train_test_split(X, y, test_size=test_size, random_state=42)

    def train_models(self, X_train_processed, y_train, task='regression'):
        """Trains the required models for the specified task."""
        if task == 'regression':
            lr = LinearRegression()
            dt = DecisionTreeRegressor(random_state=42)
            rfr = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42, n_jobs=-1)
            
            lr.fit(X_train_processed, y_train)
            dt.fit(X_train_processed, y_train)
            rfr.fit(X_train_processed, y_train)
            xgb_model.fit(X_train_processed, y_train)
            
            return lr, dt, rfr, xgb_model
        
        else: # Classification (Probability Model)
            rfr_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) 
            xgb_clf = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', n_estimators=100, random_state=42, n_jobs=-1, use_label_encoder=False)
            gbr_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
            
            rfr_clf.fit(X_train_processed, y_train)
            xgb_clf.fit(X_train_processed, y_train)
            gbr_clf.fit(X_train_processed, y_train)
            
            return rfr_clf, xgb_clf, gbr_clf

    def evaluate_model(self, model, X_test_processed, y_test, task='regression'):
        """Evaluates a single model, returning relevant metrics."""
        
        if task == 'regression':
            y_pred = model.predict(X_test_processed)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            return mae, mse, np.sqrt(mse), r2
        
        else: # Classification
            if hasattr(model, 'predict_proba') and (hasattr(model, 'n_outputs_') and model.n_outputs_ == 1 or isinstance(model, RandomForestClassifier)):
                y_prob = model.predict_proba(X_test_processed)[:, 1]
            else:
                y_prob = model.predict(X_test_processed)

            y_binary = (y_prob > 0.5).astype(int)
            
            auc = roc_auc_score(y_test, y_prob)
            f1 = f1_score(y_test, y_binary, zero_division=0)
            return auc, f1

    def print_metrics_summary(self, models, scores_dict, title_suffix=""):
        """Prints the model performance metrics in a clear, formatted summary (using your requested loop)."""

        if 'MAE' not in scores_dict or 'MSE' not in scores_dict or 'R2' not in scores_dict:
            print(f"Cannot print summary: Missing required regression metrics (MAE, MSE, or R2) for {title_suffix}.")
            return
            
        mae_scores = scores_dict['MAE']
        mse_scores = scores_dict['MSE']
        r2_scores = scores_dict['R2']

        print(f"\n--- Model Performance Summary: {title_suffix} ---")
        
        for i, model_name in enumerate(models):
            print(f"Evaluation results for {model_name}:")
            print(f" - Mean Absolute Error (MAE): R{mae_scores[i]:,.2f}")
            print(f" - Mean Squared Error (MSE): {mse_scores[i]:.2e}")
            print(f" - R-squared (R2) Score: {r2_scores[i]:.4f}")
            print("\n")
        print("----------------------------------------------------------------")
        
    def plot_linear_coefficients(self, model, feature_names, model_name):
        """Plots the absolute magnitude of coefficients for Linear Regression."""
        if not isinstance(model, LinearRegression):
            return
            
        coefficients = pd.DataFrame(model.coef_, index=feature_names, columns=["Coefficient"])
        
        # Plotting the absolute magnitude of coefficients (which indicates importance)
        coefficients['Abs_Coefficient'] = np.abs(coefficients['Coefficient'])
        coefficients = coefficients.sort_values(by="Abs_Coefficient", ascending=False).head(10)

        plt.figure(figsize=(12, 6))
        # Plotting the raw coefficient for sign awareness, but sorting by magnitude
        coefficients['Coefficient'].plot(kind='bar', color=np.where(coefficients['Coefficient'] > 0, 'green', 'red'))
        plt.title(f'Top 10 Feature Coefficients for {model_name}')
        plt.xlabel('Features')
        plt.ylabel('Coefficient Value (Impact on Claim Amount)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        # plt.show()

    def plot_feature_importance(self, model, feature_names, model_name):
        """Plots feature importance for tree-based models (DT, RFR, XGBoost)."""
        
        # Check if the model has feature_importances_ attribute
        if not hasattr(model, 'feature_importances_'):
            return
            
        feature_importance = pd.DataFrame(model.feature_importances_, index=feature_names, columns=["Importance"])
        feature_importance = feature_importance.sort_values(by="Importance", ascending=False).head(10)

        plt.figure(figsize=(12, 6))
        feature_importance.plot(kind='bar', legend=False, color='darkcyan')
        plt.title(f'Top 10 Feature Importance for {model_name}')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        # plt.show()

    def plot_metrics(self, models, scores_dict, title_suffix=""):
        """Creates bar charts comparing model performance."""
        metric_names = list(scores_dict.keys())
        
        fig, axes = plt.subplots(1, len(metric_names), figsize=(5 * len(metric_names), 5))
        if len(metric_names) == 1:
            axes = [axes]

        fig.suptitle(f'Model Performance Comparison - {title_suffix}', fontsize=16)

        for i, metric_name in enumerate(metric_names):
            scores = scores_dict[metric_name]
            axes[i].bar(models, scores, color=['skyblue', 'salmon', 'lightgreen', 'gold'])
            axes[i].set_title(metric_name)
            axes[i].set_ylabel('Score')
            axes[i].set_xlabel('Model')
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        # plt.show()