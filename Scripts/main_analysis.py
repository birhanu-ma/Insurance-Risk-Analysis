import pandas as pd
import numpy as np
import warnings

# --- IMPORT UTILITY CLASSES (Assumes these files are accessible) ---
from .model_trainer import ModelTrainer
from .data_processor import DataProcessor
class MainAnalysis:
    """
    Orchestrates the end-to-end risk modeling pipeline (Severity and Probability).
    
    Accepts the cleaned DataFrame in the constructor and runs the full analysis
    via dedicated methods.
    """
    def __init__(self, df_cleaned):
        # 1. Store the cleaned data and prepare the target flag
        self.df = df_cleaned.copy()
        self.df['Claim_Flag'] = np.where(self.df['Totalclaims'] > 0, 1, 0)
        
        # 2. Initialize helper objects
        self.m = ModelTrainer()
        self.dp = DataProcessor()
        
        # Store results for external access
        self.severity_results = {}
        self.probability_results = {}
        
        warnings.filterwarnings('ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
        print(f"MainAnalysis initialized with data of shape: {self.df.shape}")
    def run_severity_model(self):
            """Executes the full Claim Severity (Regression) pipeline."""
            print("\n--- 4.1 Claim Severity (Regression) Pipeline Starting ---")
            
            # 1-3. Data Preparation (Unchanged)
            df_severity = self.df[self.df['Claim_Flag'] == 1].copy()
            X_sev = df_severity.drop(['Totalclaims', 'Claim_Flag'], axis=1)
            y_sev = df_severity['Totalclaims']
            X_train_sev, X_test_sev, y_train_sev, y_test_sev = self.m.split_data(X_sev, y_sev, task='regression')
            dp_sev = DataProcessor()
            X_train_processed_sev = dp_sev.encode_and_scale(X_train_sev, fit=True)
            X_test_processed_sev = dp_sev.encode_and_scale(X_test_sev, fit=False)
            feature_names_sev = dp_sev.feature_names
    
            # 4. Train Models
            lr_model, dt_model, rfr_model, xgb_model = self.m.train_models(X_train_processed_sev, y_train_sev, task='regression')
            models_sev = ['Linear Regression', 'Decision Tree', 'Random Forest', 'XGBoost']
            models_sev_dict = {
                'Linear Regression': lr_model, 
                'Decision Tree': dt_model, 
                'Random Forest': rfr_model, 
                'XGBoost': xgb_model
            } # Dictionary is crucial for iterating models/names
    
            # 5-6. Evaluation and Summary (Unchanged)
            mae_scores, mse_scores, rmse_scores, r2_scores = [], [], [], []
            for model in [lr_model, dt_model, rfr_model, xgb_model]:
                mae, mse, rmse, r2 = self.m.evaluate_model(model, X_test_processed_sev, y_test_sev, task='regression')
                mae_scores.append(mae)
                mse_scores.append(mse)
                r2_scores.append(r2)
    
            self.severity_results = {
                'models': models_sev, 'mae': mae_scores, 'r2': r2_scores, 
                'xgb_model': xgb_model, 'feature_names': feature_names_sev
            }
            
            print("\nRegression Model Evaluation Summary:")
            for i, model_name in enumerate(models_sev):
                print(f"  {model_name} - R-squared (R2): {r2_scores[i]:.4f}, MAE: {mae_scores[i]:,.2f}")
    
            # 7. Visualization
            metrics_sev = {'MAE': mae_scores, 'R2': r2_scores, 'MSE': mse_scores}
            
            # Plot Metric Comparison Bar Chart
            self.m.plot_metrics(models_sev, metrics_sev, title_suffix="Claim Severity (Regression)")
            
            # 🌟 CORRECTED: Iterate over ALL models to plot feature importance 🌟
            print("\n--- Generating Feature Importance Plots for all Regression Models ---")
            for model_name, model in models_sev_dict.items():
                if model_name == 'Linear Regression': # You can check the name or type
                    # Linear Regression uses the specialized coefficient plot
                    self.m.plot_linear_coefficients(model, feature_names_sev, model_name)
                
                # Decision Tree, Random Forest, and XGBoost all use the standard attribute
                elif hasattr(model, 'feature_importances_'):
                    self.m.plot_feature_importance(model, feature_names_sev, model_name)
                
                else:
                    print(f"Skipping feature visualization for {model_name} (Method not supported).")
            # --------------------------------------------------------------------------
            
            return self.severity_results

    def run_probability_model(self):
        """Executes the full Claim Probability (Classification) pipeline."""
        print("\n--- 4.2 Claim Probability (Classification) Pipeline Starting ---")

        # 1. Prepare Data
        X_prob = self.df.drop(['Totalclaims', 'Claim_Flag'], axis=1)
        y_prob = self.df['Claim_Flag']

        # 2. Split Data (Stratified)
        X_train_prob, X_test_prob, y_train_prob, y_test_prob = self.m.split_data(X_prob, y_prob, task='classification')

        # 3. Encode and Scale (New DP instance for full dataset)
        dp_prob = DataProcessor()
        X_train_processed_prob = dp_prob.encode_and_scale(X_train_prob, fit=True)
        X_test_processed_prob = dp_prob.encode_and_scale(X_test_prob, fit=False)
        feature_names_prob = dp_prob.feature_names

        # 4. Train Models
        rfr_clf_model, xgb_clf_model, gbr_clf_model = self.m.train_models(X_train_processed_prob, y_train_prob, task='classification')
        models_prob = ['Random Forest', 'XGBoost', 'Gradient Boosting']

        # 5. Evaluate Models
        auc_scores, f1_scores = [], []
        for model in [rfr_clf_model, xgb_clf_model, gbr_clf_model]:
            auc, f1 = self.m.evaluate_model(model, X_test_processed_prob, y_test_prob, task='classification')
            auc_scores.append(auc)
            f1_scores.append(f1)

        # 6. Store and Print Summary
        self.probability_results = {
            'models': models_prob, 'auc': auc_scores, 'f1': f1_scores, 
            'xgb_model': xgb_clf_model, 'feature_names': feature_names_prob
        }

        print("\nClassification Model Evaluation Summary:")
        for i, model_name in enumerate(models_prob):
            print(f"  {model_name} - ROC AUC Score: {auc_scores[i]:.4f}, F1-Score: {f1_scores[i]:.4f}")

        # 7. Visualization
        metrics_prob = {'ROC AUC': auc_scores, 'F1-Score': f1_scores}
        self.m.plot_metrics(models_prob, metrics_prob, title_suffix="Claim Probability (Classification)")
        
        # Still only plotting XGBoost here, which is standard for classification pipelines unless otherwise requested
        self.m.plot_feature_importance(xgb_clf_model, feature_names_prob, "XGBoost Classifier")
        

        return self.probability_results
        
    def run_all(self):
        """Runs both severity and probability modeling pipelines."""
        self.run_severity_model()
        self.run_probability_model()
        print("\n--- END-TO-END MODELING PIPELINE COMPLETE ---")