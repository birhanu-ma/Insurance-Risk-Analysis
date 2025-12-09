import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

class DataProcessor:
    """Handles data preparation, feature engineering, and preprocessing pipelines."""
    def __init__(self):
        self.preprocessor = None
        self.feature_names = None

    def feature_engineer(self, data):
        """Adds VehicleAge and drops unnecessary high-cardinality/date columns."""
        temp_df = data.copy()
        
        # 1. Vehicle Age calculation (Crucial: This is where 'VehicleAge' is created)
        if 'Vehicleintrodate' in temp_df.columns:
            temp_df['Vehicleintrodate'] = pd.to_datetime(temp_df['Vehicleintrodate'], errors='coerce')
            try:
                temp_df['VehicleAge'] = (pd.to_datetime('2023-01-01') - temp_df['Vehicleintrodate']).dt.days / 365.25
                temp_df.drop('Vehicleintrodate', axis=1, inplace=True)
            except Exception:
                pass 
        
        # 2. Drop high-cardinality and irrelevant IDs/dates
        cols_to_drop = [
            'Underwrittencoverid', 'Policyid', 'Transactionmonth', 'Postalcode', 
            'Make', 'Model', 'Maincrestazone', 'Subcrestazone', 'Country', 'Citizenship', 
            'Legaltype', 'Title', 'Language', 'Accounttype', 'MaritalStatus', 
            'Itemtype', 'Mmcode', 'Vehicletype', 'Bodytype', 'Covercategory',
            'CoverType', 'CoverGroup', 'Section', 'Product', 'StatutoryClass', 
            'StatutoryRiskType', 'CalculatedPremiumPerTerm', 'TotalPremium'
        ]
        
        temp_df.drop(columns=[col for col in cols_to_drop if col in temp_df.columns], 
                     axis=1, 
                     inplace=True, 
                     errors='ignore')
        return temp_df

    def encode_and_scale(self, X_data, fit=True):
        """
        Applies feature engineering, encoding, and scaling.
        """
        
        # --- 1. ENGINEER THE DATA FIRST ---
        X_eng = self.feature_engineer(X_data)
        
        # --- 2. AGGRESSIVE ROBUST TYPE CASTING (Mixed-type string/int fix) ---
        SUSPECTED_MIXED_COLS = [
            'CapitalOutstanding', 'AlarmImmobiliser', 'TrackingDevice', 
            'NewVehicle', 'WrittenOff', 'Rebuilt', 'Converted', 'CrossBorder'
        ]
        
        for col in X_eng.columns:
            # 2a. Convert known mixed-type/numeric columns (including existing int/float) to numeric.
            if col in SUSPECTED_MIXED_COLS or X_eng[col].dtype not in ['object', 'bool']:
                X_eng[col] = pd.to_numeric(X_eng[col], errors='coerce')
                
            # 2b. Ensure all remaining object columns (true categories) are homogeneous strings.
            elif X_eng[col].dtype == 'object':
                 X_eng[col] = X_eng[col].astype(str)

        # --- 3. DEFINE COLUMNS AFTER CLEANING TYPES ---
        numerical_cols = X_eng.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = X_eng.select_dtypes(include=['object', 'bool']).columns.tolist()

        # Define transformers
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ],
            remainder='passthrough'
        )

        if fit:
            # Fit and transform the ENGINEERED data (X_eng)
            self.preprocessor = preprocessor.fit(X_eng)
            
            # 💡 FINAL FIX: Use get_feature_names_out() to correctly capture ALL feature names
            # (including the 'passthrough' remainder columns that caused the length mismatch)
            all_feature_names = self.preprocessor.get_feature_names_out()
            
            # Clean the names: remove the transformer prefixes (e.g., 'num__', 'cat__', 'remainder__')
            self.feature_names = [name.split("__")[-1] for name in all_feature_names]

            X_processed = self.preprocessor.transform(X_eng)
            
        else:
            # Transform the ENGINEERED data (X_eng)
            X_processed = self.preprocessor.transform(X_eng)
        
        return X_processed