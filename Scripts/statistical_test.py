import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu

class StatisticalTester:
    
    def __init__(self, df):
        """Initialize the tester and prepare risk/margin features."""
        self.df = df.copy()
        
        # 1. Feature Engineering for Risk Metrics
        self.df['Totalclaims'] = pd.to_numeric(self.df['Totalclaims'], errors='coerce').fillna(0)
        self.df['Totalpremium'] = pd.to_numeric(self.df['Totalpremium'], errors='coerce').fillna(0)
        
        self.df['Claim_Flag'] = np.where(self.df['Totalclaims'] > 0, 1, 0)
        self.df['Margin'] = self.df['Totalpremium'] - self.df['Totalclaims']
        
        # Severity is TotalClaims for policies where claims > 0
        self.df_claims = self.df[self.df['Totalclaims'] > 0].copy()
        print("💡 StatisticalTester initialized. Claim_Flag and Margin columns created.")
        
    def _run_ttest_or_mwu(self, group_a, group_b, metric_col, group_name):
        # ... (same as original, omitted for brevity)
        # Clean data (remove NaN/inf for robustness)
        a = group_a[metric_col].dropna().replace([np.inf, -np.inf], np.nan).dropna()
        b = group_b[metric_col].dropna().replace([np.inf, -np.inf], np.nan).dropna()
        
        # Check if groups are too small for comparison
        if len(a) < 2 or len(b) < 2:
            return None, 1.0, f"Insufficient data for {group_name}"
        
        u_stat, p_value = mannwhitneyu(a, b, alternative='two-sided')
        return u_stat, p_value, f"A_Mean: {a.mean():,.2f}, B_Mean: {b.mean():,.2f}"

    def _run_chi2_test(self, group_a, group_b, group_name):
        # ... (same as original, omitted for brevity)
        table = pd.DataFrame({
            'Group A': group_a['Claim_Flag'].value_counts().sort_index(),
            'Group B': group_b['Claim_Flag'].value_counts().sort_index()
        })
        
        if 0 not in table.index: table.loc[0] = 0
        if 1 not in table.index: table.loc[1] = 0
        table = table.fillna(0).astype(int)
        
        chi2, p_value, dof, expected = chi2_contingency(table)
        
        freq_a = group_a['Claim_Flag'].mean() * 100
        freq_b = group_b['Claim_Flag'].mean() * 100
        
        return chi2, p_value, f"A_Freq: {freq_a:.2f}%, B_Freq: {freq_b:.2f}%"

    # --- NEW HELPER METHOD ---
    def _format_p_value(self, p):
        """Formats P-Value to 4 significant figures or scientific notation."""
        if p < 0.0001 and p != 0.0:
            return f"{p:.2e}"
        elif p == 1.0:
            return "1.000"
        else:
            return f"{p:.4f}"

    # --- NEW DISPLAY METHOD ---
    def display_results_summary(self, all_results):
        """Parses the raw test output and prints the structured summary table."""
        
        flattened_data = []
        for item in all_results:
            hypothesis_base = item['Hypothesis']
            for test in item['Tests']:
                row = {
                    'Hypothesis': hypothesis_base,
                    'Metric': test['Metric'],
                    'Observation': test['Observation'],
                    'P-Value': test['p_value'],
                    'Decision': test['Result']
                }
                flattened_data.append(row)

        df_results = pd.DataFrame(flattened_data)

        # Apply custom formatting
        df_results['P-Value'] = df_results['P-Value'].apply(self._format_p_value)

        # Prepare for final display table (grouping by Hypothesis)
        df_results['Hypothesis_Base'] = df_results['Hypothesis'].apply(
            lambda x: x.split('(')[0].strip() if '(' in x else x
        )

        final_cols = ['Hypothesis', 'Metric', 'P-Value', 'Observation', 'Decision']
        df_final = pd.DataFrame(columns=final_cols)
        current_hyp = None

        for index, row in df_results.iterrows():
            hyp_display = row['Hypothesis_Base']
            
            if hyp_display != current_hyp:
                current_hyp = hyp_display
            else:
                hyp_display = '' 
            
            df_final.loc[index] = [
                hyp_display,
                row['Metric'],
                row['P-Value'],
                row['Observation'],
                row['Decision']
            ]

        # Rename and print the final table
        df_final = df_final.rename(columns={
            'Hypothesis': 'Null Hypothesis (H₀)',
            'Observation': 'Test Observation (A vs B)'
        })
        
        print("\n--- SUMMARY OF A/B HYPOTHESIS TEST RESULTS (Task 3) ---")
        print(df_final.to_markdown(index=False))

    # --- MODIFIED RUN METHOD ---
    def run_all_tests(self):
        """Runs all required hypothesis tests and calls display_results_summary."""
        
        # Identify top two zip codes for comparison (Task 3 requirement)
        zip_counts = self.df['Postalcode'].value_counts().nlargest(2)
        if len(zip_counts) < 2:
            print("Error: Not enough distinct Postal Codes for comparison.")
            return []
            
        zip_a, zip_b = zip_counts.index[0], zip_counts.index[1]
        
        print(f"\n🔬 Running Statistical Hypothesis Tests (Task 3)...")
        print(f"Comparing Top 2 Zip Codes: {zip_a} and {zip_b}")
        
        all_results = []
        all_results.append(self.test_province_risk())
        all_results.append(self.test_zipcode_risk(zip_a, zip_b))
        all_results.append(self.test_zipcode_margin(zip_a, zip_b))
        all_results.append(self.test_gender_risk())
        
        # Display the structured results table
        self.display_results_summary(all_results)
        
        return all_results # Return the raw list for further processing if needed

    # The other test methods (test_province_risk, test_zipcode_risk, etc.)
    # remain the same, ensuring they return the dictionary structure.