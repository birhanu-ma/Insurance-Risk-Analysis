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
        # Margin is the profit/loss: Premium - Claims
        self.df['Margin'] = self.df['Totalpremium'] - self.df['Totalclaims']
        
        # Severity is TotalClaims for policies where claims > 0
        self.df_claims = self.df[self.df['Totalclaims'] > 0].copy()
        
        # Ensure 'Gender' and 'Province' are ready for grouping
        self.df['Gender'] = self.df['Gender'].astype('category')
        self.df['Province'] = self.df['Province'].astype('category')
        
        print("💡 StatisticalTester initialized. Claim_Flag and Margin columns created.")
        
    # --- HELPER FUNCTIONS ---
    
    def _run_ttest_or_mwu(self, group_a, group_b, metric_col, group_name):
        """
        Runs a Mann-Whitney U test (non-parametric comparison of means).
        Returns statistic, p_value, and observation string.
        """
        # Clean data (remove NaN/inf for robustness)
        a = group_a[metric_col].dropna().replace([np.inf, -np.inf], np.nan).dropna()
        b = group_b[metric_col].dropna().replace([np.inf, -np.inf], np.nan).dropna()
        
        # Check if groups are too small for comparison
        if len(a) < 2 or len(b) < 2:
            return None, 1.0, f"Insufficient data (A:{len(a)}, B:{len(b)}) for {group_name}"
        
        # Mann-Whitney U Test (often safer than T-Test for non-normal insurance data)
        u_stat, p_value = mannwhitneyu(a, b, alternative='two-sided')
        
        return u_stat, p_value, f"A_Mean: {a.mean():,.2f}, B_Mean: {b.mean():,.2f}"

    def _run_chi2_test(self, group_a, group_b, group_name):
        """
        Runs a Chi-Squared test on Claim_Flag (frequency) for two groups.
        Returns statistic, p_value, and observation string.
        """
        # Create the contingency table
        table = pd.DataFrame({
            'Group A': group_a['Claim_Flag'].value_counts().sort_index(),
            'Group B': group_b['Claim_Flag'].value_counts().sort_index()
        })
        
        # Ensure both 0 (No Claim) and 1 (Claim) are present for a 2x2 table
        if 0 not in table.index: table.loc[0] = 0
        if 1 not in table.index: table.loc[1] = 0
        table = table.fillna(0).astype(int).loc[[0, 1]]
        
        # Check if all values are zero (no data)
        if table.sum().sum() == 0:
             return None, 1.0, "No claim data found for comparison."
             
        chi2, p_value, dof, expected = chi2_contingency(table)
        
        freq_a = group_a['Claim_Flag'].mean() * 100
        freq_b = group_b['Claim_Flag'].mean() * 100
        
        return chi2, p_value, f"A_Freq: {freq_a:.2f}%, B_Freq: {freq_b:.2f}%"

    def _format_p_value(self, p):
        """Formats P-Value to 4 significant figures or scientific notation."""
        if p < 0.0001 and p != 0.0:
            return f"{p:.2e}"
        elif p == 1.0:
            return "1.000"
        else:
            return f"{p:.4f}"

    def _format_result_dict(self, hypothesis, metric, stat_name, statistic, p_value, observation):
        """Helper to create the standardized result dictionary."""
        
        # Set a standard significance level (alpha)
        alpha = 0.05
        
        if p_value < alpha:
            result = f"Reject H₀ (P < {alpha})"
        else:
            result = f"Fail to Reject H₀ (P ≥ {alpha})"
            
        return {
            'Metric': metric,
            'Test': stat_name,
            'Statistic': statistic,
            'p_value': p_value,
            'Observation': observation,
            'Result': result
        }

    # --- HYPOTHESIS TEST METHODS ---

    def test_province_risk(self):
        """
        H₀: Claim probability and average severity are the same across the two most frequent provinces.
        Compares Claim_Flag (Chi2) and Totalclaims (MWU) for the top two provinces.
        """
        print("\n--- Running Province Risk Test ---")
        
        # Find the two most frequent provinces
        prov_counts = self.df['Province'].value_counts().nlargest(2)
        if len(prov_counts) < 2:
            return {'Hypothesis': 'Province Risk Test (Insufficient Data)', 'Tests': []}

        prov_a, prov_b = prov_counts.index[0], prov_counts.index[1]
        
        group_a = self.df[self.df['Province'] == prov_a]
        group_b = self.df[self.df['Province'] == prov_b]
        
        hyp_name = f"Claim metrics are equal for Province A ({prov_a}) vs B ({prov_b})"
        
        # 1. Claim Probability (Risk) - Chi-Squared Test
        chi2, p_risk, obs_risk = self._run_chi2_test(group_a, group_b, f'{prov_a} vs {prov_b}')
        results_list = [
            self._format_result_dict(hyp_name, 'Claim Probability (Risk)', 'Chi-Squared', chi2, p_risk, obs_risk)
        ]
        
        # 2. Claim Severity - Mann-Whitney U Test (Focus on claims > 0)
        # Use the claim-only subset for severity comparison
        claims_a_sev = self.df_claims[self.df_claims['Province'] == prov_a]
        claims_b_sev = self.df_claims[self.df_claims['Province'] == prov_b]
        
        u_sev, p_sev, obs_sev = self._run_ttest_or_mwu(claims_a_sev, claims_b_sev, 'Totalclaims', f'{prov_a} vs {prov_b} (Claims)')
        results_list.append(
            self._format_result_dict(hyp_name, 'Claim Severity (Claims > 0)', 'MWU', u_sev, p_sev, obs_sev)
        )
        
        return {'Hypothesis': hyp_name, 'Tests': results_list}


    def test_zipcode_risk(self, zip_a, zip_b):
        """
        H₀: Claim probability is the same for the two specific zip codes (zip_a, zip_b).
        Compares Claim_Flag (Chi2).
        """
        print(f"--- Running Zip Code Risk Test ({zip_a} vs {zip_b}) ---")
        
        group_a = self.df[self.df['Postalcode'] == zip_a]
        group_b = self.df[self.df['Postalcode'] == zip_b]
        
        hyp_name = f"Claim probability is equal for Postalcode A ({zip_a}) vs B ({zip_b})"
        
        # 1. Claim Probability (Risk) - Chi-Squared Test
        chi2, p_risk, obs_risk = self._run_chi2_test(group_a, group_b, f'{zip_a} vs {zip_b}')
        
        results_list = [
            self._format_result_dict(hyp_name, 'Claim Probability (Risk)', 'Chi-Squared', chi2, p_risk, obs_risk)
        ]
        
        return {'Hypothesis': hyp_name, 'Tests': results_list}


    def test_zipcode_margin(self, zip_a, zip_b):
        """
        H₀: Average policy margin (Premium - Claims) is the same for the two specific zip codes.
        Compares Margin (MWU).
        """
        print(f"--- Running Zip Code Margin Test ({zip_a} vs {zip_b}) ---")
        
        group_a = self.df[self.df['Postalcode'] == zip_a]
        group_b = self.df[self.df['Postalcode'] == zip_b]
        
        hyp_name = f"Average policy margin is equal for Postalcode A ({zip_a}) vs B ({zip_b})"
        
        # 1. Margin - Mann-Whitney U Test
        u_margin, p_margin, obs_margin = self._run_ttest_or_mwu(group_a, group_b, 'Margin', f'{zip_a} vs {zip_b}')
        
        results_list = [
            self._format_result_dict(hyp_name, 'Policy Margin (Premium-Claims)', 'MWU', u_margin, p_margin, obs_margin)
        ]
        
        return {'Hypothesis': hyp_name, 'Tests': results_list}
        
    
    def test_gender_risk(self):
        """
        H₀: Claim probability and average severity are the same between the top two genders.
        Compares Claim_Flag (Chi2) and Totalclaims (MWU).
        """
        print("\n--- Running Gender Risk Test ---")
        
        # Find the two most frequent genders (assuming Male/Female, but generalized)
        gender_counts = self.df['Gender'].value_counts().nlargest(2)
        if len(gender_counts) < 2:
            return {'Hypothesis': 'Gender Risk Test (Insufficient Data)', 'Tests': []}
            
        gender_a, gender_b = gender_counts.index[0], gender_counts.index[1]
        
        group_a = self.df[self.df['Gender'] == gender_a]
        group_b = self.df[self.df['Gender'] == gender_b]

        hyp_name = f"Claim metrics are equal for Gender A ({gender_a}) vs B ({gender_b})"
        
        # 1. Claim Probability (Risk) - Chi-Squared Test
        chi2, p_risk, obs_risk = self._run_chi2_test(group_a, group_b, f'{gender_a} vs {gender_b}')
        results_list = [
            self._format_result_dict(hyp_name, 'Claim Probability (Risk)', 'Chi-Squared', chi2, p_risk, obs_risk)
        ]
        
        # 2. Claim Severity - Mann-Whitney U Test (Focus on claims > 0)
        # Use the claim-only subset for severity comparison
        claims_a_sev = self.df_claims[self.df_claims['Gender'] == gender_a]
        claims_b_sev = self.df_claims[self.df_claims['Gender'] == gender_b]

        u_sev, p_sev, obs_sev = self._run_ttest_or_mwu(claims_a_sev, claims_b_sev, 'Totalclaims', f'{gender_a} vs {gender_b} (Claims)')
        results_list.append(
            self._format_result_dict(hyp_name, 'Claim Severity (Claims > 0)', 'MWU', u_sev, p_sev, obs_sev)
        )
        
        return {'Hypothesis': hyp_name, 'Tests': results_list}


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
            # Use the full hypothesis string for the first row of each test set
            hyp_display = row['Hypothesis']
            
            if row['Hypothesis_Base'] == current_hyp:
                # Blank the Hypothesis column for subsequent rows of the same test set
                hyp_display = '' 
            else:
                current_hyp = row['Hypothesis_Base']
            
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
        print(f"\n*Decision based on significance level $\\alpha$ = 0.05. Reject H₀ means a significant difference was found.")

        return df_final # Return the formatted DataFrame for display/use
        

    # --- MAIN EXECUTION METHOD ---
    def run_all_tests(self):
        """Runs all required hypothesis tests and calls display_results_summary."""
        
        # Identify top two zip codes for comparison
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
        # We return the formatted DataFrame for clean output
        return self.display_results_summary(all_results)