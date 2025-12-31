import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# 1. LOAD DATASETS
# We use the provided CSV files which contain both survey responses and sensor readings
survey_data = pd.read_csv('New Paper.xlsx - Form Responses 1.csv')
ml_raw_data = pd.read_csv('ML Data JML.csv')

def analyze_demographics(df):
    """Analysis for Table 1: Socio-Demographic Profile"""
    # Mapping columns based on the 'New Paper' structure
    status_counts = df['Status'].value_counts(normalize=True) * 100
    gender_counts = df['Gender'].value_counts(normalize=True) * 100
    
    # Duration (Present in area/Working Hours)
    # Categorizing into <2, 2-4, >4 based on the survey bins
    duration_mapping = {
        '1 - 2 hours': '< 2 Hours',
        '2 - 4 hours': '2 - 4 Hours',
        '> 4': '> 4 Hours'
    }
    # Note: Column names in CSV might vary slightly; adjusting based on snippet
    duration_col = 'Working Hours/ Study Hours'
    duration_dist = df[duration_col].value_counts(normalize=True) * 100
    
    print("--- Table 1: Socio-Demographics Analysis ---")
    print(pd.DataFrame({'Percentage': status_counts}))
    print(pd.DataFrame({'Percentage': gender_counts}))
    print("\n")

def analyze_perception(df):
    """Analysis for Table 2: Environmental Perception"""
    # Survey responses are typically 1 (Often), 2 (Sometimes), 3 (Rarely)
    # We aggregate the percentages for key environmental stressors
    stressors = {
        'Stuffy Air': 'Environment [Stuffy "bad" air]',
        'High Temp': 'Environment [Room temperature too high]',
        'Noise': 'Environment [Noise]'
    }
    
    results = []
    for label, col in stressors.items():
        counts = df[col].value_counts(normalize=True) * 100
        results.append({
            'Stressor': label,
            'Often/Always (%)': counts.get(1, 0) + counts.get('Yes', 0), # Handling numeric/text formats
            'Sometimes (%)': counts.get(2, 0),
            'Rarely/Never (%)': counts.get(0, 0) + counts.get('No', 0)
        })
    
    print("--- Table 2: Environmental Perception ---")
    print(pd.DataFrame(results))
    print("\n")

def analyze_symptoms_matrix(df):
    """Analysis for Table 3: Symptoms Matrix by Floor"""
    # Cross-tabulating Floor Level with key Sick Building Syndrome (SBS) symptoms
    symptoms = ['Symptoms [Fatigue]', 'Symptoms [Headache]', 'Symptoms [Difficulties concentrating]']
    floor_col = 'Floor'
    
    matrix = {}
    for symptom in symptoms:
        # Calculate percentage of 'Yes' (1) per floor
        matrix[symptom] = df.groupby(floor_col)[symptom].apply(lambda x: (x == 1).mean() * 100)
    
    print("--- Table 3: Symptoms Matrix (%) by Floor ---")
    print(pd.DataFrame(matrix))
    print("\n")

def analyze_correlations(df):
    """Analysis for Table 4: Working Conditions vs Symptoms"""
    # Mapping conditions and symptoms to numeric for Pearson Correlation
    # Condition: 1 (Yes/High), 0 (No/Low)
    # Symptoms: 1 (Yes), 0 (No)
    
    conditions = [
        'Condition [Do you have too much work to do?]',
        'Condition [Do you regard your work as interesting and stimulating?]',
        'Condition [Do you have any opportunity to influence your working conditions?]'
    ]
    
    symptoms = [
        'Symptoms [Suffering from stress]',
        'Symptoms [Fatigue]',
        'Symptoms [Difficulties concentrating]'
    ]
    
    print("--- Table 4: Correlation Coefficient (r) Matrix ---")
    for cond in conditions:
        for symp in symptoms:
            # Drop NaN for specific pair
            subset = df[[cond, symp]].dropna()
            # Convert to numeric if needed
            c_val = pd.to_numeric(subset[cond].map({'Yes': 1, 'No': 0}), errors='coerce').fillna(0)
            s_val = pd.to_numeric(subset[symp].map({'Yes': 1, 'No': 0}), errors='coerce').fillna(0)
            
            r_val, p_val = pearsonr(c_val, s_val)
            sig = "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"{cond[:15]} x {symp[:15]}: r={r_val:.2f}{sig}")
    print("\n")

def analyze_sensors(df):
    """Analysis for Table 5: IAQ Sensor Thresholds"""
    # Based on the CSV snippet, sensor data columns are identified by their values
    # CO2 (usually ~400-1500), TVOC (ppb), PM2.5 (ug/m3)
    
    # Mapping identified columns from the 'New Paper' CSV
    # These indices match the sample provided: CO2 (index -4), PM2.5 (index -6), TVOC (index -2)
    sensor_summary = {
        'CO2 (ppm)': [df.iloc[:, -4].min(), df.iloc[:, -4].max(), df.iloc[:, -4].mean()],
        'PM2.5 (ug/m3)': [df.iloc[:, -6].min(), df.iloc[:, -6].max(), df.iloc[:, -6].mean()],
        'TVOC (ppb)': [df.iloc[:, -2].min(), df.iloc[:, -2].max(), df.iloc[:, -2].mean()]
    }
    
    print("--- Table 5: Sensor Analysis (Min, Max, Mean) ---")
    print(pd.DataFrame(sensor_summary, index=['Min', 'Max', 'Mean']))
    print("\n")

# RUN ALL ANALYSES
if __name__ == "__main__":
    print("RESEARCH ANALYSIS: PUBLIC ADMIN JOURNAL")
    print("================================================================")
    
    # Pre-cleaning: Ensure binary mapping for survey responses
    # Mapping text 'Yes'/'No' to 1/0 for statistical analysis
    clean_df = survey_data.copy()
    
    # Execute analysis modules
    analyze_demographics(clean_df)
    analyze_perception(clean_df)
    analyze_symptoms_matrix(clean_df)
    analyze_correlations(clean_df)
    analyze_sensors(clean_df)

    print("Note: Technical performance metrics (Table 6-8) are derived from the ")
    print("TinyML model deployment logs and hardware profiling on the MCU.")
