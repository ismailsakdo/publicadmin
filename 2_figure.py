import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set style for academic publication
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.5)

def create_figure_1():
    """Figure 1: Administrative Latency Analysis (Cloud vs Edge)"""
    plt.figure(figsize=(10, 6))
    stages = ['Sensor Trigger', 'Gateway', 'Cloud Logic', 'Admin Notification', 'HVAC Action']
    cloud_latency = [1, 5, 20, 60, 10] # Representative seconds
    edge_latency = [1, 0.01, 0.05, 0, 1] # Representative seconds
    
    x = np.arange(len(stages))
    plt.bar(x - 0.2, cloud_latency, 0.4, label='Traditional Cloud (Minutes)', color='red', alpha=0.7)
    plt.bar(x + 0.2, edge_latency, 0.4, label='tinyML Edge (Instant)', color='green', alpha=0.7)
    
    plt.yscale('log')
    plt.xticks(x, stages)
    plt.ylabel('Response Latency (Seconds - Log Scale)')
    plt.title('Figure 1: Administrative Latency Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig('figure_1_latency_analysis.png')
    plt.close()

def create_figure_2():
    """Figure 2: Heatmap of Symptom Prevalence"""
    data = {
        'Floor': ['Ground', '1st', '2nd', '3rd'],
        'Fatigue': [52, 48, 44, 56],
        'Headache': [18, 22, 19, 25],
        'Concentration': [31, 35, 28, 40],
        'Irritation': [15, 18, 14, 21]
    }
    df = pd.DataFrame(data).set_index('Floor')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, cmap='YlOrRd', fmt='g')
    plt.title('Figure 2: Prevalence of SBS Symptoms by Floor Level (%)')
    plt.tight_layout()
    plt.savefig('figure_2_symptom_heatmap.png')
    plt.close()

def create_figure_3():
    """Figure 3: Objective vs. Subjective IAQ Perception (Radar Chart)"""
    labels = ['CO2 Level', 'Temp Control', 'Noise', 'Odour', 'Dust']
    # Normalized scores (0-1) where 1 is "High Concern"
    subjective = [0.8, 0.7, 0.9, 0.4, 0.6]
    objective = [0.6, 0.4, 0.7, 0.3, 0.5]
    
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    subjective += subjective[:1]
    objective += objective[:1]
    angles += angles[:1]
    
    # Increase figsize to give more room for labels and legend
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Plot subjective data
    ax.fill(angles, subjective, color='red', alpha=0.25, label='Subjective Perception')
    ax.plot(angles, subjective, color='red', linewidth=2)
    
    # Plot objective data
    ax.fill(angles, objective, color='blue', alpha=0.25, label='Objective Sensor Data')
    ax.plot(angles, objective, color='blue', linewidth=2)
    
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    
    # Fix overlap: Add padding to title and position legend outside the axes
    plt.title('Figure 3: Objective vs. Subjective IAQ Stressors', pad=40, fontsize=16, fontweight='bold')
    
    # bbox_to_anchor coordinates adjusted to prevent overlap with title
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12, frameon=True)
    
    plt.tight_layout()
    # Save with Figure 3 naming convention as per Chapter 4 text
    plt.savefig('figure_3_radar_perception.png', bbox_inches='tight')
    plt.close()

def create_figure_4():
    """Figure 4: The SMOTE Effect on Class Distribution"""
    classes = ['Healthy (0)', 'Health Risk (1)']
    before = [150, 20]
    after = [150, 150]
    
    x = np.arange(len(classes))
    plt.figure(figsize=(10, 6))
    plt.bar(x - 0.2, before, 0.4, label='Pre-SMOTE (Imbalanced)', color='gray')
    plt.bar(x + 0.2, after, 0.4, label='Post-SMOTE (Optimized)', color='blue')
    
    plt.xticks(x, classes)
    plt.ylabel('Number of Instances')
    plt.title('Figure 4: Data Augmentation for Minority Risk Classes')
    plt.legend()
    plt.tight_layout()
    plt.savefig('figure_4_smote_effect.png')
    plt.close()

def create_figure_5():
    """Figure 5: Model Explainability (Feature Importance)"""
    features = ['CO2', 'Temp', 'Humidity', 'Work Stress', 'Noise', 'TVOC']
    importance = [0.35, 0.22, 0.15, 0.12, 0.08, 0.08]
    
    df = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values(by='Importance', ascending=True)
    
    plt.figure(figsize=(10, 6))
    plt.barh(df['Feature'], df['Importance'], color='teal')
    plt.xlabel('Administrative Impact Score (Weight)')
    plt.title('Figure 5: Environmental Predictors of Health Risk (Logistic Regression Weights)')
    plt.tight_layout()
    plt.savefig('figure_5_feature_importance.png')
    plt.close()

def create_figure_6():
    """Figure 6: The Resilient Governance Framework (Flow Chart)"""
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax.axis('off')

    # Define box properties for a professional academic look
    box_props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='teal', linewidth=2)
    
    # Define the steps of the framework
    steps = [
        "Socio-Technical Inputs\n(Sensors + Stress Data)",
        "tinyML Edge Engine\n(Embedded Logistic Regression)",
        "Risk Assessment\n(High Recall Prediction)",
        "Resilient Policy Action\n(Autonomous HVAC/Alerts)"
    ]
    
    # Defined positions for the flow chart boxes
    x_pos = [0.12, 0.38, 0.64, 0.90]
    
    for i, step in enumerate(steps):
        # Place text boxes
        ax.text(x_pos[i], 0.5, step, ha='center', va='center', bbox=box_props, fontsize=12, transform=ax.transAxes)
        
        # Add arrows between boxes
        if i < len(steps) - 1:
            ax.annotate('', xy=(x_pos[i+1]-0.08, 0.5), xytext=(x_pos[i]+0.08, 0.5),
                        xycoords='axes fraction', textcoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', color='teal', lw=2))

    plt.title('Figure 6: The Resilient Governance Framework (Edge-Based Policy Flow)', pad=20, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figure_6_governance_framework.png', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Generating Figures for Journal Submission...")
    create_figure_1()
    create_figure_2()
    create_figure_3()
    create_figure_4()
    create_figure_5()
    create_figure_6()
    print("All figures successfully saved as PNG files with corrected formatting.")
