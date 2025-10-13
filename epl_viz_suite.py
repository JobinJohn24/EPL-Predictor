"""
EPL Match Predictor - Complete Visualization Suite
Generates 6 professional-quality visualizations for GitHub README

Requirements:
    matplotlib>=3.5.0
    seaborn>=0.11.0
    numpy>=1.21.0
    pandas>=1.3.0

Usage:
    python epl_visualizations.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import seaborn as sns
import numpy as np
import pandas as pd
import os
from pathlib import Path

# Color palette
COLORS = {
    'primary': '#1f77b4',
    'teal': '#2E86AB',
    'orange': '#ff7f0e',
    'red': '#d62728',
    'gray': '#7f7f7f',
    'light_gray': '#bcbcbc',
    'green': '#2ca02c',
    'dark_bg': '#1a1d29',
    'dark_card': '#242837',
    'accent': '#00d9ff'
}

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

def generate_realistic_data(n_matches=500):
    """Generate realistic EPL match prediction data"""
    np.random.seed(42)
    
    # Outcome probabilities: 45% Home Win, 30% Draw, 25% Away Win
    actual_outcomes = np.random.choice(['Win', 'Draw', 'Loss'], 
                                       size=n_matches, 
                                       p=[0.45, 0.30, 0.25])
    
    # Model predictions with ~60% accuracy
    predicted_outcomes = actual_outcomes.copy()
    # Introduce errors (40% of predictions)
    error_indices = np.random.choice(n_matches, size=int(n_matches * 0.40), replace=False)
    for idx in error_indices:
        current = predicted_outcomes[idx]
        alternatives = [x for x in ['Win', 'Draw', 'Loss'] if x != current]
        predicted_outcomes[idx] = np.random.choice(alternatives)
    
    # Generate prediction probabilities
    probabilities = []
    for pred, actual in zip(predicted_outcomes, actual_outcomes):
        if pred == actual:
            # Correct predictions: higher confidence (50-85%)
            prob = np.random.uniform(0.50, 0.85)
        else:
            # Incorrect predictions: lower confidence (40-65%)
            prob = np.random.uniform(0.40, 0.65)
        probabilities.append(prob)
    
    # Generate feature importance data
    features = {
        'xG_differential': 0.185,
        'form_differential': 0.162,
        'shot_efficiency': 0.143,
        'win_rate_differential': 0.128,
        'goals_differential': 0.095,
        'home_form_last5': 0.087,
        'away_form_last5': 0.076,
        'set_piece_advantage': 0.054,
        'rest_advantage': 0.041,
        'defensive_differential': 0.038,
        'possession_differential': 0.032,
        'corners_differential': 0.028,
        'head_to_head': 0.024,
        'league_position_diff': 0.021,
        'injuries_impact': 0.018
    }
    
    return {
        'actual': actual_outcomes,
        'predicted': predicted_outcomes,
        'probabilities': np.array(probabilities),
        'features': features,
        'n_matches': n_matches
    }

def create_output_dir():
    """Create visualizations directory"""
    Path('visualizations').mkdir(exist_ok=True)

def part1_performance_dashboard(data):
    """Create 4-panel performance dashboard"""
    print("Generating Part 1: Performance Dashboard...")
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Panel 1: Confusion Matrix
    ax1 = fig.add_subplot(gs[0, 0])
    labels = ['Win', 'Draw', 'Loss']
    cm = pd.crosstab(pd.Series(data['actual'], name='Actual'),
                     pd.Series(data['predicted'], name='Predicted'),
                     normalize='index')
    cm = cm.reindex(index=labels, columns=labels, fill_value=0)
    
    sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', 
                cbar_kws={'label': 'Percentage'}, ax=ax1,
                square=True, linewidths=2, linecolor='white')
    ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylabel('Actual Outcome', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Predicted Outcome', fontsize=11, fontweight='bold')
    
    # Panel 2: Classification Metrics
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Calculate metrics for each class
    metrics_data = []
    for label in labels:
        true_pos = np.sum((data['actual'] == label) & (data['predicted'] == label))
        false_pos = np.sum((data['actual'] != label) & (data['predicted'] == label))
        false_neg = np.sum((data['actual'] == label) & (data['predicted'] != label))
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_data.append({'Class': label, 'Precision': precision, 
                           'Recall': recall, 'F1-Score': f1})
    
    metrics_df = pd.DataFrame(metrics_data)
    x = np.arange(len(labels))
    width = 0.25
    
    ax2.bar(x - width, metrics_df['Precision'], width, label='Precision', 
            color=COLORS['primary'], alpha=0.8)
    ax2.bar(x, metrics_df['Recall'], width, label='Recall', 
            color=COLORS['teal'], alpha=0.8)
    ax2.bar(x + width, metrics_df['F1-Score'], width, label='F1-Score', 
            color=COLORS['green'], alpha=0.8)
    
    ax2.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax2.set_title('Classification Metrics by Outcome', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.set_ylim(0, 1)
    ax2.grid(axis='y', alpha=0.3)
    
    # Panel 3: Training vs Testing Accuracy
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Simulate train/test split
    train_acc = 0.68
    test_acc = np.sum(data['actual'] == data['predicted']) / len(data['actual'])
    
    categories = ['Training', 'Testing']
    accuracies = [train_acc, test_acc]
    colors = [COLORS['primary'], COLORS['teal']]
    
    bars = ax3.bar(categories, accuracies, color=colors, alpha=0.8, width=0.5)
    ax3.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax3.set_title('Model Accuracy: Training vs Testing', fontsize=14, fontweight='bold', pad=15)
    ax3.set_ylim(0, 1)
    ax3.axhline(y=0.333, color=COLORS['red'], linestyle='--', 
                alpha=0.5, label='Random Baseline')
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1%}', ha='center', va='bottom', 
                fontsize=12, fontweight='bold')
    
    ax3.legend(loc='lower right')
    ax3.grid(axis='y', alpha=0.3)
    
    # Panel 4: Prediction Confidence Distribution
    ax4 = fig.add_subplot(gs[1, 1])
    
    correct_mask = data['actual'] == data['predicted']
    correct_probs = data['probabilities'][correct_mask]
    incorrect_probs = data['probabilities'][~correct_mask]
    
    ax4.hist(correct_probs, bins=20, alpha=0.7, label='Correct Predictions',
             color=COLORS['green'], edgecolor='black')
    ax4.hist(incorrect_probs, bins=20, alpha=0.7, label='Incorrect Predictions',
             color=COLORS['red'], edgecolor='black')
    
    ax4.set_xlabel('Prediction Confidence', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax4.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold', pad=15)
    ax4.legend(loc='upper right', framealpha=0.9)
    ax4.grid(axis='y', alpha=0.3)
    
    plt.suptitle('EPL Match Predictor - Performance Dashboard', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    plt.savefig('visualizations/performance_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: performance_dashboard.png")

def part2_feature_importance(data):
    """Create feature importance chart"""
    print("Generating Part 2: Feature Importance Analysis...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    features_df = pd.DataFrame(list(data['features'].items()), 
                              columns=['Feature', 'Importance'])
    features_df = features_df.sort_values('Importance', ascending=True)
    
    # Create color gradient
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(features_df)))
    
    bars = ax.barh(features_df['Feature'], features_df['Importance'], 
                   color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, features_df['Importance'])):
        ax.text(val, bar.get_y() + bar.get_height()/2, 
               f' {val:.1%}', va='center', ha='left', 
               fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title('Decision Tree Feature Importance - Top 15 Features', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim(0, max(features_df['Importance']) * 1.15)
    
    # Format feature names
    feature_labels = [f.replace('_', ' ').title() for f in features_df['Feature']]
    ax.set_yticklabels(feature_labels, fontsize=10)
    
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: feature_importance.png")

def part3_probability_distributions(data):
    """Create probability distribution curves"""
    print("Generating Part 3: Probability Distribution Curves...")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Generate probability distributions for each outcome
    for outcome, color in [('Win', COLORS['green']), 
                          ('Draw', COLORS['gray']), 
                          ('Loss', COLORS['red'])]:
        outcome_probs = data['probabilities'][data['predicted'] == outcome]
        
        if len(outcome_probs) > 0:
            # Create smooth density curve
            from scipy import stats
            density = stats.gaussian_kde(outcome_probs)
            xs = np.linspace(0.3, 0.9, 200)
            ys = density(xs)
            
            ax.fill_between(xs, ys, alpha=0.3, color=color, label=f'{outcome} Predictions')
            ax.plot(xs, ys, color=color, linewidth=2.5)
    
    ax.set_xlabel('Prediction Confidence', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title('Prediction Confidence Distribution by Outcome', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0.3, 0.9)
    
    # Add vertical line for mean confidence
    mean_conf = np.mean(data['probabilities'])
    ax.axvline(mean_conf, color='black', linestyle='--', 
              label=f'Mean: {mean_conf:.1%}', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('visualizations/probability_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: probability_distributions.png")

def part4_actual_vs_predicted(data):
    """Create actual vs predicted comparison"""
    print("Generating Part 4: Actual vs Predicted Comparison...")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    labels = ['Win', 'Draw', 'Loss']
    actual_counts = [np.sum(data['actual'] == label) for label in labels]
    predicted_counts = [np.sum(data['predicted'] == label) for label in labels]
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, actual_counts, width, label='Actual Outcomes',
                   color=COLORS['primary'], alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, predicted_counts, width, label='Predicted Outcomes',
                   color=COLORS['teal'], alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', 
                   fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Number of Matches', fontsize=12, fontweight='bold')
    ax.set_xlabel('Match Outcome', fontsize=12, fontweight='bold')
    ax.set_title('Actual vs Predicted Match Outcomes', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    
    # Add accuracy text box
    accuracy = np.sum(data['actual'] == data['predicted']) / len(data['actual'])
    textstr = f'Overall Accuracy\n{accuracy:.1%}'
    props = dict(boxstyle='round', facecolor=COLORS['green'], alpha=0.8)
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=13,
           verticalalignment='top', horizontalalignment='right',
           bbox=props, color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualizations/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: actual_vs_predicted.png")

def part5_prediction_card(data):
    """Create sample prediction card infographic"""
    print("Generating Part 5: Sample Prediction Card...")
    
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Background with gradient effect
    ax.add_patch(Rectangle((0, 0), 10, 12, facecolor='#f8f9fa'))
    
    # Main card
    card = FancyBboxPatch((0.5, 1), 9, 10, boxstyle="round,pad=0.1",
                         facecolor='white', edgecolor=COLORS['primary'], 
                         linewidth=3, zorder=1)
    ax.add_patch(card)
    
    # Header
    ax.add_patch(Rectangle((0.5, 9.5), 9, 1.5, facecolor=COLORS['primary'], zorder=2))
    ax.text(5, 10.25, 'MATCH PREDICTION', ha='center', va='center',
           fontsize=22, fontweight='bold', color='white', zorder=3)
    
    # Teams
    ax.text(5, 9, 'Manchester City', ha='center', va='center',
           fontsize=18, fontweight='bold', color=COLORS['primary'])
    ax.text(5, 8.5, 'vs', ha='center', va='center',
           fontsize=14, color=COLORS['gray'])
    ax.text(5, 8, 'Arsenal', ha='center', va='center',
           fontsize=18, fontweight='bold', color=COLORS['red'])
    
    # Prediction outcome
    ax.add_patch(FancyBboxPatch((2, 6.5), 6, 1.2, boxstyle="round,pad=0.1",
                               facecolor=COLORS['green'], alpha=0.2, 
                               edgecolor=COLORS['green'], linewidth=2))
    ax.text(5, 7.1, 'PREDICTED: HOME WIN', ha='center', va='center',
           fontsize=16, fontweight='bold', color=COLORS['green'])
    
    # Confidence
    ax.text(5, 6.3, 'Confidence: 68.5%', ha='center', va='center',
           fontsize=14, color=COLORS['gray'], fontweight='bold')
    
    # Probability breakdown
    ax.text(2.5, 5.5, 'Probability Breakdown:', ha='left', va='center',
           fontsize=12, fontweight='bold')
    
    outcomes = [('Home Win', 0.685, COLORS['green']),
               ('Draw', 0.215, COLORS['gray']),
               ('Away Win', 0.100, COLORS['red'])]
    
    y_pos = 5
    for outcome, prob, color in outcomes:
        # Bar
        bar_width = prob * 6
        ax.add_patch(Rectangle((2.5, y_pos - 0.15), bar_width, 0.3,
                              facecolor=color, alpha=0.6))
        # Label
        ax.text(2.3, y_pos, f'{outcome}:', ha='right', va='center',
               fontsize=11)
        ax.text(8.7, y_pos, f'{prob:.1%}', ha='right', va='center',
               fontsize=11, fontweight='bold')
        y_pos -= 0.5
    
    # Key factors
    ax.text(2.5, 3, 'Key Factors:', ha='left', va='center',
           fontsize=12, fontweight='bold')
    
    factors = [
        '⚡ xG Differential: +0.42',
        '📊 Form Advantage: +1.8',
        '🏠 Home Record: 82% Win',
        '🎯 Shot Efficiency: 65%'
    ]
    
    y_pos = 2.5
    for factor in factors:
        ax.text(2.7, y_pos, factor, ha='left', va='center',
               fontsize=10, color=COLORS['gray'])
        y_pos -= 0.4
    
    # Footer
    ax.text(5, 1.3, 'Generated by EPL Match Predictor ML Model', 
           ha='center', va='center', fontsize=9, 
           color=COLORS['gray'], style='italic')
    
    plt.savefig('visualizations/prediction_card_sample.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: prediction_card_sample.png")

def part6_dashboard_mockup(data):
    """Create interactive dashboard mockup"""
    print("Generating Part 6: Dashboard Mockup...")
    
    fig = plt.figure(figsize=(19.2, 12))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1920)
    ax.set_ylim(0, 1200)
    ax.axis('off')
    ax.set_facecolor(COLORS['dark_bg'])
    fig.patch.set_facecolor(COLORS['dark_bg'])
    
    # Header
    ax.add_patch(Rectangle((0, 1100), 1920, 100, facecolor=COLORS['dark_card']))
    ax.text(960, 1150, 'EPL MATCH PREDICTOR', ha='center', va='center',
           fontsize=36, fontweight='bold', color='white')
    ax.text(960, 1115, 'Powered by Machine Learning Decision Tree Model', 
           ha='center', va='center', fontsize=14, color=COLORS['light_gray'])
    
    # Stats badges
    accuracy = np.sum(data['actual'] == data['predicted']) / len(data['actual'])
    stats = [
        ('Model Accuracy', f'{accuracy:.1%}'),
        ('Total Predictions', f"{data['n_matches']}"),
        ('Avg Confidence', f"{np.mean(data['probabilities']):.1%}")
    ]
    
    x_pos = 300
    for label, value in stats:
        ax.add_patch(FancyBboxPatch((x_pos, 1020), 250, 60, 
                                   boxstyle="round,pad=5",
                                   facecolor=COLORS['accent'], alpha=0.2,
                                   edgecolor=COLORS['accent'], linewidth=2))
        ax.text(x_pos + 125, 1065, value, ha='center', va='center',
               fontsize=20, fontweight='bold', color=COLORS['accent'])
        ax.text(x_pos + 125, 1035, label, ha='center', va='center',
               fontsize=11, color=COLORS['light_gray'])
        x_pos += 400
    
    # Left Panel - Upcoming Matches
    ax.add_patch(Rectangle((20, 100), 450, 900, facecolor=COLORS['dark_card']))
    ax.text(245, 970, 'UPCOMING MATCHES', ha='center', va='center',
           fontsize=16, fontweight='bold', color='white')
    
    matches = [
        ('Liverpool', 'Chelsea', 'Win', 0.72),
        ('Man United', 'Tottenham', 'Draw', 0.58),
        ('Newcastle', 'Brighton', 'Win', 0.65),
        ('Aston Villa', 'West Ham', 'Loss', 0.61),
        ('Everton', 'Brentford', 'Win', 0.55)
    ]
    
    y_pos = 900
    for home, away, pred, conf in matches:
        # Match card
        ax.add_patch(FancyBboxPatch((40, y_pos), 410, 100,
                                   boxstyle="round,pad=5",
                                   facecolor=COLORS['dark_bg'], alpha=0.8))
        ax.text(245, y_pos + 75, f'{home} vs {away}', ha='center', va='center',
               fontsize=12, color='white', fontweight='bold')
        
        # Prediction
        color = COLORS['green'] if pred == 'Win' else COLORS['gray'] if pred == 'Draw' else COLORS['red']
        ax.text(245, y_pos + 50, f'Pred: {pred}', ha='center', va='center',
               fontsize=11, color=color)
        ax.text(245, y_pos + 25, f'Conf: {conf:.0%}', ha='center', va='center',
               fontsize=10, color=COLORS['light_gray'])
        
        y_pos -= 130
    
    # Center Panel - Featured Match
    ax.add_patch(Rectangle((490, 500), 940, 500, facecolor=COLORS['dark_card']))
    ax.text(960, 960, 'FEATURED PREDICTION', ha='center', va='center',
           fontsize=18, fontweight='bold', color='white')
    
    # Large featured match
    ax.text(960, 850, 'Manchester City', ha='center', va='center',
           fontsize=28, fontweight='bold', color=COLORS['accent'])
    ax.text(960, 800, 'VS', ha='center', va='center',
           fontsize=20, color=COLORS['light_gray'])
    ax.text(960, 750, 'Arsenal', ha='center', va='center',
           fontsize=28, fontweight='bold', color=COLORS['red'])
    
    # Big prediction box
    ax.add_patch(FancyBboxPatch((660, 620), 600, 100,
                               boxstyle="round,pad=10",
                               facecolor=COLORS['green'], alpha=0.3,
                               edgecolor=COLORS['green'], linewidth=3))
    ax.text(960, 680, 'HOME WIN', ha='center', va='center',
           fontsize=32, fontweight='bold', color=COLORS['green'])
    ax.text(960, 640, 'Confidence: 68.5%', ha='center', va='center',
           fontsize=16, color='white')
    
    # Probability bars
    y_pos = 560
    for outcome, prob, color in [('Home Win', 0.685, COLORS['green']),
                                 ('Draw', 0.215, COLORS['gray']),
                                 ('Away Win', 0.100, COLORS['red'])]:
        ax.text(670, y_pos, outcome, ha='left', va='center',
               fontsize=13, color='white')
        bar_width = prob * 450
        ax.add_patch(Rectangle((820, y_pos - 10), bar_width, 20,
                              facecolor=color, alpha=0.7))
        ax.text(1280, y_pos, f'{prob:.1%}', ha='right', va='center',
               fontsize=13, fontweight='bold', color='white')
        y_pos -= 40
    
    # Right Panel - Analytics
    ax.add_patch(Rectangle((1450, 100), 450, 900, facecolor=COLORS['dark_card']))
    ax.text(1675, 970, 'MODEL ANALYTICS', ha='center', va='center',
           fontsize=16, fontweight='bold', color='white')
    
    # Mini charts placeholders
    chart_titles = ['Recent Accuracy', 'Feature Impact', 'Confidence Trend']
    y_pos = 880
    for title in chart_titles:
        ax.add_patch(FancyBboxPatch((1470, y_pos), 410, 200,
                                   boxstyle="round,pad=5",
                                   facecolor=COLORS['dark_bg'], alpha=0.8))
        ax.text(1675, y_pos + 175, title, ha='center', va='center',
               fontsize=12, color='white', fontweight='bold')
        # Placeholder chart line
        ax.plot([1500, 1850], [y_pos + 80, y_pos + 120], 
               color=COLORS['accent'], linewidth=2)
        y_pos -= 230
    
    # Center Bottom Panel - Quick Stats
    ax.add_patch(Rectangle((490, 100), 940, 380, facecolor=COLORS['dark_card']))
    ax.text(960, 450, 'TOP PERFORMING FEATURES', ha='center', va='center',
           fontsize=16, fontweight='bold', color='white')
    
    top_features = list(data['features'].items())[:5]
    y_pos = 380
    for feature, importance in top_features:
        feature_name = feature.replace('_', ' ').title()
        ax.text(550, y_pos, feature_name, ha='left', va='center',
               fontsize=12, color='white')
        bar_width = importance * 700
        ax.add_patch(Rectangle((550, y_pos - 15), bar_width, 25,
                              facecolor=COLORS['accent'], alpha=0.6))
        ax.text(1350, y_pos, f'{importance:.1%}', ha='right', va='center',
               fontsize=12, fontweight='bold', color='white')
        y_pos -= 60
    
    # Footer
    ax.add_patch(Rectangle((0, 0), 1920, 80, facecolor=COLORS['dark_bg'], alpha=0.5))
    ax.text(960, 40, '© 2025 EPL Match Predictor | Decision Tree ML Model | For Educational Purposes',
           ha='center', va='center', fontsize=11, color=COLORS['light_gray'])
    
    plt.savefig('visualizations/dashboard_mockup.png', dpi=300, bbox_inches='tight',
               facecolor=COLORS['dark_bg'])
    plt.close()
    print("✓ Saved: dashboard_mockup.png")

def main():
    """Main function to generate all visualizations"""
    print("=" * 60)
    print("EPL Match Predictor - Visualization Suite")
    print("=" * 60)
    print()
    
    # Create output directory
    create_output_dir()
    print("✓ Created 'visualizations/' directory\n")
    
    # Generate realistic data
    print("Generating realistic EPL match data...")
    data = generate_realistic_data(n_matches=500)
    accuracy = np.sum(data['actual'] == data['predicted']) / len(data['actual'])
    print(f"✓ Generated {data['n_matches']} matches with {accuracy:.1%} accuracy\n")
    
    # Generate all visualizations
    print("Creating visualizations...\n")
    
    part1_performance_dashboard(data)
    part2_feature_importance(data)
    part3_probability_distributions(data)
    part4_actual_vs_predicted(data)
    part5_prediction_card(data)
    part6_dashboard_mockup(data)
    
    print()
    print("=" * 60)
    print("✅ All visualizations generated successfully!")
    print("=" * 60)
    print("\nGenerated files in 'visualizations/' folder:")
    print("  1. performance_dashboard.png")
    print("  2. feature_importance.png")
    print("  3. probability_distributions.png")
    print("  4. actual_vs_predicted.png")
    print("  5. prediction_card_sample.png")
    print("  6. dashboard_mockup.png")
    print("\nReady for GitHub README integration!")
    print()

if __name__ == "__main__":
    main()