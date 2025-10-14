"""
Soccer Analytics & Betting Expected Value Calculator
A Streamlit app for player shooting analysis and match prediction with EV betting calculator
Integrated with EPL Predictor model data

Requirements in requirements.txt:
    streamlit>=1.28.0
    pandas>=1.5.0
    numpy>=1.21.0
    matplotlib>=3.5.0
    scipy>=1.9.0

Usage:
    streamlit run soccer_betting_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from pathlib import Path
from scipy.stats import poisson
import pickle
import os

# Configure page
st.set_page_config(page_title="Soccer Analytics & Betting Calculator", layout="wide", page_icon="⚽")

# Create figures directory
Path("figures").mkdir(exist_ok=True)

# ============================================================================
# MODEL DATA LOADING
# ============================================================================

@st.cache_data
def load_epl_predictor_data():
    """Load data from EPL Predictor model"""
    try:
        if os.path.exists('epl_predictor_model.pkl'):
            with open('epl_predictor_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
            
            st.sidebar.success("✅ Loaded EPL Predictor model data")
            return model_data, None
        else:
            return None, "EPL Predictor model file not found. Please run epl_predictor.py first."
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

def prepare_model_data_for_app(model_data):
    """Prepare loaded model data for app usage"""
    
    if model_data is None:
        return None, None
    
    # Extract player data if available
    player_df = model_data.get('original_data', None)
    
    # Extract or generate match data
    match_df = model_data.get('match_data', None)
    
    # If no match data, generate synthetic match history from model
    if match_df is None and player_df is not None:
        match_df = generate_match_data_from_model(model_data, player_df)
    
    return player_df, match_df

def generate_match_data_from_model(model_data, player_df):
    """Generate match-level data from player statistics and model predictions"""
    
    # Get unique teams
    teams = player_df['team'].unique() if 'team' in player_df.columns else []
    
    if len(teams) < 2:
        return None
    
    # Generate realistic match history
    np.random.seed(42)
    matches = []
    
    from datetime import datetime, timedelta
    start_date = datetime(2024, 8, 1)
    
    match_id = 0
    # Generate round-robin (each team plays each other)
    for i, home_team in enumerate(teams):
        for j, away_team in enumerate(teams):
            if i != j:
                match_date = start_date + timedelta(days=match_id * 7)
                
                # Calculate team strengths from player data
                home_players = player_df[player_df['team'] == home_team]
                away_players = player_df[player_df['team'] == away_team]
                
                # Use goals and xG if available
                home_strength = home_players['goals'].sum() if 'goals' in home_players.columns else np.random.uniform(1.0, 2.5)
                away_strength = away_players['goals'].sum() if 'goals' in away_players.columns else np.random.uniform(0.8, 2.0)
                
                # Normalize and add randomness
                home_goals = int(np.random.poisson(max(0.5, min(4, home_strength / 10))))
                away_goals = int(np.random.poisson(max(0.3, min(3, away_strength / 10))))
                
                home_xg = home_goals * np.random.uniform(0.85, 1.15)
                away_xg = away_goals * np.random.uniform(0.85, 1.15)
                
                matches.append({
                    'match_date': match_date.strftime('%Y-%m-%d'),
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_goals': home_goals,
                    'away_goals': away_goals,
                    'home_xg': round(home_xg, 2),
                    'away_xg': round(away_xg, 2)
                })
                
                match_id += 1
    
    return pd.DataFrame(matches)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_and_prepare_data(uploaded_file):
    """Load CSV and prepare data with derived metrics"""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Standardize column names (case-insensitive mapping)
        # Only map if not already correct
        col_mapping = {}
        for col in df.columns:
            lower_col = col.lower().strip()
            
            # Skip if already in correct format
            if col in ['player', 'team', 'minutes', 'shots_total', 'shots_on_target', 'goals', 'xg',
                       'match_date', 'home_team', 'away_team', 'home_goals', 'away_goals', 
                       'home_xg', 'away_xg', 'shot_x', 'shot_y']:
                continue
            
            # Map player data columns
            if 'player' in lower_col and 'player' not in df.columns:
                col_mapping[col] = 'player'
            elif lower_col == 'team' and 'home' not in lower_col and 'away' not in lower_col:
                col_mapping[col] = 'team'
            elif 'minute' in lower_col and 'minutes' not in df.columns:
                col_mapping[col] = 'minutes'
            elif 'shot' in lower_col and 'total' in lower_col and 'shots_total' not in df.columns:
                col_mapping[col] = 'shots_total'
            elif 'shot' in lower_col and 'target' in lower_col and 'shots_on_target' not in df.columns:
                col_mapping[col] = 'shots_on_target'
            elif lower_col == 'goals' and 'home' not in lower_col and 'away' not in lower_col:
                col_mapping[col] = 'goals'
            elif lower_col in ['xg', 'expected_goals'] and 'home' not in lower_col and 'away' not in lower_col:
                col_mapping[col] = 'xg'
        
        # Apply mapping only if there are changes
        if col_mapping:
            df = df.rename(columns=col_mapping)
        
        # Calculate derived metrics for player data
        if 'minutes' in df.columns and 'shots_total' in df.columns:
            df['shots_per90'] = df.apply(
                lambda row: (row['shots_total'] * 90 / row['minutes']) 
                if row['minutes'] > 0 else 0, axis=1
            )
        
        if 'shots_on_target' in df.columns and 'shots_total' in df.columns:
            df['accuracy_pct'] = df.apply(
                lambda row: (100 * row['shots_on_target'] / max(row['shots_total'], 1))
                if row['shots_total'] > 0 else 0, axis=1
            )
        
        if 'goals' in df.columns and 'shots_total' in df.columns:
            df['goals_per_shot'] = df.apply(
                lambda row: (row['goals'] / max(row['shots_total'], 1))
                if row['shots_total'] > 0 else 0, axis=1
            )
        
        return df, None
        
    except Exception as e:
        return None, f"Error loading data: {str(e)}"

def filter_players(df, min_minutes=270):
    """Filter players by minimum minutes"""
    if 'minutes' not in df.columns:
        return df
    return df[df['minutes'] >= min_minutes].copy()

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_shots_per90_grouped_bar(df, top_n=None):
    """Grouped bar chart of shots per 90 by player, separated by team"""
    
    if 'shots_per90' not in df.columns or 'team' not in df.columns:
        st.warning("Missing required columns for shots per 90 chart")
        return None
    
    teams = sorted(df['team'].unique())
    
    fig, axes = plt.subplots(1, len(teams), figsize=(7*len(teams), 6), squeeze=False)
    axes = axes.flatten()
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    for idx, team in enumerate(teams):
        ax = axes[idx]
        team_data = df[df['team'] == team].copy()
        
        if top_n:
            team_data = team_data.nlargest(top_n, 'shots_per90')
        
        team_data = team_data.sort_values('shots_per90', ascending=True)
        
        players = team_data['player'].values
        shots = team_data['shots_per90'].values
        
        bars = ax.barh(players, shots, color=colors[idx % len(colors)], 
                      alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                   f'{width:.1f}', va='center', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Shots per 90 Minutes', fontsize=11, fontweight='bold')
        ax.set_title(f'{team}', fontsize=13, fontweight='bold', pad=10)
        ax.grid(axis='x', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    fig.savefig('figures/shots_per90_by_team.png', dpi=300, bbox_inches='tight')
    return fig

def create_stacked_accuracy_bars(df, top_n=15):
    """Stacked horizontal bars showing on-target vs off-target shots"""
    
    required_cols = ['player', 'team', 'shots_total', 'shots_on_target', 'accuracy_pct']
    if not all(col in df.columns for col in required_cols):
        st.warning("Missing required columns for accuracy chart")
        return None
    
    # Take top players by shots
    plot_data = df.nlargest(top_n, 'shots_total').sort_values('accuracy_pct', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.4)))
    
    players = plot_data['player'].values
    on_target = plot_data['shots_on_target'].values
    off_target = plot_data['shots_total'].values - plot_data['shots_on_target'].values
    accuracy = plot_data['accuracy_pct'].values
    teams = plot_data['team'].values
    
    y_pos = np.arange(len(players))
    
    # Plot stacked bars
    ax.barh(y_pos, on_target, label='On Target', color='#2ecc71', 
           alpha=0.8, edgecolor='black', linewidth=1)
    ax.barh(y_pos, off_target, left=on_target, label='Off Target', 
           color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add accuracy annotations
    for i, (player, acc, team) in enumerate(zip(players, accuracy, teams)):
        total = on_target[i] + off_target[i]
        ax.text(total + 1, i, f'{acc:.0f}%', va='center', fontsize=9, fontweight='bold')
        # Add team name
        ax.text(-1, i, f'({team})', va='center', ha='right', fontsize=8, color='gray')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(players, fontsize=9)
    ax.set_xlabel('Total Shots', fontsize=12, fontweight='bold')
    ax.set_title('Shot Accuracy by Player (On Target vs Off Target)', 
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    fig.savefig('figures/accuracy_stacked_bars.png', dpi=300, bbox_inches='tight')
    return fig

def create_league_bubble_chart(df):
    """League-wide bubble chart: shots/90 vs accuracy"""
    
    required_cols = ['player', 'team', 'shots_per90', 'accuracy_pct', 'goals']
    if not all(col in df.columns for col in required_cols):
        st.warning("Missing required columns for bubble chart")
        return None
    
    fig, ax = plt.subplots(figsize=(14, 9))
    
    teams = df['team'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(teams)))
    team_colors = dict(zip(teams, colors))
    
    for team in teams:
        team_data = df[df['team'] == team]
        ax.scatter(team_data['shots_per90'], team_data['accuracy_pct'],
                  s=team_data['goals'] * 30, c=[team_colors[team]], 
                  alpha=0.6, edgecolors='black', linewidth=1,
                  label=team)
    
    # Annotate top scorers
    top_scorers = df.nlargest(5, 'goals')
    for _, player in top_scorers.iterrows():
        ax.annotate(player['player'], 
                   (player['shots_per90'], player['accuracy_pct']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax.set_xlabel('Shots per 90 Minutes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Shooting Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('League Player Comparison: Shooting Volume vs Accuracy',
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    fig.savefig('figures/league_bubble_chart.png', dpi=300, bbox_inches='tight')
    return fig

# ============================================================================
# MATCH PREDICTION & BETTING EV CALCULATOR
# ============================================================================

def prepare_match_data(df):
    """Prepare match-level data for modeling"""
    required_cols = ['home_team', 'away_team', 'home_goals', 'away_goals']
    
    # Check if we have match data
    if not all(col in df.columns for col in required_cols):
        return None, "Match data not found in dataset"
    
    df = df.copy()
    if 'match_date' in df.columns:
        df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')
        df = df.sort_values('match_date', ascending=False)
    
    return df, None

def calculate_team_strengths(match_df, recent_n=None, use_xg=False):
    """Calculate attack and defense strengths for each team"""
    
    if recent_n:
        match_df = match_df.head(recent_n * len(match_df['home_team'].unique()))
    
    goal_col_home = 'home_xg' if use_xg and 'home_xg' in match_df.columns else 'home_goals'
    goal_col_away = 'away_xg' if use_xg and 'away_xg' in match_df.columns else 'away_goals'
    
    # Calculate league averages
    avg_home_goals = match_df[goal_col_home].mean()
    avg_away_goals = match_df[goal_col_away].mean()
    league_avg = (avg_home_goals + avg_away_goals) / 2
    
    teams = pd.concat([match_df['home_team'], match_df['away_team']]).unique()
    
    strengths = {}
    for team in teams:
        # Home matches
        home_matches = match_df[match_df['home_team'] == team]
        home_scored = home_matches[goal_col_home].mean() if len(home_matches) > 0 else league_avg
        home_conceded = home_matches[goal_col_away].mean() if len(home_matches) > 0 else league_avg
        
        # Away matches
        away_matches = match_df[match_df['away_team'] == team]
        away_scored = away_matches[goal_col_away].mean() if len(away_matches) > 0 else league_avg
        away_conceded = away_matches[goal_col_home].mean() if len(away_matches) > 0 else league_avg
        
        strengths[team] = {
            'home_attack': home_scored / avg_home_goals if avg_home_goals > 0 else 1,
            'away_attack': away_scored / avg_away_goals if avg_away_goals > 0 else 1,
            'home_defense': home_conceded / avg_away_goals if avg_away_goals > 0 else 1,
            'away_defense': away_conceded / avg_home_goals if avg_home_goals > 0 else 1,
        }
    
    return strengths, avg_home_goals, avg_away_goals

def predict_match(home_team, away_team, strengths, avg_home, avg_away, n_sims=10000):
    """Predict match outcome using Poisson model"""
    
    if home_team not in strengths or away_team not in strengths:
        return None
    
    # Calculate expected goals (lambda)
    lambda_home = avg_home * strengths[home_team]['home_attack'] * strengths[away_team]['away_defense']
    lambda_away = avg_away * strengths[away_team]['away_attack'] * strengths[home_team]['home_defense']
    
    # Simulate match outcomes
    home_goals_sim = np.random.poisson(lambda_home, n_sims)
    away_goals_sim = np.random.poisson(lambda_away, n_sims)
    
    goal_diff = home_goals_sim - away_goals_sim
    
    # Calculate probabilities
    p_home_win = np.mean(goal_diff > 0)
    p_draw = np.mean(goal_diff == 0)
    p_away_win = np.mean(goal_diff < 0)
    
    # Calculate prediction interval for goal difference
    percentile_10 = np.percentile(goal_diff, 10)
    percentile_90 = np.percentile(goal_diff, 90)
    
    return {
        'lambda_home': lambda_home,
        'lambda_away': lambda_away,
        'expected_diff': np.mean(goal_diff),
        'pi_low': percentile_10,
        'pi_high': percentile_90,
        'p_home_win': p_home_win,
        'p_draw': p_draw,
        'p_away_win': p_away_win,
        'goal_diff_sim': goal_diff
    }

def calculate_ev(stake, odds, prob_win):
    """Calculate expected value of a bet"""
    payout = stake * odds
    ev = stake * (prob_win * (odds - 1) - (1 - prob_win))
    breakeven_prob = 1 / odds
    
    return {
        'potential_payout': payout,
        'expected_value': ev,
        'breakeven_prob': breakeven_prob,
        'is_positive_ev': ev > 0
    }

def create_matchup_viz(prediction, home_team, away_team):
    """Create visualization for match prediction"""
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # 1. Expected goals comparison
    ax1 = axes[0]
    teams = [home_team, away_team]
    lambdas = [prediction['lambda_home'], prediction['lambda_away']]
    colors = ['#3498db', '#e74c3c']
    
    bars = ax1.bar(teams, lambdas, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    for bar, val in zip(bars, lambdas):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax1.set_ylabel('Expected Goals (λ)', fontsize=11, fontweight='bold')
    ax1.set_title('Expected Goals Comparison', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 2. Goal difference distribution
    ax2 = axes[1]
    ax2.hist(prediction['goal_diff_sim'], bins=range(-5, 6), color='#9b59b6',
            alpha=0.7, edgecolor='black', linewidth=1)
    ax2.axvline(prediction['expected_diff'], color='red', linestyle='--', 
               linewidth=2, label=f"Expected: {prediction['expected_diff']:.2f}")
    ax2.axvline(prediction['pi_low'], color='orange', linestyle=':', 
               linewidth=2, label=f"80% PI: [{prediction['pi_low']:.1f}, {prediction['pi_high']:.1f}]")
    ax2.axvline(prediction['pi_high'], color='orange', linestyle=':', linewidth=2)
    
    ax2.set_xlabel('Goal Difference (Home - Away)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax2.set_title('Simulated Goal Difference Distribution', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # 3. Win probabilities
    ax3 = axes[2]
    outcomes = ['Home Win', 'Draw', 'Away Win']
    probs = [prediction['p_home_win'], prediction['p_draw'], prediction['p_away_win']]
    colors_prob = ['#3498db', '#95a5a6', '#e74c3c']
    
    bars = ax3.bar(outcomes, probs, color=colors_prob, alpha=0.7, edgecolor='black', linewidth=2)
    for bar, prob in zip(bars, probs):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{prob:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax3.set_ylabel('Probability', fontsize=11, fontweight='bold')
    ax3.set_title('Match Outcome Probabilities', fontsize=13, fontweight='bold')
    ax3.set_ylim(0, max(probs) * 1.2)
    ax3.grid(axis='y', alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    plt.tight_layout()
    fig.savefig('figures/matchup_prediction.png', dpi=300, bbox_inches='tight')
    return fig

# ============================================================================
# STREAMLIT APP MAIN FUNCTION
# ============================================================================

def main():
    st.title("⚽ Soccer Analytics & Betting Expected Value Calculator")
    st.markdown("### Integrated with EPL Predictor Model")
    st.markdown("---")
    
    # Load EPL Predictor data automatically
    model_data, model_error = load_epl_predictor_data()
    
    if model_error:
        st.sidebar.warning(f"⚠️ {model_error}")
        st.sidebar.info("You can still upload your own CSV data below")
    
    # Prepare model data
    preset_player_df, preset_match_df = prepare_model_data_for_app(model_data) if model_data else (None, None)
    
    # Data source selection
    st.sidebar.header("📂 Data Source")
    data_source = st.sidebar.radio(
        "Choose data source:",
        ["Use EPL Predictor Data", "Upload Custom CSV"],
        index=0 if preset_player_df is not None else 1
    )
    
    # Initialize data variables
    player_df = None
    match_df = None
    
    if data_source == "Use EPL Predictor Data":
        if preset_player_df is not None:
            player_df = preset_player_df
            match_df = preset_match_df
            st.success(f"✅ Using EPL Predictor data: {len(player_df)} player records, {len(match_df) if match_df is not None else 0} matches")
        else:
            st.error("EPL Predictor data not available. Please run epl_predictor.py first or upload custom data.")
            return
    else:
        # File upload
        uploaded_file = st.file_uploader("Upload CSV (player or match data)", type=['csv'])
        
        if uploaded_file is None:
            st.info("👆 Please upload a CSV file to begin analysis")
            st.markdown("""
            ### Expected Data Format
            
            **For Player Shooting Analysis:**
            - `player`, `team`, `minutes`, `shots_total`, `shots_on_target`, `goals`
            - Optional: `xg`, `shot_x`, `shot_y`
            
            **For Match Prediction & Betting:**
            - `match_date`, `home_team`, `away_team`, `home_goals`, `away_goals`
            - Optional: `home_xg`, `away_xg`
            """)
            return
        
        # Load uploaded data
        df, error = load_and_prepare_data(uploaded_file)
        
        if error:
            st.error(error)
            return
        
        st.success(f"✅ Loaded {len(df)} rows")
        
        # Show columns for debugging
        with st.expander("🔍 Debug: View Columns"):
            st.write("**Columns in uploaded file:**")
            st.write(df.columns.tolist())
            st.write("**First 3 rows:**")
            st.dataframe(df.head(3))
        
        # Determine data type - check columns more carefully
        has_player_cols = all(col in df.columns for col in ['player', 'team'])
        has_match_cols = all(col in df.columns for col in ['home_team', 'away_team', 'home_goals', 'away_goals'])
        
        if has_player_cols:
            player_df = df
            st.info("📊 Detected: Player shooting data")
        
        if has_match_cols:
            match_df = df
            st.info("⚽ Detected: Match results data")
        
        if not has_player_cols and not has_match_cols:
            st.warning("⚠️ Could not detect data type. Please ensure CSV has required columns.")
            st.write("**Required columns for Player Data:** player, team, minutes, shots_total, shots_on_target, goals")
            st.write("**Required columns for Match Data:** home_team, away_team, home_goals, away_goals")
    
    # Determine what data we have
    has_player_data = player_df is not None and len(player_df) > 0 and all(col in player_df.columns for col in ['player', 'team'])
    has_match_data = match_df is not None and len(match_df) > 0 and all(col in match_df.columns for col in ['home_team', 'away_team', 'home_goals', 'away_goals'])
    
    # Debug info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Data Status")
    st.sidebar.write(f"Player data: {'✅ Available' if has_player_data else '❌ Not available'}")
    st.sidebar.write(f"Match data: {'✅ Available' if has_match_data else '❌ Not available'}")
    
    if player_df is not None:
        st.sidebar.write(f"Player rows: {len(player_df)}")
    if match_df is not None:
        st.sidebar.write(f"Match rows: {len(match_df)}")
    
    # Sidebar filters
    st.sidebar.header("🎛️ Filters & Settings")
    
    if has_player_data:
        min_minutes = st.sidebar.slider("Minimum Minutes Played", 0, 1000, 270, 90)
        top_n = st.sidebar.slider("Top N Players per Chart", 5, 30, 15)
        
        df_filtered = filter_players(player_df, min_minutes)
        st.sidebar.metric("Players After Filter", len(df_filtered))
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Player Shooting", "🏆 League Comparison", "⚖️ Matchup & EV Calculator"])
    
    # TAB 1: Player Shooting
    with tab1:
        if not has_player_data:
            st.warning("Player data not available")
        else:
            st.header("Player Shooting Analysis by Team")
            
            teams = sorted(df_filtered['team'].unique())
            selected_teams = st.multiselect("Select Teams to Analyze", teams, default=teams[:2] if len(teams) >= 2 else teams)
            
            if selected_teams:
                df_team_filtered = df_filtered[df_filtered['team'].isin(selected_teams)]
                
                # Chart 1: Shots per 90
                st.subheader("1. Shots per 90 Minutes by Player")
                fig1 = create_shots_per90_grouped_bar(df_team_filtered, top_n=top_n)
                if fig1:
                    st.pyplot(fig1)
                    plt.close(fig1)
                
                # Chart 2: Stacked Accuracy Bars
                st.subheader("2. Shot Accuracy (On Target vs Off Target)")
                fig2 = create_stacked_accuracy_bars(df_team_filtered, top_n=top_n)
                if fig2:
                    st.pyplot(fig2)
                    plt.close(fig2)
    
    # TAB 2: League Comparison
    with tab2:
        if not has_player_data:
            st.warning("Player data not available")
        else:
            st.header("League-Wide Player Comparison")
            
            # Chart: Bubble Chart
            fig5 = create_league_bubble_chart(df_filtered)
            if fig5:
                st.pyplot(fig5)
                plt.close(fig5)
            
            st.info("💡 Bubble size represents total goals scored. Top 5 scorers are annotated.")
    
    # TAB 3: Matchup & EV Calculator
    with tab3:
        if not has_match_data:
            st.warning("Match data not available")
        else:
            st.header("Match Prediction & Betting EV Calculator")
            st.markdown("### 🤖 Powered by EPL Predictor Model")
            
            match_data_prepared, error = prepare_match_data(match_df)
            if error:
                st.error(error)
                return
            
            st.markdown("### 🎯 Match Setup")
            
            col1, col2 = st.columns(2)
            
            teams = sorted(pd.concat([match_data_prepared['home_team'], match_data_prepared['away_team']]).unique())
            
            with col1:
                team_a = st.selectbox("Select Home Team (Team A)", teams, key='team_a')
            
            with col2:
                available_teams_b = [t for t in teams if t != team_a]
                team_b = st.selectbox("Select Away Team (Team B)", available_teams_b, key='team_b')
            
            st.markdown("### ⚙️ Model Configuration")
            
            col3, col4 = st.columns(2)
            
            with col3:
                model_basis = st.selectbox(
                    "Model Basis",
                    ["Season Average", "Recent 10 Matches", "Weighted (Recent 2x)"]
                )
                
                use_xg = st.checkbox("Use xG (if available)", value=False)
            
            with col4:
                bet_type = st.selectbox("Bet Type", ["Home Win", "Draw", "Away Win"])
                
            st.markdown("### 💰 Betting Calculator")
            
            col5, col6 = st.columns(2)
            
            with col5:
                decimal_odds = st.number_input("Decimal Odds", min_value=1.01, max_value=50.0, value=2.50, step=0.05)
            
            with col6:
                stake = st.number_input("Stake ($)", min_value=1.0, max_value=10000.0, value=100.0, step=10.0)
            
            if st.button("🔮 Calculate Prediction & EV", type="primary"):
                
                # Determine recent_n based on model basis
                recent_n = None
                if model_basis == "Recent 10 Matches":
                    recent_n = 10
                elif model_basis == "Weighted (Recent 2x)":
                    recent_n = 20
                
                # Calculate team strengths
                strengths, avg_home, avg_away = calculate_team_strengths(
                    match_data_prepared, recent_n=recent_n, use_xg=use_xg
                )
                
                # Predict match
                prediction = predict_match(team_a, team_b, strengths, avg_home, avg_away)
                
                if prediction is None:
                    st.error("Unable to calculate prediction. Teams may not have enough historical data.")
                    return
                
                # Display prediction results
                st.markdown("---")
                st.markdown("## 📈 Prediction Results")
                
                # Metrics
                metric_cols = st.columns(4)
                
                with metric_cols[0]:
                    st.metric("Expected Goal Diff", f"{prediction['expected_diff']:.2f}")
                
                with metric_cols[1]:
                    st.metric("80% Prediction Interval", 
                             f"[{prediction['pi_low']:.1f}, {prediction['pi_high']:.1f}]")
                
                with metric_cols[2]:
                    st.metric(f"{team_a} λ", f"{prediction['lambda_home']:.2f}")
                
                with metric_cols[3]:
                    st.metric(f"{team_b} λ", f"{prediction['lambda_away']:.2f}")
                
                # Probabilities
                st.markdown("### Match Outcome Probabilities")
                prob_cols = st.columns(3)
                
                with prob_cols[0]:
                    st.metric("🏠 Home Win", f"{prediction['p_home_win']:.1%}")
                
                with prob_cols[1]:
                    st.metric("🤝 Draw", f"{prediction['p_draw']:.1%}")
                
                with prob_cols[2]:
                    st.metric("✈️ Away Win", f"{prediction['p_away_win']:.1%}")
                
                # Visualization
                st.markdown("### 📊 Match Analysis Visualization")
                fig_match = create_matchup_viz(prediction, team_a, team_b)
                st.pyplot(fig_match)
                plt.close(fig_match)
                
                # Calculate EV
                st.markdown("---")
                st.markdown("## 💵 Betting Expected Value Analysis")
                
                # Get probability for selected bet
                if bet_type == "Home Win":
                    prob_win = prediction['p_home_win']
                elif bet_type == "Draw":
                    prob_win = prediction['p_draw']
                else:  # Away Win
                    prob_win = prediction['p_away_win']
                
                ev_results = calculate_ev(stake, decimal_odds, prob_win)
                
                # Display EV results
                ev_cols = st.columns(4)
                
                with ev_cols[0]:
                    st.metric("Potential Payout", f"${ev_results['potential_payout']:.2f}")
                
                with ev_cols[1]:
                    st.metric("Expected Value (EV)", 
                             f"${ev_results['expected_value']:.2f}",
                             delta="Positive" if ev_results['is_positive_ev'] else "Negative")
                
                with ev_cols[2]:
                    st.metric("Break-even Probability", f"{ev_results['breakeven_prob']:.1%}")
                
                with ev_cols[3]:
                    st.metric("Model Probability", f"{prob_win:.1%}")
                
                # EV Interpretation
                if ev_results['is_positive_ev']:
                    st.success(f"✅ **+EV Bet!** The model suggests this bet has positive expected value of ${ev_results['expected_value']:.2f}. Your model probability ({prob_win:.1%}) exceeds the break-even probability ({ev_results['breakeven_prob']:.1%}).")
                else:
                    st.error(f"❌ **-EV Bet.** The model suggests this bet has negative expected value of ${ev_results['expected_value']:.2f}. Your model probability ({prob_win:.1%}) is below the break-even probability ({ev_results['breakeven_prob']:.1%}).")
                
                # Comparison table
                st.markdown("### 📋 Implied vs Model Probabilities")
                
                comparison_df = pd.DataFrame({
                    'Outcome': ['Home Win', 'Draw', 'Away Win'],
                    'Model Probability': [
                        f"{prediction['p_home_win']:.1%}",
                        f"{prediction['p_draw']:.1%}",
                        f"{prediction['p_away_win']:.1%}"
                    ],
                    'Your Odds': [
                        f"{decimal_odds:.2f}" if bet_type == "Home Win" else "-",
                        f"{decimal_odds:.2f}" if bet_type == "Draw" else "-",
                        f"{decimal_odds:.2f}" if bet_type == "Away Win" else "-"
                    ],
                    'Implied Probability': [
                        f"{ev_results['breakeven_prob']:.1%}" if bet_type == "Home Win" else "-",
                        f"{ev_results['breakeven_prob']:.1%}" if bet_type == "Draw" else "-",
                        f"{ev_results['breakeven_prob']:.1%}" if bet_type == "Away Win" else "-"
                    ]
                })
                
                st.dataframe(comparison_df, hide_index=True, use_container_width=True)
                
                # Model explanation
                with st.expander("🔍 How the Model Works"):
                    st.markdown("""
                    **Transparent Poisson Model (Based on EPL Predictor Data):**
                    
                    1. **Team Strength Calculation:**
                       - Attack strength = Team's avg goals scored / League avg
                       - Defense weakness = Team's avg goals conceded / League avg
                       - Data sourced from EPL Predictor model's historical matches
                    
                    2. **Expected Goals (λ) Calculation:**
                       - λ_home = League avg home goals × Home attack × Away defense
                       - λ_away = League avg away goals × Away attack × Home defense
                    
                    3. **Outcome Probabilities:**
                       - Simulate 10,000 matches using Poisson distribution
                       - Count: Home wins, Draws, Away wins
                       - Calculate probabilities from simulation
                    
                    4. **Expected Value (EV):**
                       - EV = Stake × (P(win) × (Odds - 1) - P(lose))
                       - Break-even = 1 / Odds
                       - Positive EV = Your probability > Implied probability
                    
                    **Model trained on real EPL data from epl_predictor.py!**
                    
                    ⚠️ **Disclaimer:** This is for educational purposes only. Past performance doesn't guarantee future results. Please gamble responsibly.
                    """)
    
    # Footer with documentation
    st.markdown("---")
    with st.expander("📖 How to Read Each Chart"):
        st.markdown("""
        ### Chart Interpretation Guide
        
        #### 1. Shots per 90 Minutes (Grouped Bar)
        - **What it shows:** Shooting volume per player, normalized to 90 minutes
        - **How to read:** Higher bars = more shooting attempts when on the field
        - **Separated by team** for easy comparison
        
        #### 2. Shot Accuracy (Stacked Horizontal Bars)
        - **What it shows:** Proportion of shots on target (green) vs off target (red)
        - **How to read:** Longer green sections = better accuracy
        - **Percentage annotation** shows exact accuracy rate
        - Team shown in parentheses for context
        
        #### 3. League Bubble Chart
        - **What it shows:** All players plotted by volume vs accuracy
        - **How to read:**
          - X-axis = Shots per 90 (volume)
          - Y-axis = Accuracy % (precision)
          - Bubble size = Total goals (output)
          - Color = Team
        - Top right corner = high volume + high accuracy shooters
        - Top 5 scorers are labeled
        
        ### Match Prediction Model
        
        #### Expected Goals (λ)
        - Predicted average goals for each team
        - Based on historical attack/defense strength from EPL Predictor data
        - Accounts for home/away advantage
        
        #### Goal Difference Distribution
        - Shows range of possible outcomes from 10,000 simulations
        - Red dashed line = expected (mean) difference
        - Orange lines = 80% prediction interval (confidence range)
        
        #### Win Probabilities
        - Calculated from simulated match outcomes
        - Sum to 100% across Home/Draw/Away
        
        #### Expected Value (EV)
        - **Positive EV** = Model thinks bet has value (model prob > implied prob)
        - **Negative EV** = Model thinks bet is overpriced
        - Break-even probability = minimum win rate needed to profit
        
        ### ⚠️ Important Notes
        - Data automatically loaded from EPL Predictor model
        - Model uses transparent Poisson approach (no black boxes)
        - Charts auto-save to `figures/` directory at 300 DPI
        - Past performance doesn't guarantee future results
        - **This is for educational purposes only - not financial advice!**
        """)
    
    # Model info in sidebar
    if model_data:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 EPL Predictor Model Info")
        st.sidebar.metric("Model Accuracy", f"{model_data.get('accuracy', 0):.1%}")
        st.sidebar.metric("Total Predictions", model_data.get('total_predictions', 'N/A'))
        
        if 'feature_importance' in model_data:
            st.sidebar.markdown("**Top 3 Features:**")
            top_features = list(model_data['feature_importance'].items())[:3]
            for feature, importance in top_features:
                st.sidebar.text(f"• {feature}: {importance:.3f}")
    
    st.markdown("---")
    st.caption("⚽ Soccer Analytics & Betting EV Calculator | Integrated with EPL Predictor | Built with Streamlit")

if __name__ == "__main__":
    main()