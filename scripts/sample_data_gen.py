"""
Generate Sample Soccer Data for Testing
Creates both player shooting data and match data in the correct format
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# Realistic player names
FIRST_NAMES = [
    'Mohamed', 'Erling', 'Harry', 'Kevin', 'Bukayo', 'Marcus', 'Phil', 'Bruno',
    'Heung-Min', 'Jack', 'Raheem', 'James', 'Gabriel', 'Declan', 'Aleksandar',
    'Darwin', 'Luis', 'Kai', 'Eddie', 'Ivan', 'Joao', 'Miguel', 'Mateo',
    'Sergio', 'Andrew', 'Cole', 'Callum', 'Mason', 'Ollie', 'Jarrod',
    'Dominic', 'Wilfried', 'Neal', 'Eberechi', 'Anthony', 'Jadon', 'Christian',
    'Leon', 'Raphael', 'Gabriel', 'Martin', 'Emile', 'Leandro', 'Nicolas',
    'Roberto', 'Pedro', 'Matheus', 'Brennan', 'Harvey', 'Curtis'
]

LAST_NAMES = [
    'Salah', 'Haaland', 'Kane', 'De Bruyne', 'Saka', 'Rashford', 'Foden', 'Fernandes',
    'Son', 'Grealish', 'Sterling', 'Maddison', 'Jesus', 'Rice', 'Mitrovic',
    'Nunez', 'Diaz', 'Havertz', 'Nketiah', 'Toney', 'Felix', 'Almiron', 'Kovacic',
    'Aguero', 'Robertson', 'Palmer', 'Wilson', 'Mount', 'Watkins', 'Bowen',
    'Calvert-Lewin', 'Zaha', 'Maupay', 'Eze', 'Gordon', 'Sancho', 'Pulisic',
    'Bailey', 'Varane', 'Martinelli', 'Odegaard', 'Smith Rowe', 'Trossard', 'Jackson',
    'Firmino', 'Neto', 'Cunha', 'Johnson', 'Barnes', 'Jones'
]

# ============================================================================
# PLAYER SHOOTING DATA
# ============================================================================

def generate_player_data():
    """Generate realistic player shooting statistics with proper names"""
    
    teams = ['Manchester City', 'Liverpool', 'Chelsea', 'Arsenal', 'Tottenham', 
             'Man United', 'West Ham', 'Leicester', 'Brighton', 'Newcastle']
    
    players_per_team = 15
    data = []
    
    # Shuffle names to get variety
    np.random.shuffle(FIRST_NAMES)
    np.random.shuffle(LAST_NAMES)
    
    name_index = 0
    
    for team in teams:
        for i in range(players_per_team):
            # Generate realistic stats
            minutes = np.random.randint(180, 3000)
            shots_total = int(np.random.gamma(3, 8) * (minutes / 900))
            shots_on_target = int(shots_total * np.random.uniform(0.25, 0.65))
            goals = int(shots_on_target * np.random.uniform(0.1, 0.4))
            xg = goals * np.random.uniform(0.8, 1.3)
            
            # Optional shot coordinates (simplified - attacking third)
            shot_x = np.random.uniform(70, 105)  # Attacking half
            shot_y = np.random.uniform(15, 53)   # Width of pitch
            
            # Create realistic player name
            first_name = FIRST_NAMES[name_index % len(FIRST_NAMES)]
            last_name = LAST_NAMES[name_index % len(LAST_NAMES)]
            player_name = f"{first_name} {last_name}"
            
            data.append({
                'player': player_name,
                'team': team,
                'minutes': minutes,
                'shots_total': shots_total,
                'shots_on_target': shots_on_target,
                'goals': goals,
                'xg': round(xg, 2),
                'shot_x': round(shot_x, 1),
                'shot_y': round(shot_y, 1)
            })
            
            name_index += 1
    
    df = pd.DataFrame(data)
    df.to_csv('player_shooting_data.csv', index=False)
    print(f"✅ Created player_shooting_data.csv with {len(df)} players")
    print(f"   Sample players: {df['player'].head(5).tolist()}")
    return df

# ============================================================================
# MATCH DATA
# ============================================================================

def generate_match_data():
    """Generate realistic match results"""
    
    teams = ['Manchester City', 'Liverpool', 'Chelsea', 'Arsenal', 'Tottenham', 
             'Man United', 'West Ham', 'Leicester', 'Brighton', 'Newcastle']
    
    data = []
    start_date = datetime(2024, 8, 1)
    
    # Generate round-robin matches (each team plays each other twice - home and away)
    match_id = 0
    for round_num in range(2):  # Home and away
        for i, home_team in enumerate(teams):
            for j, away_team in enumerate(teams):
                if i != j:
                    match_date = start_date + timedelta(days=match_id * 7)
                    
                    # Simulate match with home advantage
                    home_strength = np.random.uniform(0.8, 2.2)
                    away_strength = np.random.uniform(0.6, 1.8)
                    
                    home_goals = int(np.random.poisson(1.5 * home_strength))
                    away_goals = int(np.random.poisson(1.2 * away_strength))
                    
                    home_xg = home_goals * np.random.uniform(0.85, 1.15)
                    away_xg = away_goals * np.random.uniform(0.85, 1.15)
                    
                    data.append({
                        'match_date': match_date.strftime('%Y-%m-%d'),
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_goals': home_goals,
                        'away_goals': away_goals,
                        'home_xg': round(home_xg, 2),
                        'away_xg': round(away_xg, 2)
                    })
                    
                    match_id += 1
    
    df = pd.DataFrame(data)
    df.to_csv('match_results_data.csv', index=False)
    print(f"✅ Created match_results_data.csv with {len(df)} matches")
    print(f"   Sample matches: {df[['home_team', 'away_team']].head(3).to_dict('records')}")
    return df

# ============================================================================
# COMBINED SAMPLE FOR QUICK TESTING
# ============================================================================

def generate_small_sample_files():
    """Generate smaller sample files for quick testing"""
    
    # Small player sample (30 players, 3 teams)
    teams_small = ['Manchester City', 'Liverpool', 'Arsenal']
    data = []
    
    # Use first 30 names
    name_pairs = list(zip(FIRST_NAMES[:30], LAST_NAMES[:30]))
    name_index = 0
    
    for team in teams_small:
        for i in range(10):
            minutes = np.random.randint(500, 2500)
            shots_total = int(np.random.gamma(3, 8) * (minutes / 900))
            shots_on_target = int(shots_total * np.random.uniform(0.3, 0.6))
            goals = int(shots_on_target * np.random.uniform(0.15, 0.35))
            xg = goals * np.random.uniform(0.85, 1.2)
            
            first_name, last_name = name_pairs[name_index]
            player_name = f"{first_name} {last_name}"
            
            data.append({
                'player': player_name,
                'team': team,
                'minutes': minutes,
                'shots_total': shots_total,
                'shots_on_target': shots_on_target,
                'goals': goals,
                'xg': round(xg, 2)
            })
            
            name_index += 1
    
    df_small = pd.DataFrame(data)
    df_small.to_csv('player_sample_small.csv', index=False)
    print(f"✅ Created player_sample_small.csv with {len(df_small)} players (quick test)")
    print(f"   Sample players: {df_small['player'].head(5).tolist()}")
    
    # Small match sample
    matches_small = []
    start_date = datetime(2024, 8, 1)
    match_id = 0
    
    for i, home_team in enumerate(teams_small):
        for j, away_team in enumerate(teams_small):
            if i != j:
                for round_num in range(3):  # 3 meetings
                    match_date = start_date + timedelta(days=match_id * 10)
                    
                    home_goals = int(np.random.poisson(1.6))
                    away_goals = int(np.random.poisson(1.3))
                    
                    matches_small.append({
                        'match_date': match_date.strftime('%Y-%m-%d'),
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_goals': home_goals,
                        'away_goals': away_goals,
                        'home_xg': round(home_goals * np.random.uniform(0.9, 1.1), 2),
                        'away_xg': round(away_goals * np.random.uniform(0.9, 1.1), 2)
                    })
                    
                    match_id += 1
    
    df_matches_small = pd.DataFrame(matches_small)
    df_matches_small.to_csv('match_sample_small.csv', index=False)
    print(f"✅ Created match_sample_small.csv with {len(df_matches_small)} matches (quick test)")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Generate all sample data files"""
    print("=" * 70)
    print("GENERATING SAMPLE SOCCER DATA FOR STREAMLIT APP")
    print("=" * 70)
    print()
    
    print("📊 Generating FULL datasets...")
    print()
    player_df = generate_player_data()
    match_df = generate_match_data()
    
    print()
    print("📊 Generating SMALL sample datasets for quick testing...")
    print()
    generate_small_sample_files()
    
    print()
    print("=" * 70)
    print("✅ ALL SAMPLE DATA FILES GENERATED!")
    print("=" * 70)
    print()
    print("📁 Files created:")
    print("   1. player_shooting_data.csv      - 150 players, 10 teams (FULL)")
    print("   2. match_results_data.csv        - 180 matches (FULL)")
    print("   3. player_sample_small.csv       - 30 players, 3 teams (QUICK TEST)")
    print("   4. match_sample_small.csv        - 18 matches (QUICK TEST)")
    print()
    print("🎯 Usage in Streamlit app:")
    print("   • Run: streamlit run soccer_betting_app.py")
    print("   • Upload any of these CSV files")
    print("   • Or use 'Use EPL Predictor Data' if model is trained")
    print()
    print("💡 Recommendation:")
    print("   • Use SMALL files for quick testing and development")
    print("   • Use FULL files for complete demonstrations")
    print()
    
    # Display sample data preview
    print("=" * 70)
    print("SAMPLE DATA PREVIEW")
    print("=" * 70)
    print()
    print("Player Data (first 10 rows):")
    print(player_df.head(10).to_string())
    print()
    print("Match Data (first 5 rows):")
    print(match_df.head().to_string())
    print()

if __name__ == "__main__":
    main()

# ============================================================================
# MATCH DATA
# ============================================================================

def generate_match_data():
    """Generate realistic match results"""
    
    teams = ['Manchester City', 'Liverpool', 'Chelsea', 'Arsenal', 'Tottenham', 
             'Man United', 'West Ham', 'Leicester', 'Brighton', 'Newcastle']
    
    data = []
    start_date = datetime(2024, 8, 1)
    
    # Generate round-robin matches (each team plays each other twice - home and away)
    match_id = 0
    for round_num in range(2):  # Home and away
        for i, home_team in enumerate(teams):
            for j, away_team in enumerate(teams):
                if i != j:
                    match_date = start_date + timedelta(days=match_id * 7)
                    
                    # Simulate match with home advantage
                    home_strength = np.random.uniform(0.8, 2.2)
                    away_strength = np.random.uniform(0.6, 1.8)
                    
                    home_goals = int(np.random.poisson(1.5 * home_strength))
                    away_goals = int(np.random.poisson(1.2 * away_strength))
                    
                    home_xg = home_goals * np.random.uniform(0.85, 1.15)
                    away_xg = away_goals * np.random.uniform(0.85, 1.15)
                    
                    data.append({
                        'match_date': match_date.strftime('%Y-%m-%d'),
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_goals': home_goals,
                        'away_goals': away_goals,
                        'home_xg': round(home_xg, 2),
                        'away_xg': round(away_xg, 2)
                    })
                    
                    match_id += 1
    
    df = pd.DataFrame(data)
    df.to_csv('match_results_data.csv', index=False)
    print(f"✅ Created match_results_data.csv with {len(df)} matches")
    return df

# ============================================================================
# COMBINED SAMPLE FOR QUICK TESTING
# ============================================================================

def generate_small_sample_files():
    """Generate smaller sample files for quick testing"""
    
    # Small player sample (30 players, 3 teams)
    teams_small = ['Manchester City', 'Liverpool', 'Arsenal']
    data = []
    
    for team in teams_small:
        for i in range(10):
            minutes = np.random.randint(500, 2500)
            shots_total = int(np.random.gamma(3, 8) * (minutes / 900))
            shots_on_target = int(shots_total * np.random.uniform(0.3, 0.6))
            goals = int(shots_on_target * np.random.uniform(0.15, 0.35))
            xg = goals * np.random.uniform(0.85, 1.2)
            
            data.append({
                'player': f"{team.replace(' ', '_')}_Player_{i+1}",
                'team': team,
                'minutes': minutes,
                'shots_total': shots_total,
                'shots_on_target': shots_on_target,
                'goals': goals,
                'xg': round(xg, 2)
            })
    
    df_small = pd.DataFrame(data)
    df_small.to_csv('player_sample_small.csv', index=False)
    print(f"✅ Created player_sample_small.csv with {len(df_small)} players (quick test)")
    
    # Small match sample
    matches_small = []
    start_date = datetime(2024, 8, 1)
    match_id = 0
    
    for i, home_team in enumerate(teams_small):
        for j, away_team in enumerate(teams_small):
            if i != j:
                for round_num in range(3):  # 3 meetings
                    match_date = start_date + timedelta(days=match_id * 10)
                    
                    home_goals = int(np.random.poisson(1.6))
                    away_goals = int(np.random.poisson(1.3))
                    
                    matches_small.append({
                        'match_date': match_date.strftime('%Y-%m-%d'),
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_goals': home_goals,
                        'away_goals': away_goals,
                        'home_xg': round(home_goals * np.random.uniform(0.9, 1.1), 2),
                        'away_xg': round(away_goals * np.random.uniform(0.9, 1.1), 2)
                    })
                    
                    match_id += 1
    
    df_matches_small = pd.DataFrame(matches_small)
    df_matches_small.to_csv('match_sample_small.csv', index=False)
    print(f"✅ Created match_sample_small.csv with {len(df_matches_small)} matches (quick test)")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Generate all sample data files"""
    print("=" * 70)
    print("GENERATING SAMPLE SOCCER DATA FOR STREAMLIT APP")
    print("=" * 70)
    print()
    
    print("📊 Generating FULL datasets...")
    print()
    player_df = generate_player_data()
    match_df = generate_match_data()
    
    print()
    print("📊 Generating SMALL sample datasets for quick testing...")
    print()
    generate_small_sample_files()
    
    print()
    print("=" * 70)
    print("✅ ALL SAMPLE DATA FILES GENERATED!")
    print("=" * 70)
    print()
    print("📁 Files created:")
    print("   1. player_shooting_data.csv      - 150 players, 10 teams (FULL)")
    print("   2. match_results_data.csv        - 180 matches (FULL)")
    print("   3. player_sample_small.csv       - 30 players, 3 teams (QUICK TEST)")
    print("   4. match_sample_small.csv        - 18 matches (QUICK TEST)")
    print()
    print("🎯 Usage in Streamlit app:")
    print("   • Run: streamlit run soccer_betting_app.py")
    print("   • Upload any of these CSV files")
    print("   • Or use 'Use EPL Predictor Data' if model is trained")
    print()
    print("💡 Recommendation:")
    print("   • Use SMALL files for quick testing and development")
    print("   • Use FULL files for complete demonstrations")
    print()
    
    # Display sample data preview
    print("=" * 70)
    print("SAMPLE DATA PREVIEW")
    print("=" * 70)
    print()
    print("Player Data (first 5 rows):")
    print(player_df.head())
    print()
    print("Match Data (first 5 rows):")
    print(match_df.head())
    print()

if __name__ == "__main__":
    main()