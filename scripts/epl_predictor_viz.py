import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import requests # type: ignore
from bs4 import BeautifulSoup # type: ignore
import time
import pickle
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os
warnings.filterwarnings('ignore')

class EPLMatchPredictor:
    """
    Premier League Match Outcome Predictor using Decision Tree Algorithm
    """
    
    def __init__(self, output_dir='images'):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.team_stats = {}
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created directory: {self.output_dir}/")
        
    def scrape_fbref_data(self, season='2023-2024', cache_file='epl_data.csv'):
        """
        Scrape Premier League data from FBref
        """
        print("Scraping data from FBref...")
        
        # FBref Premier League URL structure
        base_url = "https://fbref.com/en/comps/9/Premier-League-Stats"
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            # Get main stats page
            response = requests.get(base_url, headers=headers)
            time.sleep(3)  # Rate limiting
            
            # Parse tables using pandas
            tables = pd.read_html(response.text)
            
            # Typically, shooting stats and match results are in different tables
            # This is a simplified example - adjust based on actual FBref structure
            
            print(f"Found {len(tables)} tables on the page")
            print("Note: For full implementation, you'll need to scrape multiple pages")
            print("including shooting stats, fixtures, and set piece data")
            
            return tables
            
        except Exception as e:
            print(f"Error scraping data: {e}")
            print("Loading sample data instead...")
            return self.create_sample_data()
    
    def create_sample_data(self):
        """
        Create sample training data for demonstration
        """
        np.random.seed(42)
        n_matches = 500
        
        teams = ['Man City', 'Arsenal', 'Liverpool', 'Chelsea', 'Man United', 
                 'Tottenham', 'Newcastle', 'Brighton', 'Aston Villa', 'West Ham']
        
        data = {
            'home_team': np.random.choice(teams, n_matches),
            'away_team': np.random.choice(teams, n_matches),
            'home_shots': np.random.normal(15, 4, n_matches),
            'away_shots': np.random.normal(12, 4, n_matches),
            'home_shots_on_target': np.random.normal(6, 2, n_matches),
            'away_shots_on_target': np.random.normal(5, 2, n_matches),
            'home_xg': np.random.normal(1.8, 0.7, n_matches),
            'away_xg': np.random.normal(1.3, 0.6, n_matches),
            'home_free_kick_rate': np.random.uniform(0.05, 0.15, n_matches),
            'away_free_kick_rate': np.random.uniform(0.05, 0.15, n_matches),
            'home_penalty_rate': np.random.uniform(0.70, 0.90, n_matches),
            'away_penalty_rate': np.random.uniform(0.70, 0.90, n_matches),
            'home_form_last5': np.random.uniform(0, 15, n_matches),
            'away_form_last5': np.random.uniform(0, 15, n_matches),
            'home_win_rate': np.random.uniform(0.3, 0.7, n_matches),
            'away_win_rate': np.random.uniform(0.2, 0.6, n_matches),
            'home_goals_avg': np.random.normal(1.8, 0.5, n_matches),
            'away_goals_avg': np.random.normal(1.3, 0.5, n_matches),
            'home_conceded_avg': np.random.normal(1.1, 0.4, n_matches),
            'away_conceded_avg': np.random.normal(1.4, 0.5, n_matches),
            'days_since_last_match_home': np.random.randint(3, 14, n_matches),
            'days_since_last_match_away': np.random.randint(3, 14, n_matches),
        }
        
        df = pd.DataFrame(data)
        
        # Filter out matches where team plays itself
        df = df[df['home_team'] != df['away_team']].reset_index(drop=True)
        
        # Create outcome based on features (with some randomness)
        df['outcome'] = df.apply(self._generate_outcome, axis=1)
        
        return df
    
    def _generate_outcome(self, row):
        """
        Generate realistic match outcomes based on features
        """
        home_strength = (
            row['home_xg'] * 0.3 +
            row['home_form_last5'] * 0.2 +
            row['home_win_rate'] * 0.2 +
            (row['home_goals_avg'] - row['home_conceded_avg']) * 0.3
        )
        
        away_strength = (
            row['away_xg'] * 0.3 +
            row['away_form_last5'] * 0.2 +
            row['away_win_rate'] * 0.2 +
            (row['away_goals_avg'] - row['away_conceded_avg']) * 0.3
        )
        
        # Add home advantage
        home_strength *= 1.2
        
        diff = home_strength - away_strength
        noise = np.random.normal(0, 0.5)
        diff += noise
        
        if diff > 0.3:
            return 'Win'
        elif diff < -0.3:
            return 'Loss'
        else:
            return 'Draw'
    
    def engineer_features(self, df):
        """
        Create advanced features for prediction
        """
        print("Engineering features...")
        
        # Differential features
        df['shot_differential'] = df['home_shots'] - df['away_shots']
        df['xg_differential'] = df['home_xg'] - df['away_xg']
        df['form_differential'] = df['home_form_last5'] - df['away_form_last5']
        df['win_rate_differential'] = df['home_win_rate'] - df['away_win_rate']
        df['goals_differential'] = df['home_goals_avg'] - df['away_goals_avg']
        
        # Shot efficiency
        df['home_shot_efficiency'] = df['home_shots_on_target'] / (df['home_shots'] + 1)
        df['away_shot_efficiency'] = df['away_shots_on_target'] / (df['away_shots'] + 1)
        df['shot_efficiency_diff'] = df['home_shot_efficiency'] - df['away_shot_efficiency']
        
        # Set piece advantage
        df['set_piece_advantage'] = (df['home_free_kick_rate'] + df['home_penalty_rate']) - \
                                     (df['away_free_kick_rate'] + df['away_penalty_rate'])
        
        # Rest advantage
        df['rest_advantage'] = df['days_since_last_match_home'] - df['days_since_last_match_away']
        
        # Defensive strength
        df['defensive_differential'] = df['away_conceded_avg'] - df['home_conceded_avg']
        
        return df
    
    def prepare_training_data(self, df):
        """
        Prepare data for model training
        """
        print("Preparing training data...")
        
        # Select features for training
        feature_cols = [
            'home_shots', 'away_shots', 'home_shots_on_target', 'away_shots_on_target',
            'home_xg', 'away_xg', 'home_form_last5', 'away_form_last5',
            'home_win_rate', 'away_win_rate', 'home_goals_avg', 'away_goals_avg',
            'home_conceded_avg', 'away_conceded_avg',
            'shot_differential', 'xg_differential', 'form_differential',
            'win_rate_differential', 'goals_differential', 'shot_efficiency_diff',
            'set_piece_advantage', 'rest_advantage', 'defensive_differential'
        ]
        
        self.feature_names = feature_cols
        
        X = df[feature_cols]
        y = df['outcome']
        
        # Handle any missing values
        X = X.fillna(X.mean())
        
        return X, y
    
    def train_model(self, X, y, tune_hyperparameters=True):
        """
        Train Decision Tree model with optional hyperparameter tuning
        """
        print("\nTraining Decision Tree model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        if tune_hyperparameters:
            print("Tuning hyperparameters...")
            param_grid = {
                'max_depth': [5, 10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            }
            
            dt = DecisionTreeClassifier(random_state=42)
            grid_search = GridSearchCV(
                dt, param_grid, cv=5, scoring='accuracy', n_jobs=-1
            )
            grid_search.fit(X_train_scaled, y_train)
            
            self.model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            self.model = DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                criterion='entropy',
                random_state=42
            )
            self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model and save visualizations
        self._evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test)
        
        return X_test, y_test
    
    def _evaluate_model(self, X_train, X_test, y_train, y_test):
        """
        Evaluate model performance and save visualizations
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        # Training accuracy
        train_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        print(f"\nTraining Accuracy: {train_acc:.3f}")
        
        # Testing accuracy
        test_pred = self.model.predict(X_test)
        test_acc = accuracy_score(y_test, test_pred)
        print(f"Testing Accuracy: {test_acc:.3f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, test_pred))
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, test_pred)
        print(cm)
        
        # Save confusion matrix visualization
        self._save_confusion_matrix(y_test, test_pred)
        
        # Save accuracy comparison
        self._save_accuracy_comparison(train_acc, test_acc)
        
        # Feature importance
        print("\nTop 10 Most Important Features:")
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for idx, row in feature_importance.head(10).iterrows():
            print(f"{row['feature']:30s}: {row['importance']:.4f}")
        
        # Save feature importance visualization
        self._save_feature_importance(feature_importance)
    
    def _save_confusion_matrix(self, y_test, y_pred):
        """
        Save confusion matrix heatmap
        """
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.model.classes_, 
                    yticklabels=self.model.classes_,
                    cbar_kws={'label': 'Count'})
        
        plt.title('Confusion Matrix - Match Outcome Predictions', fontsize=14, fontweight='bold')
        plt.ylabel('Actual Outcome', fontsize=12)
        plt.xlabel('Predicted Outcome', fontsize=12)
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, 'confusion_matrix.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Confusion matrix saved to: {filepath}")
    
    def _save_accuracy_comparison(self, train_acc, test_acc):
        """
        Save training vs testing accuracy comparison
        """
        plt.figure(figsize=(8, 6))
        
        metrics = ['Training Accuracy', 'Testing Accuracy']
        values = [train_acc, test_acc]
        colors = ['#2E86AB', '#A23B72']
        
        bars = plt.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1%}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.ylim(0, 1.0)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Model Performance: Training vs Testing', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, 'accuracy_comparison.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Accuracy comparison saved to: {filepath}")
    
    def _save_feature_importance(self, feature_importance):
        """
        Save feature importance chart
        """
        plt.figure(figsize=(10, 8))
        
        top_features = feature_importance.head(15)
        
        plt.barh(range(len(top_features)), top_features['importance'], 
                color='#F18F01', alpha=0.8, edgecolor='black', linewidth=1.2)
        
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance Score', fontsize=12)
        plt.title('Top 15 Most Important Features for Match Prediction', 
                 fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, 'feature_importance.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Feature importance saved to: {filepath}")
    
    def predict_match(self, home_team, away_team, match_features, save_visualization=False):
        """
        Predict outcome for a specific match
        
        Parameters:
        -----------
        home_team : str
            Home team name
        away_team : str
            Away team name
        match_features : dict
            Dictionary containing all required features
        save_visualization : bool
            Whether to save prediction visualization
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Create DataFrame from features
        feature_df = pd.DataFrame([match_features])
        feature_df = feature_df[self.feature_names]
        
        # Scale features
        features_scaled = self.scaler.transform(feature_df)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Get class labels
        classes = self.model.classes_
        
        # Create probability dictionary
        prob_dict = dict(zip(classes, probabilities))
        confidence = max(probabilities) * 100
        
        # Display prediction
        print("\n" + "="*60)
        print(f"MATCH PREDICTION: {home_team} vs {away_team}")
        print("="*60)
        print(f"\nPrediction: {prediction}")
        print(f"Confidence: {confidence:.1f}%")
        print(f"\nProbabilities:")
        for outcome, prob in prob_dict.items():
            print(f"  {outcome}: {prob*100:.1f}%")
        
        # Show key factors
        self._show_key_factors(feature_df, home_team, away_team)
        
        # Save visualization if requested
        if save_visualization:
            self._save_prediction_visualization(home_team, away_team, prob_dict, prediction)
        
        return prediction, prob_dict
    
    def _save_prediction_visualization(self, home_team, away_team, prob_dict, prediction):
        """
        Save prediction probabilities as a bar chart
        """
        plt.figure(figsize=(10, 6))
        
        outcomes = list(prob_dict.keys())
        probs = [prob_dict[outcome] * 100 for outcome in outcomes]
        colors = ['#06A77D' if outcome == prediction else '#D3D3D3' for outcome in outcomes]
        
        bars = plt.bar(outcomes, probs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add percentage labels
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{prob:.1f}%',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.ylim(0, 100)
        plt.ylabel('Probability (%)', fontsize=12)
        plt.xlabel('Match Outcome', fontsize=12)
        plt.title(f'Match Prediction: {home_team} vs {away_team}', 
                 fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        filename = f"prediction_{home_team.replace(' ', '_')}_vs_{away_team.replace(' ', '_')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Prediction visualization saved to: {filepath}")
    
    def _show_key_factors(self, features, home_team, away_team):
        """
        Display key factors influencing the prediction
        """
        print(f"\nKey Factors:")
        
        feature_values = features.iloc[0]
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_,
            'value': feature_values.values
        }).sort_values('importance', ascending=False)
        
        for i, row in importance.head(5).iterrows():
            feature = row['feature']
            value = row['value']
            
            if 'differential' in feature or 'advantage' in feature:
                if value > 0:
                    favor = home_team
                else:
                    favor = away_team
                print(f"{i+1}. {feature}: {value:.2f} (favors {favor})")
            else:
                print(f"{i+1}. {feature}: {value:.2f}")
    
    def save_model(self, filename='epl_predictor_model.pkl'):
        """Save trained model to disk"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names
            }, f)
        print(f"\nModel saved to {filename}")
    
    def load_model(self, filename='epl_predictor_model.pkl'):
        """Load trained model from disk"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']
        print(f"Model loaded from {filename}")


# Example usage
if __name__ == "__main__":
    print("EPL Match Predictor - Decision Tree Model")
    print("="*60)
    
    # Initialize predictor
    predictor = EPLMatchPredictor(output_dir='images')
    
    # Load/scrape data (using sample data for demo)
    df = predictor.create_sample_data()
    print(f"\nLoaded {len(df)} matches")
    
    # Engineer features
    df = predictor.engineer_features(df)
    
    # Prepare training data
    X, y = predictor.prepare_training_data(df)
    
    # Train model
    X_test, y_test = predictor.train_model(X, y, tune_hyperparameters=True)
    
    # Example prediction
    print("\n" + "="*60)
    print("EXAMPLE PREDICTION")
    
    example_features = {
        'home_shots': 16.5,
        'away_shots': 11.2,
        'home_shots_on_target': 6.8,
        'away_shots_on_target': 4.5,
        'home_xg': 2.1,
        'away_xg': 1.2,
        'home_form_last5': 12,
        'away_form_last5': 7,
        'home_win_rate': 0.65,
        'away_win_rate': 0.42,
        'home_goals_avg': 2.1,
        'away_goals_avg': 1.4,
        'home_conceded_avg': 0.9,
        'away_conceded_avg': 1.3,
        'shot_differential': 5.3,
        'xg_differential': 0.9,
        'form_differential': 5,
        'win_rate_differential': 0.23,
        'goals_differential': 0.7,
        'shot_efficiency_diff': 0.15,
        'set_piece_advantage': 0.08,
        'rest_advantage': 1,
        'defensive_differential': 0.4
    }
    
    prediction, probabilities = predictor.predict_match(
        "Manchester City", 
        "Arsenal",
        example_features,
        save_visualization=True  # Save the prediction chart
    )
    
    # Save model
    predictor.save_model()
    
    print("\n" + "="*60)
    print("Training complete! Use predict_match() for new predictions.")
    print(f"All visualizations saved to '{predictor.output_dir}/' folder")
    print("="*60)