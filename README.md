# ⚽️ EPL Match Predictor & Betting Intelligence Platform ⚽️ 

## Overview

This project combines machine learning, interactive analytics, and betting mathematics to transform EPL data into actionable insights. Whether you're a data enthusiast exploring football statistics or looking to make informed betting decisions, this platform delivers professional-grade analysis in an accessible format.

**What it does:** Predicts match results, tracks team metrics across the season, and identifies positive EV betting opportunities by comparing predictions against bookmaker odds.



## Data Source

Match data is scraped using from **FBref.com**, a comprehensive football statistics database that provides official Premier League data including:

- Match results and fixtures
- Team shooting statistics (shots, shots on target, expected goals)
- Set piece conversion rates (free kicks, penalties)
- Team form metrics (recent performance, win rates)
- Defensive statistics (goals conceded, clean sheets)
- Squad rotation and rest periods between matches

The scraper uses rate-limiting to respect FBref's servers and caches data locally for efficient processing. The system generates synthetic training data that mirrors real EPL statistical distributions. Python was used for the backend, sckit-learn for machine learning, BeautifulSoup  & requests for web scraping, pandas for data preprocessing, Streamlit for the dashboard, & kelly criterion for bet sizing based on bankroll management.


## Prediction Engine

The match predictor uses advanced features to forecast outcomes:

- **Team Form Metrics** – Recent win/loss streaks, goals scored/conceded over last 5-10 matches
- **Head-to-Head History** – Historical performance between specific matchups
- **Home/Away Splits** – Location-based performance differentials
- **Expected Goals (xG)** – Shot quality and scoring efficiency indicators
- **Defensive Strength** – Goals against, clean sheets, and defensive ratings
- **Squad Value & Injuries** – Team strength adjusted for player availability
- **Betting Market Movement** – Odds shifts indicating informed money

The model is trained on 2024-2025 season of historical data using Decision Tree Classifier - GridSearchCV for hyperparameter for optimzation to achieve 90.8% accuracy on training data & 73.6% on the test data.

![Top 15 Most Important Features for Match Prediction](https://github.com/JobinJohn24/EPL-Predictor/blob/main/images/feature_importance.png)
*Figure 1.1 - Represets the most important features for match prediction.*

![Confusion Matrix](https://github.com/JobinJohn24/EPL-Predictor/blob/main/images/confusion_matrix.png)
*Figure 1.2 - Represents the confusion matrix, indicators the performance of the classification model. Broken down to true positives, true negatives, false postives, and false negatives.*


## Interactive Dashboard

Explore EPL data through dynamic visualizations:

- **League Table & Standings** – Live rankings with goal difference and form indicators
- **Team Performance Radar** – Multi-dimensional analysis (attack, defense, possession, discipline)
- **Form Timeline** – Points earned over rolling windows to identify momentum
- **Top Scorers & Assists** – Player leaderboards with trend analysis
- **Head-to-Head Comparisons** – Side-by-side team statistics for any matchup

![interactive dashboard](https://github.com/JobinJohn24/EPL-Predictor/blob/main/images/premier_league_dashboard.png)
*Figure 2.1 - Represents a dashboard of the league table, standings, performance radars, top leaders in statistical categories, & head-to-head comparisons.*

---

## EV Betting Calculator

The Expected Value calculator identifies profitable betting opportunities by comparing model predictions against bookmaker odds.

**How it works:**
1. Upload a CSV with matches, predictions (win/draw/loss probabilities), and bookmaker odds
2. The calculator computes EV for each betting market using the Kelly Criterion
3. Positive EV bets are highlighted with recommended stake sizes based on bankroll

Disclaimer: There is a dropdown legend to help navigate and understand the terms. 

**Key Features:**
- Multi-market analysis (1X2, Over/Under, Both Teams to Score)
- Bankroll management with Kelly Criterion stake suggestions
- ROI tracking and bet history analysis
- Risk-adjusted filtering to exclude high-variance opportunities

### Video Demonstration

Watch the EV calculator in action using sample data:

https://github.com/user-attachments/assets/f004c24f-0d9a-4735-a68e-6771d2f7ea04

## Installation

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/epl-predictor.git
   cd epl-predictor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the dashboard**
   ```bash
   streamlit run dashboard.py
   ```
   The dashboard will open automatically in your browser at `localhost:8501`

4. **Use the EV calculator**
   ```bash
   python ev_calculator.py --input data/sample_matches.csv
   ```
   Or access it through the dashboard's "Betting Calculator" tab

### Sample Data

A dummy CSV file is included in `/data/sample_matches.csv` for testing the EV calculator without real betting data.

## Workflow

![workflow diagram](https://github.com/JobinJohn24/EPL-Predictor/blob/main/images/workflow.png)
*Workflow Diagram represents the steps in creating or recreating this repository*

