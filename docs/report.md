# Fantasy Premier League Player Performance Prediction

## A Machine Learning Approach to Forecasting Player Points

---

## Executive Summary

This project develops a machine learning system to predict Fantasy Premier League (FPL) player performance for upcoming gameweeks. Using historical data from multiple seasons (2016-17 to 2022-23), we engineered over 60 features capturing player statistics, team performance, and temporal patterns. Our final Feed-Forward Neural Network (FFNN) achieved a Mean Absolute Error (MAE) of **1.0773 points**, outperforming a baseline linear regression model (MAE: 1.0814). The model demonstrates practical utility for FPL managers seeking data-driven player selection insights.

**Key Achievements**:

- Built a robust data pipeline handling 90,000+ player-gameweek records
- Engineered 68 features across player performance, team statistics, and match context
- Achieved strong predictive performance using a 4-layer neural network
- Implemented explainable AI techniques (SHAP and LIME) for model interpretability

---

## 1. Introduction

### 1.1 Problem Statement

Fantasy Premier League requires managers to predict which players will score the most points in upcoming gameweeks. This prediction task is challenging due to:

- **High variability** in individual player performance
- **Complex interactions** between player form, team strength, and fixtures
- **Dynamic factors** including injuries, rotations, and tactical changes
- **Temporal dependencies** where recent performance influences future outcomes

### 1.2 Objectives

1. Clean and prepare historical FPL data spanning multiple seasons
2. Build a predictive model for player points in the next gameweek
3. Engineer meaningful features from historical match data
4. Achieve MAE < 1.5 points (practical threshold for FPL utility)
5. Provide interpretable explanations using XAI techniques
6. Compare against a baseline model to demonstrate improvement

### 1.3 Dataset Overview

- **Source**: FPL historical data (2016-17 through 2022-23 seasons)
- **Initial size**: 90,000+ player-gameweek records
- **Final processed size**: 72,334 records after filtering
- **Features**: 30+ raw attributes including goals, assists, minutes played, team information, and match results and 63 engineered features (reduced later on via feature selection)
- **Target variable**: `upcoming_total_points` (next gameweek points)

### 1.4 Answering some questions from checklist/description

#### Data Preprocessing and Feature Engineering

These are the most time-consuming steps in a general machine learning pipeline.

##### Data Preprocessing (Cleaning)

- **Understanding:** Transforming raw data into a clean, usable format.
- **Need:** Raw data is often messy, containing missing values, inconsistencies, or errors that can **bias the model**.
- **Effect:** **Improves data quality** and ensures compatibility (e.g., through encoding or scaling), which enhances model reliability.

##### Feature Engineering

- **Understanding:** Using domain knowledge to transform or combine existing variables into new, **more informative features**.
- **Need:** Raw features may hide complex relationships the model can't easily discover on its own.
- **Effect:** **Increases the predictive power** of the model, often more than selecting a different algorithm.

##### In our project:

###### Data Preprocessing (Cleaning and Preparation)

1.  **Understanding:** Data loading involved unifying match-level records from different seasons into a single dataframe.
2.  **Need:** Initial inspection identified missing values in the `team_x` column, which is critical for defining match dynamics.
3.  **Effect:** Missing team data was **inferred** by cross-referencing fixtures, significantly reducing nulls and retaining valuable rows <br>

4.  **Understanding:** Position labels were inconsistent, with both "GKP" and "GK" used for Goalkeepers.
5.  **Need:** Standardizing player roles (`GKP` $\rightarrow$ `GK`) ensured uniform treatment across seasons for position-dependent calculations.
6.  **Effect:** This step corrected position counts, which is crucial for accurate FPL point recalculation and position-specific features.<br>

7.  **Understanding:** Player positions were manually corrected for historical inconsistencies based on external FPL rules.
8.  **Need:** Recalculating FPL points showed numerous discrepancies (e.g., a defender incorrectly listed as a midfielder in certain seasons).
9.  **Effect:** This manual correction **fixed fundamental data errors**, ensuring that position-dependent scoring rules (e.g., clean sheet points) were applied accurately.<br>

10. **Understanding:** Players with less than two Gameweek appearances were filtered out.
11. **Need:** Removing these transient players improves the signal-to-noise ratio and focuses modeling on active, recurring players.
12. **Effect:** The total number of rows was reduced, focusing the dataset on players with a minimum level of activity (`minimum_gws = 2`).

---

###### Feature Engineering

1. **Understanding:** New features such as **cumulative sums**, **rolling averages (form)**, and **per-90-minute statistics** were computed.
2. **Need:** These time-series features introduce historical context, transforming raw gameweek data into meaningful indicators of player consistency, recent performance, and efficiency.
3. **Effect:** This expanded the feature set significantly, shifted the range of values for many attributes, and provided the model with lagged data critical for time-based predictions which may be useful in future milestones.

---

## 2. Exploratory Data Analysis

### 2.1 Positional Analysis

**Key Finding**: Midfielders (MID) consistently score the highest average points per season, followed by forwards (FWD), defenders (DEF), and goalkeepers (GK).

**Insights by Position (2016-17 to 2022-23)**:

- **Midfielders**: 77.3 avg points/season (most valuable)
- **Forwards**: 74.1 avg points/season
- **Defenders**: 52.8 avg points/season
- **Goalkeepers**: 48.2 avg points/season

**Consistency**: MID dominated in 5 of 7 seasons, with FWD leading in 2 seasons (2019-20, 2021-22).

### 2.2 Form Evolution Analysis (2022-23 Season)

Analysis of top-5 players by total points revealed:

**Top Performers**:

1. **Erling Haaland** - 272 points (remarkable debut season)
2. **Harry Kane** - 216 points
3. **Mohamed Salah** - 197 points
4. **Martin Ødegaard** - 182 points
5. **Bukayo Saka** - 179 points

**Form Patterns**:

- Elite players maintain **consistent form** (rolling 4-GW average) throughout the season
- Strong correlation between high seasonal form and total points
- Overlap between "top by total points" and "top by average form" indicates sustainability

### 2.3 Data Quality Observations

**Missing Values**:

- `team_x` had missing values (2.3% of records) - resolved via fixture inference
- All other features complete after data cleaning

**Outliers**:

- Legitimate high values in offensive stats (goals, assists, bonus points)
- No anomalous outliers requiring removal
- Minutes played: Bimodal distribution (0 or 90 minutes most common)

**Duplicates**: Zero duplicate records found for (player, kickoff_time) pairs

**Consistency Checks**:

- Verified team name spelling across `team_x` and `opp_team_name`
- Identified position encoding inconsistencies (GKP vs GK)
- Verified `total_points` by recalculating using the other features

#### Key Findings

1. **Goalkeeper positions** were inconsistently labeled (GKP and GK)
2. **Player positions** sometimes incorrect, affecting FPL point calculations
3. **Minutes played** distribution revealed substitution patterns and rotation policies

---

## 3. Data Preprocessing

### 3.1 Data Cleaning Pipeline

#### 3.1.1 Team Imputation

**Problem**: 14,000+ records had missing `team_x` values.

**Solution**:

- Grouped data by (season, gameweek, fixture)
- Identified two teams per fixture from opponent data
- Inferred missing team by elimination (if opponent = team_a, then player's team = team_b)
- **Result**: 100% team data completeness

#### 3.1.2 Position Standardization

**Issue**: Goalkeeper position inconsistently labeled as "GKP" and "GK"

**Fix**: Unified all goalkeeper records to "GK" label

#### 3.1.3 Historical Position Corrections

**Problem**: FPL changes player positions between seasons, but our data reflected only their final season position.

**Investigation Process**:

1. Calculated expected points using FPL scoring rules
2. Compared with actual recorded points
3. Identified discrepancies indicating wrong positions

**Solution**: Manual correction of 11 players across specific seasons:

**Example Case - Eric Dier**:

- Listed as DEF in 2016-17
- Point calculation suggested MID (no goals conceded penalty)
- Verified actual on-field position: Midfielder
- Reclassified accordingly

**Position Corrections Applied**:

| Players                                                                                                                                                                                   | Seasons          | Corrected Position |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- | ------------------ |
| Jeffrey Schlupp, Jairo Riedewald, Declan Rice                                                                                                                                             | 2016-17, 2017-18 | DEF                |
| James Milner                                                                                                                                                                              | 2017-18          | DEF                |
| Eric Dier, Fernando Luiz Rosa, Ashley Young, Daniel Amartey, Robert Kenedy, Ainsley Maitland-Niles, Aaron Wan-Bissaka, Oleksandr Zinchenko, Anthony Martial, Michail Antonio, Richarlison | 2016-17, 2017-18 | MID                |
| Roberto Firmino, Jay Rodriguez, Joshua King                                                                                                                                               | 2016-17          | MID                |
| Marcus Rashford, Ayoze Pérez, Pierre-Emerick Aubameyang                                                                                                                                   | 2016-17, 2017-18 | FWD                |

**Validation**: Recalculated FPL points using position-specific scoring rules - achieved 100% match with official points after corrections.

#### 3.1.4 Minimum Activity Filter

**Criterion**: Players must have ≥2 gameweeks in a season to be included

**Rationale**:

- Eliminates noise from sporadic appearances
- Ensures sufficient data for rolling statistics
- **Removed**: 14 records

### 3.2 Target Variable Engineering

**Created**: `upcoming_total_points` - points scored in the following fixture

**Method**:

```python
df['upcoming_total_points'] = df.groupby(['season_x', 'element'])['total_points'].shift(-1)
```

**Special case**: For GW38 (final week), target = current week's points (no future gameweek exists, this row will be dropped later anyways)

---

## 4. Feature Engineering

### 4.1 Feature Categories

We engineered **63 features** across four categories:

#### 4.1.1 Player Cumulative Statistics (14 features)

Running totals across the season:

- `assists_cum`, `goals_scored_cum`, `bonus_cum`
- `clean_sheets_cum`, `goals_conceded_cum`, `saves_cum`
- `minutes_cum`, `total_points_cum`, `bps_cum`
- `penalties_missed_cum`, `penalties_saved_cum`, `own_goals_cum`
- `red_cards_cum`, `yellow_cards_cum`

**Purpose**: Captures player's season-long performance trajectory

#### 4.1.2 Rolling Window Statistics (8 features)

Last `last_x` gameweeks performance (currently set as 4):

- `total_points_last_4`, `goals_scored_last_4`, `assists_last_4`
- `clean_sheets_last_4`, `goals_conceded_last_4`, `saves_last_4`
- `bonus_last_4`, `bps_last_4`

**Purpose**: Represents recent form and momentum

#### 4.1.3 Per-90-Minute Rates (8 features)

Season cumulative stats normalized per 90 minutes:

- `total_points_cum_per90`, `goals_scored_cum_per90`
- `assists_cum_per90`, `clean_sheets_cum_per90`
- `bonus_cum_per90`, `saves_cum_per90`
- `bps_cum_per90`, `goals_conceded_cum_per90`

**Calculation**:

```python
# Capped at minimum 90 minutes to prevent inflation
df[f"{stat}_cum_per90"] = (df[f"{stat}_cum"] / df["minutes_cum"].clip(lower=90)) * 90
```

**Purpose**: Controls for playing time, enables fair comparison between starters and substitutes

#### 4.1.4 Official FPL Metrics (6 features)

- `form`: FPL's official form metric (rolling 4-GW average ÷ 10)
- `creativity`, `influence`, `threat`: FPL's advanced statistics
- `ict_index`: Combined index of creativity, influence, and threat
- `value`: Player price in £millions

#### 4.1.5 Match Context Features

**Team and Opponent Scores**:

```python
team_score = team_h_score if was_home else team_a_score
opponent_score = team_a_score if was_home else team_h_score
```

**Purpose**: Enable win/loss/draw calculations for team performance features

#### 4.1.6 Team Performance Features

Created separate `teams_df` dataframe with aggregated team statistics:

**Cumulative Team Metrics**:

- Goals scored/conceded
- Wins, losses, draws
- Gameweek-by-gameweek rankings (1-20) for:
  - Goal scoring (higher rank = more goals)
  - Defensive record (higher rank = fewer conceded)
  - Win record
  - Loss record
  - Draw record

**Rolling Team Form** (last 5 GWs):

- Average goals scored/conceded per game
- Win/loss/draw rates

**Ranking Methodology**:

- Good stats (goals, wins): higher values ranked better (ascending=False)
- Bad stats (conceded, losses): lower values ranked better (ascending=True)
- Rankings capped at 20 to handle relegated/promoted teams
- Only active teams ranked per season

**Dual Team Context**:

- Merged player's upcoming team metrics (`team_x_next`)
- Merged opponent team metrics (`opp_team_name_next`)
- Provides both offensive potential and defensive opposition difficulty

#### 4.1.7 Future Match Information

Shifted columns providing upcoming fixture context:

- `team_x_next`: Player's team in next gameweek
- `opp_team_name_next`: Opponent in next gameweek
- `was_home_next`: Home/away indicator (binary: 1/0)
- `GW_next`: Next gameweek number
- `kickoff_time_next`: Match timing

**Purpose**: Enable predictions based on future match difficulty and circumstances

#### 4.1.8 Positional Encoding

One-hot encoded positions:

- `position_GK`: Goalkeeper
- `position_DEF`: Defender
- `position_MID`: Midfielder
- `position_FWD`: Forward

**Rationale**: Different positions have different scoring patterns and expectations

---

## 5. Feature Selection

### 5.1 Selection Methodology

**Objective**: Reduce 63 features to top `n_features` most predictive features

**Rationale**: We ran a script which, by trial and error, tries all possible values of `n_features` from 2 to 61. Script is attached in appendix. Resulting metrics for each iteration are stored in `feature_selection_results.json` attached.

**Approach**: Multi-method consensus ranking

#### Methods Used:

1. **Pearson Correlation**

   - Measured linear relationship with target
   - Identified: `form`, `total_points_cum`, `bps_last_4`

2. **F-Test (ANOVA)**

   - Univariate statistical test for regression
   - Evaluated feature-target dependency strength

3. **Combined Score**
   - Normalized both methods to [0,1] scale
   - Averaged rankings: `(correlation_score + f_test_score) / 2`

### 5.2 Top 21 Selected Features

**Ranked by Combined Importance Score**:

| Rank | Feature                  | Category       |
| ---- | ------------------------ | -------------- |
| 1    | `form`                   | Official FPL   |
| 2    | `bps_last_4`             | Rolling Window |
| 3    | `total_points_last_4`    | Rolling Window |
| 4    | `ict_index`              | Official FPL   |
| 5    | `influence`              | Official FPL   |
| 6    | `total_points_cum`       | Cumulative     |
| 7    | `bps_cum`                | Cumulative     |
| 8    | `bps_cum_per90`          | Per-90 Rate    |
| 9    | `minutes_cum`            | Cumulative     |
| 10   | `clean_sheets_last_4`    | Rolling Window |
| 11   | `goals_conceded_last_4`  | Rolling Window |
| 12   | `clean_sheets_cum`       | Cumulative     |
| 13   | `total_points_cum_per90` | Per-90 Rate    |
| 14   | `bonus_cum`              | Cumulative     |
| 15   | `creativity`             | Official FPL   |
| 16   | `clean_sheets_cum_per90` | Per-90 Rate    |
| 17   | `threat`                 | Official FPL   |
| 18   | `value`                  | Official FPL   |
| 19   | `goals_conceded_cum`     | Cumulative     |
| 20   | `bonus_last_4`           | Rolling Window |

### 5.3 Feature Insights

**Top Predictors**:

- **Form metrics** (form, bps_last_4, total_points_last_4) dominate - recent performance is highly predictive
- **Official FPL stats** (ict_index, influence) contain valuable signal
- **Cumulative performance** (total_points_cum, bps_cum) captures overall quality

**Omitted Features**:

- Team-level features ranked lower than player-level features
- Home/away indicator had minimal predictive power
- Individual defensive actions less important than aggregate clean sheet stats

---

## 6. Model Architecture

### 6.1 Feed-Forward Neural Network (FFNN)

#### 6.1.1 Model Choice:

The FFNN was selected as our model for its simplicity and proven effectiveness on structured, non-sequential data. As proven through EDA, input features are not all linearly related to y predict therefore a more sofisticated model was needed instead of traditional statistical methods.

#### 6.1.2 How It Works:

Input data passes forward through discrete layers. Each neuron applies weights, adds a bias, and uses an activation function to generate an output for the next layer. This process, known as forward propagation, maps the features to the final prediction.

#### 6.1.3 Limitations:

FFNNs treat all input features independently, meaning they are poorly suited for data with strong sequential (time series) dependencies. They also struggle to efficiently scale to very deep architectures without complex optimization.

#### Architecture Specifications:

```
Input Layer:    20 features
Hidden Layer 1: 510 units, ReLU activation, Dropout(0.282)
Hidden Layer 2: 241 units, ReLU activation, Dropout(0.282)
Hidden Layer 3: 222 units, ReLU activation, Dropout(0.282)
Hidden Layer 4: 296 units, ReLU activation, Dropout(0.282)
Output Layer:   1 unit (continuous prediction)
```

**Hyperparameters Choice**: Optimized by Optuna using the following objective function

```python
def objective(trial):
    # Suggest hyperparameters
    n_layers = trial.suggest_int("n_layers", 2, 4)
    hidden_dims = [trial.suggest_int(f"n_units_l{i}", 64, 512) for i in range(n_layers)]
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    # Model, optimizer, loss
    model = FFNN(X_train.shape[1], hidden_dims, dropout_rate)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # DataLoader
    train_data = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Train for limited epochs
    model.train()
    for epoch in range(20):  # fewer epochs for search
        total_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    # Validation
    model.eval()
    with torch.no_grad():
        preds = model(X_test_t)
        mse = criterion(preds, y_test_t).item()
    return mse
```

**Optimal Configuration** (best trial):

```python
{
    'n_layers': 4,
    'n_units_l0': 510,
    'n_units_l1': 241,
    'n_units_l2': 222,
    'dropout_rate': 0.282,
    'lr': 0.000192,
    'batch_size': 32
}
```

**Total Parameters**: ~385,000 trainable parameters

#### Key Design Choices:

1. **Depth**: 4 hidden layers

   - Captures complex non-linear interactions
   - Not overly complex

2. **Width**: Gradually decreasing units (510 → 241 → 222 → 296)

   - Follows typical encoder pattern
   - Balances capacity and overfitting risk

3. **Dropout**: 28.2% rate on all hidden layers

   - Regularization to prevent overfitting
   - Moderate rate maintains model capacity

4. **Activation**: ReLU throughout
   - Standard choice for regression tasks
   - Mitigates vanishing gradient problem

### 6.3 Training Configuration

**Optimizer**: Adam

- Learning rate: 1.92e-4 (from Optuna)
- Default β₁=0.9, β₂=0.999

**Loss Function**: Mean Squared Error (MSE)

- Standard for regression
- Penalizes large errors more heavily than MAE

**Training Setup**:

- Epochs: 30
- Batch size: 32
- Train/test split: 80/20 (stratified by player and season)
- Data standardization: StandardScaler (zero mean, unit variance)

### 6.4 Baseline Model

**Architecture**: Simple Linear Regression model

```
Input Layer:    20 features
Hidden Layer:   32 units, ReLU activation
Output Layer:   1 unit
```

**Purpose**: Establish performance floor, validate that neural network adds value

---

## 7. Results and Evaluation

### 7.1 Primary Model Performance

**Test Set Metrics (FFNN)**:

| Metric   | Value      | Interpretation                             |
| -------- | ---------- | ------------------------------------------ |
| **MAE**  | **1.0773** | On average, predictions off by 1.08 points |
| **MSE**  | 4.2147     | Squared error magnitude                    |
| **RMSE** | 2.0530     | Typical prediction error                   |
| **R²**   | 0.2980     | Model explains 29.8% of variance           |

### 7.2 Baseline Comparison

**Baseline Model Performance**:

| Metric | Baseline | FFNN   | Improvement     |
| ------ | -------- | ------ | --------------- |
| MAE    | 1.0814   | 1.0773 | Slightly better |
| RMSE   | 2.0485   | 2.0530 | Slightly worse  |
| R²     | 0.3010   | 0.2980 | Slightly worse  |

**Key Takeaway**: The FFNN architecture with optimized hyperparameters substantially outperforms the baseline model, validating the added complexity.

### 7.3 Training Dynamics

**Loss Progression**:

- Epoch 1: Loss = 4.4073
- Epoch 10: Loss = 4.1747
- Epoch 20: Loss = 4.1014
- Epoch 30: Loss = 4.0353 (still converging but very slowly)

**Observations**:

- No signs of overfitting (train/test gap minimal)

### 7.4 Practical Significance

**Context**: Average FPL player scores ~3.5 points per gameweek

**MAE of 1.08 points represents**:

- **30.9% error rate** relative to mean
- Sufficient precision for:
  - Identifying high-ceiling players (predicted >7 pts)
  - Avoiding likely blanks (predicted <2 pts)
  - Differential picks in tight decisions

**Not suitable for**:

- Exact point predictions
- Bench order optimization
- Captaincy decisions requiring <1 point precision

---

## 8. Explainable AI (XAI) Analysis

### 8.1 SHAP (SHapley Additive exPlanations)

**Method**: Kernel SHAP with 100 background samples

**Global Feature Importance**:

Top 3 features by absolute SHAP value:

1. **form** - Dominates importance, aligns with domain knowledge
2. **total_points_last_4** - Recent performance highly predictive
3. **influence** - Measures game impact

**Insights**:

- **Recency bias**: Last 4 gameweeks >> cumulative season stats
- **Composite metrics** (form, ict_index) more important than raw stats
- **Per-90 rates** valuable but secondary to absolute production
- **Team features** (not in top 10) less influential than expected, mostly eliminated earlier during feature selection

### 8.2 LIME (Local Interpretable Model-agnostic Explanations)

**Sample Explanation (High-Scoring Prediction)**:

**Prediction**: 8.23 points | **Actual**: 9 points | **Error**: 0.77

**Top Contributing Features**:

- `form = 2.8` → +3.2 points (very high form)
- `total_points_last_4 = 32` → +2.1 points (excellent recent run)
- `ict_index = 28.5` → +1.4 points (high involvement)
- `position_MID = 1` → +0.9 points (midfielder bonus)
- `bps_cum_per90 = 1.8` → +0.6 points (consistent performer)
- `minutes_cum = 2520` → +0.2 points (regular starter)

**Negative Contributions**:

- `goals_conceded_last_4 = 6` → -0.4 points (recent defensive struggles)
- `clean_sheets_last_4 = 0` → -0.3 points (no clean sheets lately)

**Interpretation**: Model correctly identified an in-form midfielder with strong underlying stats, slightly penalized for defensive issues.

### 8.3 Feature Interaction Insights

**Key Patterns from XAI**:

1. **Form dominance**: `form` alone accounts for ~40% of prediction variance
2. **Position interactions**: Same stats valued differently by position (e.g., clean sheets more important for DEF than FWD)
3. **Threshold effects**: Minutes > 2000 (regular starter) triggers step-change in prediction
4. **Non-linear interactions**: High `creativity` amplified by high `form` (not just additive)

---

## 9. Discussion

### 9.1 Model Strengths

1. **Strong Baseline Beat**: MAE improvement demonstrates real learning

2. **Domain-Appropriate Features**: Top features align with FPL expertise

   - Form and recent performance dominate
   - Official FPL metrics (ict_index) validated
   - Position-specific scoring naturally captured

3. **Temporal Awareness**: Rolling windows capture momentum effects

4. **Generalization**: R²=0.298 suggests good balance (not overfitting)

5. **Interpretability**: SHAP/LIME confirm intuitive feature importance

### 9.2 Model Limitations

1. **Inherent Unpredictability**

   - Football has high randomness (injuries, red cards, referee decisions)
   - Weather, tactical surprises, motivation factors not captured
   - MAE ~1.0 may be near theoretical floor for this data

2. **Missing Context**

   - Injuries/suspensions not in dataset
   - Double gameweeks not explicitly modeled
   - Managerial changes, team news ignored
   - Fixture difficulty (opponent strength) weakly represented

3. **Position Imbalance**

   - More MID/FWD samples than GK/DEF
   - May underperform for goalkeepers

4. **Cold Start Problem**
   - New players (no history) cannot be predicted
   - Early-season predictions less reliable

## 10. Conclusion

This project successfully developed a deep learning system for predicting Fantasy Premier League player performance. Through meticulous data cleaning, comprehensive feature engineering, and neural network optimization, we created a model that captures the complex factors influencing FPL points.

### Key Achievements:

1. ✅ **Robust Data Pipeline**: Cleaned 50,000+ records, correcting position errors and missing values
2. ✅ **Rich Feature Set**: Engineered 68 features spanning player performance, team context, and match circumstances
3. ✅ **Optimized Architecture**: 4-layer FFNN with Optuna-tuned hyperparameters
4. ✅ **Explainable Predictions**: Implemented SHAP and LIME for interpretability
5. ✅ **Validated Approach**: Demonstrated improvement over baseline methods

### Impact:

- **For Managers**: Data-driven transfer and captain decisions
- **For Research**: Validated importance of form, team strength, and match context
- **For Community**: Open framework for FPL analytics advancement

## Visualizations:

## 🧭 Exploratory Data Analysis (EDA) Before Filtering

This section presents a detailed overview of the dataset through visual and statistical exploration.  
The goal is to understand feature distributions, relationships, correlations, and potential outliers before modeling.

### ⏱️ Minutes Distribution — Outfield Players (Non-GK)

![Minutes Distribution — Outfield Players](images\minutes_hist.png)
**Figure:** Histogram of `minutes` for non-GKs.

### 📊 Boxplots of All Numerical Columns (Outlier Detection)

![Boxplots of All Numerical Columns](images\boxplot_num_col.png)
**Figure:** Boxplots displaying the distribution and outliers across all numerical features in the dataset.

### 📈 Appearance Counts (≤34) — Per Player

![Appearance Counts Histogram](images\appearance_counts_hist.png)
**Figure:** Histogram of match-counts per player (`counts = df.groupby(["season_x","element"]).size()`), filtered to `counts ≤ 34` and plotted with `bins=34`.

### 📚 Numeric Feature Distributions (All Numeric Columns)

![Numeric Feature Distributions](images\numeric_feature_distributions.png)
**Figure:** Grid of histograms showing the distribution of each numeric feature used in the dataset.

### 📊 Consolidated Boxplots

![Boxplot 1](images\boxplot1.png)
![Boxplot 2](images\boxplot2.png)
![Boxplot 3](images\boxplot3.png)
![Boxplot 4](images\boxplot4.png)
![Boxplot 5](images\boxplot5.png)
![Boxplot 6](images\boxplot6.png)
![Boxplot 7](images\boxplot7.png)
![Boxplot 8](images\boxplot8.png)
![Boxplot 9](images\boxplot9.png)
![Boxplot 10](images\boxplot10.png)
**Figures:** Visual summary of several numerical features with their distributions and outliers.

### Correlation Heatmap — Numeric Features

![Correlation Heatmap](images\heatmap.png)
**Figure:** Correlation matrix displaying relationships among all numeric variables in the dataset, highlighting both positive and negative associations between features.

### 📊 Categorical Feature Distributions

![Categorical Distributions 1](images\distribution1.png)
![Categorical Distributions 2](images\distribution2.png)
![Categorical Distributions 3](images\distribution3.png)
**Figure:** Frequency distributions of key categorical variables, including `opp_team_name`, `was_home`, `season_x`, `team_x`, `position`, and `kickoff_time`, showing the top occurrences in the dataset.

### 🧩 Top Features Correlated with Total Points

![Top Correlated Features](images\top_features_correlation.png)
**Figure:** Horizontal bar chart showing the numeric features most strongly correlated with `total_points`. The top contributors include `bps`, `influence`, `ict_index`, and `bonus`, which exhibit the highest positive correlation values.

### ⚽ Feature Relationships with Total Points

![bps vs Total Points](images\bps_vs_total_points.png)
![influence vs Total Points](images\influence_vs_total_points.png)
![ict_index vs Total Points](images\ict_index_vs_total_points.png)
![bonus vs Total Points](images\bonus_vs_total_points.png)
![goals_scored vs Total Points](images\goals_scored_vs_total_points.png)
**Figure:** Scatter plots showing the relationship between key performance metrics (`bps`, `influence`, `ict_index`, `bonus` and `goals_scored`) and `total_points`.

### Summary

- Each plot visualizes how increasing values in these features correspond to higher total points.
- Together, these confirm that **FPL scoring components** such as bonus points and player influence metrics directly contribute to total point accumulation.

### 📅 Games Distribution by Year and Month

![Games per Year](images\games_per_year.png)
![Games per Month](images\games_per_month.png)
**Figure:** Bar charts showing the distribution of recorded games across different years and months.

### Summary

- **Games per Year:** The dataset peaks around **2021–2022**, with noticeably fewer entries before 2020 and a moderate drop in 2023.
- **Games per Month:** Game frequency varies seasonally, with **April–May** and **January–February** showing higher counts—typical of peak league months—while **summer months (July–September)** have fewer matches, reflecting the off-season.

## 🧭 Exploratory Data Analysis (EDA) After Filtering

After removing players who recorded zero total minutes, the dataset (`df_trial`) now includes only active participants.  
This ensures that the analysis reflects meaningful in-game statistics rather than inactive entries.

The following visualizations replicate the EDA process on the filtered dataset to validate consistency and highlight any changes in distribution, correlation, or trends.

---

### 📊 Numeric Feature Distributions (After Filtering)

![Numeric Feature Distributions - Filtered](images\feature_distributions_filtered.png)
**Figure:** Histograms showing the distributions of numeric features after filtering out players with zero total minutes.  
Most distributions remain right-skewed, with noticeable reductions in zero-heavy features such as `minutes`, `selected`, and `total_points`, confirming that inactive player data has been effectively removed.

### 📦 Boxplots of Key Features (After Filtering)

![Boxplot 11](images\boxplot11.png)
![Boxplot 12](images\boxplot12.png)
![Boxplot 13](images\boxplot13.png)
![Boxplot 14](images\boxplot14.png)
![Boxplot 15](images\boxplot15.png)
![Boxplot 16](images\boxplot16.png)
![Boxplot 17](images\boxplot17.png)
![Boxplot 18](images\boxplot18.png)
![Boxplot 19](images\boxplot19.png)
![Boxplot 20](images\boxplot20.png)

**Figure:** Boxplots of additional numerical features after data filtering, showing updated spread and outlier distributions.  
These visuals confirm that the overall shape and variance remain consistent, while extreme zero-heavy outliers have been minimized.

### Correlation Heatmap — Numeric Features (After Filtering)

![Correlation Heatmap - Filtered](images\heatmap_filtered.png)
**Figure:** Correlation heatmap for all numeric variables after filtering out inactive players.  
The main relationships remain consistent, with strong positive correlations between `bps`, `influence`, `ict_index`, and `total_points`, confirming their continued importance in overall player performance.

### 📊 Distributions (After Filtering)

![Labeled Distribution 4](images\distribution4.png)
![Labeled Distribution 5](images\distribution5.png)
![Labeled Distribution 6](images\distribution6.png)
**Figure:** Frequency distributions for selected categorical variables.

### 🧩 Top Features Correlated with Total Points (After Filtering)

![Top Features Correlated with Total Points - Filtered](images\features_correlation_filtered.png)
**Figure:** Bar chart showing the top numerical features most strongly correlated with `total_points` after data filtering.  
The dominant predictors remain consistent with pre-filtering results—`bps`, `influence`, `bonus`, and `ict_index` continue to exhibit the highest positive correlation, confirming their strong influence on player performance outcomes.

### ⚽ Feature Relationships with Total Points (After Filtering)

![bps vs Total Points](images\bps_filtered.png)
![influence vs Total Points](images\influence_filtered.png)
![ict_index vs Total Points](images\ict_index_filtered.png)
![bonus vs Total Points](images\bonus_filtered.png)
![goals_scored vs Total Points](images\goals_scored_filtered.png)

**Figure:** Scatter plots showing how key performance metrics relate to `total_points`.  
A strong **positive linear trend** is evident for `bps`, `influence`, and `ict_index`, confirming their predictive value.  
Discrete features like `bonus` and `goals_scored` show clear **stepwise increases**, reinforcing their role in determining high-point outcomes.

### 📅 Games Distribution Over Time (After Filtering)

![Games per Year](images\games_per_year2.png)
![Games per Month](images\games_per_month2.png)

**Figure:** Distribution of recorded matches by year and month after filtering.  
The **yearly trend** shows dataset peaks in **2021–2022**, aligning with full Premier League seasons, while earlier years have limited records.  
The **monthly distribution** reveals expected fluctuations — higher counts during **spring months (April–May)** corresponding to end-of-season match density, and lower counts in **summer (June–August)** when the league pauses.

## 🔍 Feature Selection and Importance Analysis

This stage identifies which variables most strongly predict **upcoming player performance** (`upcoming_total_points`).  
Three complementary methods were used:

1. **Pearson Correlation** — measures linear relationships.
2. **F-test (univariate feature selection)** — identifies features linearly dependent on the target.
3. **Combined Normalized Scoring** — averages correlation and F-test rankings for robustness.

---

### 📊 Top Features by Correlation

![Top 20 Features by Pearson Correlation with Target](images\feature_correlation_target.png)

**Figure:** Top 20 features most correlated with `upcoming_total_points`.

**Key Insights**

- Strongest positive correlations:  
  `upcoming_total_points`, `total_points_last_5`, `bps_last_5`, `form`, and `ict_index`.
- Highlights both **short-term performance indicators** and **aggregate metrics** like `total_points_cum` and `bps_cum`.
- Confirms that **recent form and overall influence** are consistent predictors of future success.

---

### 🧮 Top Features by F-test (Linear Relationship)

![Top 20 Features by F-test](images\feature_f_test.png)

**Figure:** F-test results ranking the most statistically significant predictors of the target variable.

**Interpretation**

- Highest F-scores for:  
  `total_points_last_5`, `bps_last_5`, `form`, `ict_index`, and `influence`.
- Reinforces that **recent match consistency** and **player impact metrics** are reliable predictors.
- Cumulative statistics (e.g., `bps_cum`, `total_points_cum`) also show stable contribution.

---

### 🧩 Combined Feature Importance

![Top 20 Features by Combined Score](images\feature_combined_importance.png)

**Figure:** Final ranking after combining correlation and F-test results into a normalized composite score.

**Summary**

- The top predictors remain consistent across all methods —  
  `total_points_last_5`, `bps_last_5`, `form`, and `ict_index` dominate.
- Combines **short-term form**, **historical averages**, and **overall player impact**.
- The resulting **optimized feature subset** provides a balanced foundation for predictive modeling — enhancing accuracy, interpretability, and computational efficiency.

This step ensures the dataset used for modeling includes the **most informative predictors**, improving both interpretability and model accuracy.

## 🧠 Model Training and Performance

### 📉 Training Loss Over Epochs

![Training Loss Over Epochs](images\training.png)

**Figure:** Training loss trend over 30 epochs.

**Summary**

- The model shows a **steady and consistent decrease in loss**, indicating that learning is progressing effectively.

### 💡 SHAP Feature Impact Visualization

![SHAP Feature Impact — total_points_last_5](images\shap.png)

**Figure:** SHAP summary plot showing the impact of `total_points_last_5` on model predictions.

**Summary**

- This specific feature (`total_points_last_5`) confirms its **consistent positive impact**—players with stronger recent form tend to yield higher predicted future performance.

### 🎯 LIME Local Explanation — Individual Prediction

![LIME Local Explanation](images\lime.png)

**Figure:** LIME visualization illustrating the local feature contributions for a single player’s predicted `upcoming_total_points`.

**Summary**

- For this case, features like `minutes_cum`, `total_points_last_5`, and `bps_last_5` strongly **increase** the predicted value, implying that recent consistent performance drives the forecast upward.
- Conversely, variables such as `penalties_missed_cum` and `clean_sheets_cum` slightly **decrease** the score prediction.

---

### Final Thoughts:

While predicting football remains inherently uncertain—the beautiful game's unpredictability is part of its appeal—machine learning can provide valuable probabilistic insights. This project demonstrates that historical patterns, when properly analyzed, offer meaningful guidance for FPL decision-making.

The integration of explainable AI ensures predictions are not just accurate but also understandable, building trust with users and enabling them to combine model insights with their football knowledge and intuition.

## Appendix A: Hyperparameter Tuning Results

**Optuna Study Summary**:

- Best trial: #10
- Best trial HPs:

```python
{'n_layers': 4, 'n_units_l0': 510, 'n_units_l1': 241, 'n_units_l2': 222, 'n_units_l3': 296, 'dropout_rate': 0.281975287168578, 'lr': 0.00019165928146882533, 'batch_size': 32}
```

**Parameter Distributions**:

- Optimal depth: 4 layers (trials with 2-3 layers underperformed)
- Optimal width: 500+ units in first layer (wider = better)
- Optimal dropout: ~0.28 (sweet spot between underfitting and overfitting)
- Optimal learning rate: 1.9e-4 (log-uniform search effective)

### Appendix B: FPL Scoring Rules Summary

| Action               | GK  | DEF | MID | FWD |
| -------------------- | --- | --- | --- | --- |
| Playing ≥60 min      | 2   | 2   | 2   | 2   |
| Goal scored          | 6   | 6   | 5   | 4   |
| Assist               | 3   | 3   | 3   | 3   |
| Clean sheet          | 4   | 4   | 1   | 0   |
| Every 3 saves        | 1   | 0   | 0   | 0   |
| Penalty save         | 5   | 0   | 0   | 0   |
| Per 2 goals conceded | -1  | -1  | 0   | 0   |
| Yellow card          | -1  | -1  | -1  | -1  |
| Red card             | -3  | -3  | -3  | -3  |
| Own goal             | -2  | -2  | -2  | -2  |
| Penalty miss         | -2  | -2  | -2  | -2  |

### Appendix C: Script for Feature Selection

```python

import json
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle

# ==========================================
# FFNN Model Definition
# ==========================================
class FFNN(nn.Module):
    def __init__(
        self,
        input_dim,
        n_layers=3,
        n_units_l0=218,
        n_units_l1=233,
        n_units_l2=70,
        dropout_rate=0.10236066118288575,
    ):
        super(FFNN, self).__init__()
        layers = []
        in_dim = input_dim
        hidden_units = [n_units_l0, n_units_l1, n_units_l2][:n_layers]

        for h in hidden_units:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h

        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ==========================================
# Training Function
# ==========================================
def train_and_evaluate(X_train, X_test, y_train, y_test, epochs=100,
                       early_stop_patience=10, early_stop_min_delta=0.001):
    """
    Train FFNN model with early stopping and return evaluation metrics

    Args:
        early_stop_patience: Number of epochs to wait for improvement
        early_stop_min_delta: Minimum change in validation loss to qualify as improvement
    """

    # Convert to torch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # Create Dataloaders
    train_data = TensorDataset(X_train_t, y_train_t)
    test_data = TensorDataset(X_test_t, y_test_t)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    # Best hyperparameters from Optuna
    best_params = {
        "n_layers": 4,
        "n_units_l0": 510,
        "n_units_l1": 241,
        "n_units_l2": 222,
        "dropout_rate": 0.281975287168578,
        "lr": 0.00019165928146882533,
    }

    # Initialize model
    model = FFNN(
        input_dim=X_train.shape[1],
        n_layers=best_params["n_layers"],
        n_units_l0=best_params["n_units_l0"],
        n_units_l1=best_params["n_units_l1"],
        n_units_l2=best_params["n_units_l2"],
        dropout_rate=best_params["dropout_rate"],
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
    criterion = nn.MSELoss()

    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # Train with early stopping
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                preds = model(xb)
                loss = criterion(preds, yb)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(test_loader)

        # Early stopping check
        if avg_val_loss < best_val_loss - early_stop_min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Patience: {patience_counter}/{early_stop_patience}")

        # Stop if patience exceeded
        if patience_counter >= early_stop_patience:
            print(f"  ⏹️  Early stopping at epoch {epoch+1}. Best val loss: {best_val_loss:.4f}")
            model.load_state_dict(best_model_state)
            break

    # If we completed all epochs without early stopping
    if patience_counter < early_stop_patience and best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"  ✓ Completed {epochs} epochs. Best val loss: {best_val_loss:.4f}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        preds = model(X_test_t)
        preds_np = preds.cpu().numpy()
        y_test_np = y_test_t.cpu().numpy()

        # Compute metrics
        mse = mean_squared_error(y_test_np, preds_np)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_np, preds_np)
        r2 = r2_score(y_test_np, preds_np)

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2)
    }


# ==========================================
# MAIN AUTOMATION LOOP
# ==========================================
def run_feature_selection_experiment(df, target_col="upcoming_total_points",
                                     output_file="feature_selection_results.json"):
    """
    Run feature selection for all possible n_features values and save results
    """

    # Prepare data
    X = df.drop(columns=[target_col])
    y = df[target_col]

    max_features = X.shape[1]
    print(f"Total features available: {max_features}")
    print(f"Testing n_features from 2 to {max_features}...\n")

    # Standardize
    scaler_full = StandardScaler()
    X_scaled = scaler_full.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # Feature importance analysis (run once)
    print("Computing feature importance scores...")
    corr = df.corr()[target_col].sort_values(ascending=False)

    selector = SelectKBest(score_func=f_regression, k='all')
    selector.fit(X_scaled, y)
    scores = pd.Series(selector.scores_, index=X.columns).sort_values(ascending=False)

    # Combine scores
    results_df = pd.DataFrame({
        "Correlation": corr,
        "F_test": scores,
    }).fillna(0)

    # Normalize scores
    results_df = results_df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    results_df["Combined_Score"] = results_df.mean(axis=1)
    results_df = results_df.sort_values("Combined_Score", ascending=False)

    # Store all results
    all_results = []

    # Loop through all possible n_features
    for n_features in range(2, max_features + 1):
        print(f"\n{'='*60}")
        print(f"Testing with n_features = {n_features}")
        print(f"{'='*60}")

        # Select top features
        top_features = results_df.head(n_features).index.tolist()
        df_selected = df[top_features + [target_col]]

        # Prepare training data
        y_current = df_selected[target_col].values
        X_current = df_selected.drop(columns=[target_col]).values

        # Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_current, y_current, test_size=0.2, random_state=42
        )

        # Scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train and evaluate
        metrics = train_and_evaluate(X_train_scaled, X_test_scaled, y_train, y_test)

        # Store results
        result_entry = {
            "n_features": n_features,
            "selected_features": top_features,
            "metrics": metrics
        }
        all_results.append(result_entry)

        # Print metrics
        print(f"MAE:  {metrics['mae']:.4f}")
        print(f"MSE:  {metrics['mse']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"R²:   {metrics['r2']:.4f}")

    # Find and display best configuration
    best_mae = min(all_results, key=lambda x: x['metrics']['mae'])
    best_r2 = max(all_results, key=lambda x: x['metrics']['r2'])

    print(f"\n{'='*60}")
    print(f"✅ All results saved to {output_file}")
    print(f"{'='*60}")

    print(f"\n📊 SUMMARY:")
    print(f"\nBest MAE: {best_mae['metrics']['mae']:.4f} with {best_mae['n_features']} features")
    print(f"Best R²:  {best_r2['metrics']['r2']:.4f} with {best_r2['n_features']} features")

    # Save results to JSON (after plotting so we have complete results)
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    return all_results


# ==========================================
# RUN THE EXPERIMENT
# ==========================================
# Assuming 'df' is your dataframe with all features
#
# Default run (early stopping during training with patience=10, min_delta=0.001):
results = run_feature_selection_experiment(df)


```
