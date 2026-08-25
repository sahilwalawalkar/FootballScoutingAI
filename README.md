# ScoutVision AI

An interactive football recruitment intelligence platform for exploring, comparing, and shortlisting Premier League players using position-relative performance data.

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.62%2B-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6%2B-F7931E?logo=scikitlearn&logoColor=white)

## Overview

ScoutVision turns a 2023/24 Premier League player dataset into a practical scouting workflow. It combines interactive filtering, position-specific percentile profiles, role scores, cosine-similarity modelling, and a working recruitment shortlist in one Streamlit application.

The project currently covers:

- 339 players
- 20 Premier League clubs
- Four position groups
- 132 raw, derived, percentile, and role-scoring fields
- Player records with at least 900 league minutes

> ScoutVision is a decision-support project. Statistical output should complement video scouting, tactical analysis, medical review, character assessment, and financial due diligence.

## Product highlights

### Command center

Review league-wide coverage, leading scout scores, the relationship between age and performance, and role-specific leaderboards.

### Player finder

Create a target pool using position, club, age, minutes, and scout-score filters. Select any result to open its player dossier without leaving the page.

### Head-to-head comparison

Compare positional peers across core information, role scores, position-relative radar profiles, and individual metrics.

### Recruitment lab

Choose a reference player and discover stylistically similar alternatives. Candidates can be constrained by age, minutes, similarity, and current club.

### Shortlist

Save targets throughout the app, review their dossiers, remove candidates, and export the shortlist as CSV.

## Analytical approach

The workflow uses position-specific football metrics rather than ranking every player against the same generic feature set.

1. Raw player and team statistics are collected from the project CSV exports.
2. Player identity, club, age, minutes, and position groups are standardized.
3. Per-90 and efficiency metrics are prepared for eligible players.
4. Metrics are converted to percentiles within each position group.
5. Tactical role scores are assembled for profiles such as ball-playing defender, ball-winning midfielder, and goal scorer.
6. Similar players are identified by standardizing position-specific features and calculating cosine similarity.

The current similarity model requires at least 70% feature coverage. Missing values within an eligible position pool are median-imputed before scaling.

## Technology

- **Application:** Streamlit
- **Data processing:** pandas and NumPy
- **Machine learning:** scikit-learn
- **Visualization:** Plotly
- **Analysis:** Jupyter Notebook

## Project structure

```text
FootballScoutingAI/
├── .streamlit/
│   └── config.toml
├── app/
│   └── app.py
├── data/
│   ├── processed/
│   └── raw/
├── notebooks/
│   └── 01_data_preparation.ipynb
├── docs/
├── models/
├── src/
├── tests/
├── .gitignore
├── README.md
└── requirements.txt
```

The current application reads:

```text
data/processed/premier_league_2023_24_scouting_final.csv
```

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/sahilwalawalkar/FootballScoutingAI.git
cd FootballScoutingAI
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the application

From the project root:

```bash
python -m streamlit run app/app.py
```

Open the local URL displayed by Streamlit, normally `http://localhost:8501`.

## Data notes

- The processed application dataset contains players with at least 900 minutes.
- Goalkeepers are present in the dataset, but the current role-score and similarity models focus on defenders, midfielders, and forwards.
- Percentiles describe performance relative to positional peers in this dataset and season; they are not universal player ratings.
- Empty or incomplete source fields can affect coverage and the availability of individual profiles.
- Before redistributing or using the data commercially, review the terms associated with the original data sources.

## Deployment

The app can be deployed on Streamlit Community Cloud:

1. Push this repository to GitHub.
2. Create a new app in Streamlit Community Cloud.
3. Select this repository and branch.
4. Set the entry point to `app/app.py`.
5. Deploy.

No secrets are required for the current local CSV-based version.

## Roadmap

- Add goalkeeper-specific metrics and similarity modelling
- Add transfer-value, wage, contract, and injury context
- Persist shortlists between sessions
- Add team-style and tactical-fit models
- Add automated tests for data quality and scoring
- Expand coverage to additional leagues and seasons

## Responsible use

ScoutVision is an analytical and portfolio project, not an official Premier League product. The model can surface patterns in available data, but it cannot observe off-ball behavior, tactical instructions, injuries, mentality, personality, adaptation risk, or transfer feasibility.

## Author

Developed by [Sahil Walawalkar](https://github.com/sahilwalawalkar).
