# Knowledge Graph-based Game Recommendation on Steam

## Project Overview

This project builds a Knowledge Graph (KG) from Steam game metadata and user behavioral data to recommend similar games to players. It combines **logic-based reasoning** (Datalog rules) and **KG embeddings** (TransE) to predict game similarity and generate personalized recommendations.

---

## Datasets

| Dataset | Source | Description |
|---|---|---|
| Steam Games Dataset | [Kaggle](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset/data) | ~200 games, 47 columns (genres, tags, developers, scores, playtime...) |
| Steam 200k User Interactions | Kaggle | ~200k users, 4 columns (user_id, game_name, behavior, hours) |

---

## Pipeline

```
1. Data Preparation
   ├── Load Steam Games Dataset
   ├── Load Steam 200k Dataset
   ├── Fuzzy entity matching on game names (rapidfuzz)
   └── Filter users with >= 3 games in common with Steam Games Dataset

2. Knowledge Graph Construction (Neo4j)
   ├── Nodes   : Game, Genre, Tag, Developer, Publisher, User
   └── Relations: HAS_TAG, HAS_GENRE, DEVELOPED_BY, PUBLISHED_BY,
                  PLAYED (with hours), PURCHASED, SIMILAR_TO

3. Logic-based Component (LO2)
   ├── Write Datalog rules encoding game similarity
   │   e.g. SIMILAR_TO(X,Y) ← HAS_TAG(X,T), HAS_TAG(Y,T),
   │                           HAS_GENRE(X,G), HAS_GENRE(Y,G)
   └── Derive SIMILAR_TO links via logical reasoner

4. KG Embedding Component (LO1)
   ├── Export KG triplets
   ├── Train TransE out-of-the-box (PyKEEN)
   └── Predict missing SIMILAR_TO links via link prediction

5. Recommendation
   ├── Combine Datalog-derived + TransE-predicted SIMILAR_TO links
   └── Rank candidate games based on user play history (PLAYED/PURCHASED)

6. Service Layer (LO11)
   ├── Cypher queries for recommendation
   └── Neo4j Bloom for visualization
```

---

## Learning Outcomes Coverage

| LO | Description | Level |
|---|---|---|
| LO1 | KG Embeddings (TransE) | **Focus** |
| LO2 | Datalog rules for similarity | **Focus** |
| LO4 | Data models (property graph vs RDF, temporal) | Basic |
| LO5 | KG Architecture (Neo4j + Python + PyKEEN) | Basic |
| LO6 | Scalable reasoning (Datalog + TransE combined) | Basic |
| LO7 | KG Creation (CSV → Neo4j, entity matching) | Basic |
| LO8 | KG Evolution (link prediction as completion) | Basic |
| LO9 | Real-world applications (game recommendation) | Basic |
| LO10 | Financial angle (publisher ad targeting) | Basic |
| LO11 | Services (Cypher queries, Neo4j Bloom) | Basic |
| LO12 | KG / ML / AI connections | Basic |
| LO3 | Graph Neural Networks | **Not included** |

---

## Tech Stack

| Tool | Role |
|---|---|
| Python | Data preparation, entity matching, pipeline |
| pandas | Data loading and cleaning |
| rapidfuzz | Fuzzy entity matching between datasets |
| Neo4j Desktop | KG storage and querying |
| Cypher | KG queries and recommendation |
| Neo4j Bloom | KG visualization |
| PyKEEN | TransE training and link prediction |
| Jupyter Notebook | Reproducible pipeline |

---

## Project Structure

```
steam-kg-recommendation/
├── data/
│   ├── steam_games.csv          # Steam Games Dataset (Kaggle)
│   └── steam_200k.csv           # Steam 200k Dataset (Kaggle)
├── notebooks/
│   ├── 01_data_preparation.ipynb    # Loading, matching, filtering
│   ├── 02_kg_construction.ipynb     # Neo4j import, node/relation creation
│   ├── 03_datalog_rules.ipynb       # Datalog rules + SIMILAR_TO derivation
│   ├── 04_transe_embedding.ipynb    # PyKEEN TransE training + link prediction
│   └── 05_recommendation.ipynb     # Final recommendation pipeline
├── cypher/
│   └── queries.cypher               # Recommendation Cypher queries
├── README.md
└── requirements.txt
```

---

## Getting Started

```bash
# 1. Clone the repo and install dependencies
pip install -r requirements.txt

# 2. Download datasets from Kaggle and place in data/

# 3. Start Neo4j Desktop and create a local database

# 4. Run notebooks in order (01 → 05)
```

---

## Key Design Decisions

- **TransE used out-of-the-box** via PyKEEN — no modifications to the method.
- **Steam 200k filtered** to users with ≥ 3 games matching the Steam Games Dataset.
- **No production interface** — Cypher queries and Neo4j Bloom serve as the service layer.
- **Datalog rules feed into TransE** — logical similarity links enrich the graph before embedding training.