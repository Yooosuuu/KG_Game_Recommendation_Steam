# Knowledge Graph Game Recommendation for Steam

## Overview

This project builds a knowledge-graph-based recommendation system for Steam games. It was conducted as part of the course : 192.194 Knowledge Graphs (VU) 2026S from TU Wien. It combines cleaned Steam datasets, user play and preference data, and graph-based modeling to produce game similarity and personalized recommendations. The repository contains data preparation scripts, use of datalog rules and tools to construct and embed a knowledge graph, and a small interface to run recommendations locally using a Neo4j backend.

## Key Features

- Construct a knowledge graph from Steam and user data
- Use datalog rules or logic to infer relationships and enrich the graph
- Generate graph embeddings for games and users
- Compute game similarities and recommend titles
- Integrate with Neo4j for storage and graph queries

## Repository Structure

- **data/**: CSV datasets used by the project
  - `steam_games.csv`, `user_data.csv`, `cleaned_steam_games.csv`, `cleaned_user_data.csv`, `merged_user_game_data.csv`, `similar_games.csv`
- **models/**: Configuration and training files for embedding and KG training
  - `kg_factory.pkl` — saved PyKEEN TriplesFactory
  - `kg_model.pkl` — pretrained KG embedding model
  - `kg_training_config.json` — configuration for KG embedding/training
- **src/**: Main source code
  - `data_preparation.py` — cleaning and preprocessing raw CSV data
  - `kg_construction.py` — builds the knowledge graph (nodes, relationships)
  - `kg_embedding.py` — routines to compute or load graph embeddings
  - `datalog_rules.py` — Datalog or logic rules used for inference
  - `neo4jConnector.py` — helper to connect and push data to Neo4j
  - `interface.py` — simple Streamlit interface to run recommendations
  - `main.py` — example entrypoint to run the pipeline or experiments
- `neo4j_Credentials.txt` : local file storing Neo4j connection details (not included in the repo for security)
- `readme.md` — this file

## Installation

1. Install Python 3.8+.

2. Install required packages using requirements.txt:

```powershell
pip install -r requirements.txt
```

## Configuration

- Neo4j: Create a local Neo4j instance and place connection info in `neo4j_Credentials.txt` or set environment variables. Keep credentials private.
- `models/kg_training_config.json` contains training and model parameters — edit as needed before running embedding scripts.

## Usage

1. Prepare data (clean and merge): data_preparation.py

2. Build and populate the knowledge graph in Neo4j: kg_construction.py

3. Train or compute embeddings: kg_embedding.py

4. Run the interface or change main.py to execute the pipeline:

for example, to run the Streamlit interface:
```powershell
python -m streamlit run main.py
```

to run the full pipeline in main.py, set the appropriate flags at the top of the file.

## Data Notes

- The `data/` folder contains both raw and cleaned CSVs.
- `similar_games.csv` stores precomputed similarity scores and can be used to quickly fetch recommendations without recomputing embeddings.

## More information about Datalog rules and embeddings

### Datalog Rules

The project implements a set of rule-based scoring functions that derive pairwise `SIMILAR_TO` candidates from the knowledge graph. The rules are implemented in `src/datalog_rules.py` inside the `DatalogReasoner` class. Key behaviours:

- compute_genre_tag_scores: finds pairs of games that share genres and tags. It requires at least 3 shared tags and computes a score as `(1 + shared_genres) + (0.5 * shared_tags)`.
- compute_developer_scores: gives a fixed boost (+3) to game pairs developed by the same developer.
- compute_publisher_scores: gives a small boost (+1) to game pairs published by the same publisher.
- compute_coplayed_scores: computes a co-play signal from shared users, keeping pairs with at least 5 common users; score is `common_users / (plays_g1 + plays_g2 - common_users) * 2` (a normalized co-play score).

The individual rule outputs are merged by `_merge_and_top10`, which:

- concatenates rule outputs, filters out pairs with total score < 1.0,
- sums scores for pairs appearing in multiple rules,
- keeps the top-10 candidates per game (considering both directions),
- and writes the resulting edges to Neo4j with `_write_similar_to`, setting `r.score` and `r.source = 'datalog'`.

Utility functions include `reset_similar_to()` (delete existing `SIMILAR_TO` relations), `enforce_top10()` (prune to top-10 per game), and `datalog_recommendations_per_game(appid, top_k)` to fetch top-ranked datalog recommendations from the graph.

### Embeddings

Graph-embedding based recommendations are implemented in `src/kg_embedding.py` via the `KGEmbedder` class and leverage PyKEEN + PyTorch. Main points:

- Triplet export: `export_triplets()` extracts (head, relation, tail) triples from Neo4j for configured relation types (defaults: `HAS_GENRE`, `HAS_TAG`, `DEVELOPED_BY`, `PUBLISHED_BY`, `SIMILAR_TO`). The exported triples can use typed labels such as `game:<appid>` to keep entity types distinct.
- Training: `train()` builds a `TriplesFactory`, splits data, and runs a PyKEEN `pipeline` (default model used in the code is `RotatE`). Default hyperparameters in the code: `embedding_dim=256`, `epochs=250`, `batch_size=1024`, `learning_rate=5e-4`, `num_negs_per_pos=10`. Trained model and factory are saved to `models/`.
- Prediction: `predict_similar_to()` uses the trained model to score candidate `SIMILAR_TO` links and returns ranked recommendations. The implementation maps Neo4j appids to the model's entity labels, filters/excludes existing links if requested, and supports batch scoring.
- Metadata reranking: predicted candidates can be reranked by a metadata score computed from genre/tag overlap and shared developer/publisher. The metadata score combines signals as `0.45 * tag_score + 0.35 * genre_score + 0.15 * shared_dev + 0.05 * shared_pub` where tag/genre scores use Jaccard similarity between per-game profiles fetched from Neo4j.
- Evaluation & utilities: evaluation uses a RankBasedEvaluator from PyKEEN; helpers exist to save/load predictions (`export_predictions`, `load_predictions`) and write model predictions back to Neo4j (`write_predictions`).

Together, the datalog rules and embedding-based signals can be fused (see `src/interface.py`) to produce hybrid recommendations combining symbolic and learned signals.

## Acknowledgements
- The project was developed as part of the Knowledge Graphs course at TU Wien.
- Datasets sourced from Kaggle and cleaned for use in this project:
  - [Steam Games Dataset](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset?resource=download&select=games_march2025_cleaned.csv)
  - [Steam Video Games Dataset](https://www.kaggle.com/datasets/tamber/steam-video-games?select=steam-200k.csv)