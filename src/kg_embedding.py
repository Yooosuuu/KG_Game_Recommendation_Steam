import json
import logging
import math
import os
from collections import defaultdict

import pandas as pd
import torch
import tqdm
from pykeen.pipeline import pipeline
from pykeen.predict import predict_target
from pykeen.evaluation import RankBasedEvaluator
from pykeen.triples import TriplesFactory

from neo4jConnector import Neo4jConnector

logger = logging.getLogger(__name__)


class KGEmbedder(Neo4jConnector):
    DEFAULT_RELATIONS = [
        'HAS_GENRE',
        'HAS_TAG',
        'DEVELOPED_BY',
        'PUBLISHED_BY',
        'SIMILAR_TO',
    ]
    REL_SIMILAR = 'SIMILAR_TO'

    def __init__(self, model_dir='../models', use_typed_labels=True, relation_types=None, driver=None):
        super().__init__(driver=driver)
        self.model = None
        self.factory = None
        self.model_dir = model_dir
        self.use_typed_labels = use_typed_labels
        self.relation_types = relation_types or list(self.DEFAULT_RELATIONS)
        self.training_config = {}
        self.train_data = None
        self.test_data = None
        self.val_data = None

    @staticmethod
    def _typed_entity_label(node_labels, appid, name, node_id, use_typed_labels=True):
        """Return a stable label for an entity, optionally prefixed by type."""
        labels = set(node_labels or [])
        if use_typed_labels:
            if 'Game' in labels and appid is not None:
                return f"game:{appid}"
            if 'Genre' in labels and name is not None:
                return f"genre:{name}"
            if 'Tag' in labels and name is not None:
                return f"tag:{name}"
            if 'Developer' in labels and name is not None:
                return f"developer:{name}"
            if 'Publisher' in labels and name is not None:
                return f"publisher:{name}"
            if 'User' in labels and node_id is not None:
                return f"user:{node_id}"

        if appid is not None:
            return str(appid)
        if name is not None:
            return str(name)
        if node_id is not None:
            return str(node_id)
        return None

    @staticmethod
    def _jaccard_similarity(left_set, right_set):
        """Compute Jaccard similarity for two sets defined as the size of the intersection divided by the size of the union of the sets."""
        if not left_set or not right_set:
            return 0.0
        union = left_set | right_set
        if not union:
            return 0.0
        return len(left_set & right_set) / len(union)

    def _normalize_scores(self, scores, method='minmax'):
        """Normalize scores for a single head entity using minmax or softmax."""
        if scores is None:
            return pd.Series(dtype=float)

        scores = pd.Series(scores)
        if scores.empty:
            return scores

        if method == 'softmax':
            max_score = scores.max()
            exps = scores.apply(lambda x: math.exp(x - max_score))
            denom = exps.sum()
            if denom == 0:
                return pd.Series([0.0] * len(scores), index=scores.index)
            return exps / denom

        if method == 'minmax':
            min_score = scores.min()
            max_score = scores.max()
            if max_score > min_score:
                return (scores - min_score) / (max_score - min_score)
            return pd.Series([0.5] * len(scores), index=scores.index)

        return scores

    def _fetch_game_profiles(self):
        """Fetch per-game metadata profiles (genres, tags, developers, publishers)."""
        query = """
            MATCH (g:Game)
            RETURN
                toString(g.appid) AS appid,
                [(g)-[:HAS_GENRE]->(genre:Genre) | toLower(genre.name)] AS genres,
                [(g)-[:HAS_TAG]->(tag:Tag) | toLower(tag.name)] AS tags,
                [(g)-[:DEVELOPED_BY]->(dev:Developer) | toLower(dev.name)] AS developers,
                [(g)-[:PUBLISHED_BY]->(pub:Publisher) | toLower(pub.name)] AS publishers
        """
        rows = self._run_query(query)
        profiles = {}
        for row in rows:
            appid = row.get('appid')
            if appid is None:
                continue
            appid = str(appid)
            profiles[appid] = {
                'genres': set(row.get('genres') or []),
                'tags': set(row.get('tags') or []),
                'developers': set(row.get('developers') or []),
                'publishers': set(row.get('publishers') or []),
            }
        return profiles

    def _compute_metadata_score(self, head_appid, tail_appid, profiles):
        """Compute a metadata overlap score between two game ids."""
        head_profile = profiles.get(head_appid)
        tail_profile = profiles.get(tail_appid)
        if not head_profile or not tail_profile:
            return 0.0

        genre_score = self._jaccard_similarity(head_profile['genres'], tail_profile['genres'])
        tag_score = self._jaccard_similarity(head_profile['tags'], tail_profile['tags'])
        shared_dev = 1.0 if head_profile['developers'] & tail_profile['developers'] else 0.0
        shared_pub = 1.0 if head_profile['publishers'] & tail_profile['publishers'] else 0.0

        # Keep KGEmbedder as the core signal while favoring coherent metadata overlap.
        return (0.45 * tag_score) + (0.35 * genre_score) + (0.15 * shared_dev) + (0.05 * shared_pub)

    def _fetch_scores(self, query, params=None):
        """ Helper function to fetch scores from Neo4j and return as a DataFrame """
        rows = self._run_query(query, params)
        return pd.DataFrame(rows)
    
    def _fetch_existing_similar_to(self):
        """Fetch SIMILAR_TO links already present in the graph."""
        known_links = defaultdict(set)
        query = """
            MATCH (g1:Game)-[:SIMILAR_TO]->(g2:Game)
            RETURN toString(g1.appid) AS head_appid, toString(g2.appid) AS tail_appid
        """
        rows = self._run_query(query)
        for row in rows:
            head = str(row.get('head_appid'))
            tail = str(row.get('tail_appid'))
            if head and tail:
                known_links[head].add(tail)
        return known_links

    def _get_model_game_label_maps(self):
        """Map between Neo4j appids and model entity labels."""
        rows = self._run_query("MATCH (g:Game) RETURN toString(g.appid) AS appid")
        game_ids = [str(r.get('appid')) for r in rows if r.get('appid') is not None]

        entity_labels = set(self.factory.entity_to_id.keys())
        uses_typed_labels = any(label.startswith('game:') for label in entity_labels)

        appid_to_model_label = {}
        model_label_to_appid = {}
        for appid in game_ids:
            if uses_typed_labels:
                candidate_labels = [f"game:{appid}", appid]
            else:
                candidate_labels = [appid, f"game:{appid}"]

            for candidate in candidate_labels:
                if candidate in entity_labels:
                    appid_to_model_label[appid] = candidate
                    model_label_to_appid[candidate] = appid
                    break

        missing = len(game_ids) - len(appid_to_model_label)
        if missing:
            logger.warning("Skipped %s games absent from the trained entity vocabulary.", missing)

        return appid_to_model_label, model_label_to_appid

    def _mmr_rerank(self, candidates_df, profiles, top_k=10, lambda_diversity=0.7, candidate_pool=50):
        """Rerank recommendations using Maximal Marginal Relevance (MMR)."""
        if candidates_df.empty:
            return candidates_df

        work_df = candidates_df.sort_values('score', ascending=False).head(candidate_pool).copy()
        work_df = work_df.drop_duplicates(subset=['tail_appid'])
        work_df['tail_appid'] = work_df['tail_appid'].astype(str)

        def feature_set(appid):
            profile = profiles.get(appid, {})
            features = set()
            for key in ('genres', 'tags', 'developers', 'publishers'):
                features |= set(profile.get(key) or [])
            return features

        feature_cache = {appid: feature_set(appid) for appid in work_df['tail_appid']}
        score_map = work_df.set_index('tail_appid')['score'].to_dict()

        selected = []
        remaining = list(work_df['tail_appid'])
        while remaining and len(selected) < top_k:
            best_id = None
            best_score = None
            for cand_id in remaining:
                relevance = score_map.get(cand_id, 0.0)
                if not selected:
                    mmr_score = relevance
                else:
                    max_sim = max(
                        self._jaccard_similarity(feature_cache[cand_id], feature_cache[sel_id])
                        for sel_id in selected
                    )
                    mmr_score = (lambda_diversity * relevance) - ((1 - lambda_diversity) * max_sim)

                if best_score is None or mmr_score > best_score:
                    best_score = mmr_score
                    best_id = cand_id

            if best_id is None:
                break
            selected.append(best_id)
            remaining.remove(best_id)

        if not selected:
            return work_df.head(top_k)

        order = {appid: idx + 1 for idx, appid in enumerate(selected)}
        reranked = work_df[work_df['tail_appid'].isin(order)].copy()
        reranked['mmr_rank'] = reranked['tail_appid'].map(order)
        reranked = reranked.sort_values('mmr_rank').drop(columns=['mmr_rank'])
        return reranked

    def export_triplets(self, relation_types=None, use_typed_labels=None):
        """Export relationships as triplets (head, relation, tail) for KGEmbedder training."""
        relation_types = relation_types or self.relation_types or self.DEFAULT_RELATIONS
        use_typed_labels = self.use_typed_labels if use_typed_labels is None else use_typed_labels

        query = """
            MATCH (h)-[r]->(t)
            WHERE type(r) IN $relation_types
            RETURN
                labels(h) AS head_labels,
                h.appid AS head_appid,
                h.name AS head_name,
                h.ID AS head_id,
                type(r) AS relation,
                labels(t) AS tail_labels,
                t.appid AS tail_appid,
                t.name AS tail_name,
                t.ID AS tail_id
        """
        result = self._run_query(query, {'relation_types': relation_types})

        rows = []
        for r in result:
            head = self._typed_entity_label(
                r.get('head_labels'),
                r.get('head_appid'),
                r.get('head_name'),
                r.get('head_id'),
                use_typed_labels=use_typed_labels,
            )
            tail = self._typed_entity_label(
                r.get('tail_labels'),
                r.get('tail_appid'),
                r.get('tail_name'),
                r.get('tail_id'),
                use_typed_labels=use_typed_labels,
            )
            relation = r.get('relation')
            if head is None or tail is None or relation is None:
                continue
            head = str(head).strip()
            tail = str(tail).strip()
            relation = str(relation).strip()
            if not head or not tail or not relation:
                continue
            if head == tail:
                continue
            rows.append((head, relation, tail))

        triplets = pd.DataFrame(rows, columns=['head', 'relation', 'tail']).drop_duplicates()
        if not triplets.empty:
            triplets = triplets[['head', 'relation', 'tail']]
        logger.info("Exported %s triplets", len(triplets))
        return triplets

    def _save_training_config(self, config, model_dir):
        """Persist training configuration to a JSON file."""
        path = os.path.join(model_dir, 'kg_training_config.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

    def _load_training_config(self, model_dir):
        """Load training configuration if present on disk."""
        path = os.path.join(model_dir, 'kg_training_config.json')
        if not os.path.exists(path):
            return {}
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def train(
        self,
        model_dir="../models",
        relation_types=None,
        use_typed_labels=None,
        embedding_dim=256,
        epochs=250,
        batch_size=1024,
        learning_rate=5e-4,
        num_negs_per_pos=10,
        create_inverse_triples=True,
        pykeen_model='RotatE', # TransE, DistMult, ComplEx, RotatE (after testing RotatE performs way better for this task)
        force_retrain=False,
        random_state=42,
    ):
        """Train (or load) a KGEmbedder model with configurable hyperparameters."""
        model_dir = model_dir or self.model_dir
        relation_types = relation_types or self.relation_types or self.DEFAULT_RELATIONS
        use_typed_labels = self.use_typed_labels if use_typed_labels is None else use_typed_labels

        triplets = self.export_triplets(relation_types=relation_types, use_typed_labels=use_typed_labels)
        if triplets.empty:
            raise ValueError("No triplets exported from Neo4j. Check graph population and filters.")

        similar_count = int((triplets['relation'] == self.REL_SIMILAR).sum())
        if similar_count == 0:
            logger.warning("0 %s triples in training data. Recommendations may be unreliable.", self.REL_SIMILAR)

        factory = TriplesFactory.from_labeled_triples(
            triplets.values,
            create_inverse_triples=create_inverse_triples,
        )

        model_path = os.path.join(model_dir, 'kg_model.pkl')
        factory_path = os.path.join(model_dir, 'kg_factory.pkl')
        os.makedirs(model_dir, exist_ok=True)

        if not force_retrain and os.path.exists(model_path) and os.path.exists(factory_path):
            logger.info("Loading existing model...")
            self.factory = torch.load(factory_path, weights_only=False)
            self.model = torch.load(model_path, weights_only=False)
            self.training_config = self._load_training_config(model_dir)
            return

        logger.info("Training new KGEmbedder model...")
        train, test, val = factory.split([0.8, 0.1, 0.1], random_state=random_state)
        result = pipeline(
            training=train,
            testing=test,
            validation=val,
            model=pykeen_model,
            model_kwargs={'embedding_dim': embedding_dim},
            optimizer='Adam',
            optimizer_kwargs={'lr': learning_rate},
            negative_sampler='basic',
            negative_sampler_kwargs={'num_negs_per_pos': num_negs_per_pos},
            training_kwargs={'num_epochs': epochs, 'batch_size': batch_size},
            random_seed=random_state,
        )

        self.model = result.model
        self.factory = factory
        self.train_data = train
        self.test_data = test
        self.val_data = val
        torch.save(self.model, model_path)
        torch.save(self.factory, factory_path)

        self.training_config = {
            'relation_types': relation_types,
            'use_typed_labels': use_typed_labels,
            'embedding_dim': embedding_dim,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_negs_per_pos': num_negs_per_pos,
            'create_inverse_triples': create_inverse_triples,
            'random_state': random_state,
        }
        self._save_training_config(self.training_config, model_dir)
        logger.info("Model saved, path: %s.", model_path)
        self.evaluate(model=self.model, test_data=self.test_data)
        
    def evaluate(self, evaluator=None, model=None, test_data=None):
        """Evaluate the trained model on test data and return metrics."""
        model_to_eval = model or self.model
        test_to_use = test_data or self.test_data
        evaluator_to_use = evaluator or RankBasedEvaluator()

        if model_to_eval is None or test_to_use is None:
            raise ValueError("Model and test data must be available for evaluation.")

        mapped_triples = getattr(test_to_use, 'mapped_triples', test_to_use)
        additional_filter_triples = []
        if self.train_data is not None:
            additional_filter_triples.append(self.train_data.mapped_triples)
        if self.val_data is not None:
            additional_filter_triples.append(self.val_data.mapped_triples)

        metrics = evaluator_to_use.evaluate(
            model_to_eval,
            mapped_triples=mapped_triples,
            additional_filter_triples=additional_filter_triples or None,
        )

        # Format MetricResults into a readable dict/string for logging
        metrics_obj = metrics
        metrics_dict = None
        if hasattr(metrics_obj, 'to_flat_dict'):
            try:
                metrics_dict = metrics_obj.to_flat_dict()
            except Exception:
                metrics_dict = None
        if metrics_dict is None and hasattr(metrics_obj, 'to_dict'):
            try:
                metrics_dict = metrics_obj.to_dict()
            except Exception:
                metrics_dict = None

        if metrics_dict is None:
            metrics_str = str(metrics_obj)
        else:
            try:
                metrics_str = json.dumps(metrics_dict, default=str, indent=2, ensure_ascii=False)
            except Exception:
                metrics_str = str(metrics_dict)

        logger.info("Evaluation metrics:\n%s", metrics_str)
        return metrics

    def load_model(self, path='../models'):
        """Load a previously trained model and factory from disk."""
        self.model = torch.load(f'{path}/kg_model.pkl', weights_only=False)
        self.factory = torch.load(f'{path}/kg_factory.pkl', weights_only=False)
        self.training_config = self._load_training_config(path)

    def predict_similar_to(
        self,
        top_k=10,
        candidate_pool=150,
        alpha=0.75,
        min_meta_score=0.025,
        exclude_existing_links=True,
        head_game_ids=None,
        normalize_method='minmax',
        diversify=False,
        diversity_lambda=0.7,
        diversity_candidates=50,
    ):
        """Predict SIMILAR_TO links using KGEmbedder with optional metadata reranking."""
        if self.model is None or self.factory is None:
            raise ValueError("Model and triples factory must be loaded before prediction.")
        if self.REL_SIMILAR not in self.factory.relation_to_id:
            logger.warning("'%s' relation is missing in the loaded triples factory.", self.REL_SIMILAR)
            return pd.DataFrame(columns=['head_label', 'tail_label', 'score'])

        appid_to_model_label, model_label_to_appid = self._get_model_game_label_maps()
        game_ids = sorted(appid_to_model_label.keys())
        if head_game_ids is not None:
            requested = {str(game_id) for game_id in head_game_ids}
            game_ids = [game_id for game_id in game_ids if game_id in requested]
        if not game_ids:
            logger.warning("No game entities found in the current model vocabulary.")
            return pd.DataFrame(columns=['head_label', 'tail_label', 'score'])

        profiles = self._fetch_game_profiles()
        known_similar = self._fetch_existing_similar_to() if exclude_existing_links else defaultdict(set)
        all_predictions = []

        with tqdm.tqdm(total=len(game_ids), desc='Predicting SIMILAR_TO') as pbar:
            for head_appid in game_ids:
                head_label = appid_to_model_label[head_appid]
                preds_obj = predict_target(
                    model=self.model,
                    head=head_label,
                    relation=self.REL_SIMILAR,
                    triples_factory=self.factory,
                )

                df = preds_obj.df.copy()
                if df.empty:
                    pbar.update(1)
                    continue

                df['tail_label'] = df['tail_label'].astype(str)
                df = df[df['tail_label'].isin(model_label_to_appid)]
                df = df[df['tail_label'] != head_label]

                if exclude_existing_links:
                    existing_tails = known_similar.get(head_appid, set())
                    blocked_model_labels = {
                        appid_to_model_label[tail]
                        for tail in existing_tails
                        if tail in appid_to_model_label
                    }
                    if blocked_model_labels:
                        df = df[~df['tail_label'].isin(blocked_model_labels)]

                if df.empty:
                    pbar.update(1)
                    continue

                df = df.nlargest(candidate_pool, 'score').copy()
                df['head_label'] = head_appid
                df['tail_appid'] = df['tail_label'].map(model_label_to_appid)
                df = df[df['tail_appid'].notna()]

                if df.empty:
                    pbar.update(1)
                    continue

                # KGEmbedder score is often negative because PyKEEN uses the negative distance.
                df['kg_score'] = df['score']
                df['kg_score_norm'] = self._normalize_scores(df['kg_score'], method=normalize_method)
                df['meta_score'] = df['tail_appid'].map(
                    lambda tail_appid: self._compute_metadata_score(head_appid, tail_appid, profiles)
                )

                if min_meta_score > 0:
                    filtered = df[df['meta_score'] >= min_meta_score]
                    if not filtered.empty:
                        df = filtered

                # Final ranking score keeps KGEmbedder dominant while improving semantic coherence.
                df['score'] = (alpha * df['kg_score_norm']) + ((1 - alpha) * df['meta_score'])
                df = df.sort_values(['score', 'kg_score'], ascending=False)
                df = df.drop_duplicates(subset=['tail_appid'])

                if diversify:
                    df = self._mmr_rerank(
                        df,
                        profiles,
                        top_k=top_k,
                        lambda_diversity=diversity_lambda,
                        candidate_pool=diversity_candidates,
                    )
                else:
                    df = df.head(top_k)

                if df.empty:
                    pbar.update(1)
                    continue

                df['tail_label'] = df['tail_appid'].astype(str)
                df['rank'] = range(1, len(df) + 1)

                all_predictions.append(
                    df[
                        [
                            'head_label',
                            'tail_label',
                            'score',
                            'rank',
                            'kg_score',
                            'kg_score_norm',
                            'meta_score',
                        ]
                    ]
                )
                pbar.update(1)

        if not all_predictions:
            return pd.DataFrame(
                columns=[
                    'head_label',
                    'tail_label',
                    'score',
                    'rank',
                    'kg_score',
                    'kg_score_norm',
                    'meta_score',
                ]
            )
            
        result = pd.concat(all_predictions, ignore_index=True)
        self.export_predictions(result)
        return result

    def show_predictions(self, predictions=None, top_k=10):
        """Display a preview of predictions or compute them on demand."""
        if predictions is None or (hasattr(predictions, 'empty') and predictions.empty):
            predictions = self.predict_similar_to(top_k=top_k)
        logger.info("%s", predictions.head(20))
        return predictions

    def show_predictions_for_game(self, prediction, game_name, top_k=10):
        """Show top similar games for a specific game name."""
        result = self._run_query(
            "MATCH (g:Game {name: $game_name}) RETURN toString(g.appid) AS appid",
            {'game_name': game_name},
        )
        game_id = result[0]['appid'] if result else None

        if not game_id:
            logger.warning("Game '%s' not found.", game_name)
            return

        if prediction is None or (hasattr(prediction, 'empty') and prediction.empty):
            preds = self.predict_similar_to(top_k=top_k, head_game_ids=[game_id])
        else:
            preds = prediction.copy()

        if 'head_label' not in preds.columns:
            logger.warning("No 'head_label' column in predictions")
            return preds

        preds['head_label'] = preds['head_label'].astype(str)
        preds['tail_label'] = preds['tail_label'].astype(str)
        game_id_str = str(game_id)
        preds = preds[preds['head_label'] == game_id_str]
        preds = preds[preds['tail_label'] != game_id_str]
        preds = preds.sort_values('score', ascending=False).head(top_k)

        if not preds.empty:
            tail_ids = preds['tail_label'].tolist()
            query = """
                UNWIND $tail_ids AS tail_id
                MATCH (g:Game {appid: toInteger(tail_id)})
                RETURN toString(g.appid) AS tail_id, g.name AS tail_name
            """
            result = self._run_query(query, {'tail_ids': tail_ids})
            id_to_name = {r['tail_id']: r['tail_name'] for r in result}

            preds['tail_name'] = preds['tail_label'].map(id_to_name)
            display_cols = ['tail_name', 'score']
            if 'kg_score' in preds.columns:
                display_cols.append('kg_score')
            if 'meta_score' in preds.columns:
                display_cols.append('meta_score')

            logger.info("Top %s similar games to '%s':", top_k, game_name)
            logger.info("%s", preds[display_cols])

        return preds

    def export_predictions(self, predictions, path='../data/similar_games.csv', top_k=10):
        """Export predicted SIMILAR_TO relationships to CSV."""
        export_df = predictions.copy()
        for col in [
            'head_label',
            'tail_label',
            'score',
            'kg_score',
            'kg_score_norm',
            'meta_score',
            'rank',
        ]:
            if col not in export_df.columns:
                export_df[col] = None

        export_df = export_df[
            [
                'head_label',
                'tail_label',
                'score',
                'kg_score',
                'kg_score_norm',
                'meta_score',
                'rank',
            ]
        ]
        export_df = (
            export_df.sort_values('score', ascending=False)
            .groupby('head_label', as_index=False, group_keys=False)
            .head(top_k)
        )
        export_df.to_csv(path, index=False)
        logger.info("Predictions exported to %s", path)

    def load_predictions(self, path='../data/similar_games.csv'):
        """Load predicted SIMILAR_TO relationships from CSV."""
        try:
            df = pd.read_csv(
                path,
                dtype={
                    'head_label': str,
                    'tail_label': str,
                },
            )
        except FileNotFoundError:
            logger.warning("Predictions file not found: %s", path)
            return pd.DataFrame(columns=['tail_id', 'score', 'tail_label', 'head_label'])
        return df

    def show_best_predictions(self, predictions, top_k=10):
        """Show top predicted SIMILAR_TO relationships with game names."""
        if predictions.empty:
            logger.warning("No predictions to show.")
            return

        preds = predictions.copy()
        preds['head_label'] = preds['head_label'].astype(str)
        preds['tail_label'] = preds['tail_label'].astype(str)
        preds = preds[preds['head_label'] != preds['tail_label']]

        game_ids = set(preds['head_label'].tolist() + preds['tail_label'].tolist())
        query = """
            UNWIND $game_ids AS game_id
            MATCH (g:Game {appid: toInteger(game_id)})
            RETURN toString(g.appid) AS game_id, g.name AS game_name
        """
        result = self._run_query(query, {'game_ids': list(game_ids)})
        id_to_name = {r['game_id']: r['game_name'] for r in result}

        preds['head_name'] = preds['head_label'].map(id_to_name)
        preds['tail_name'] = preds['tail_label'].map(id_to_name)

        best_preds = preds.nlargest(top_k, 'score')
        display_cols = ['head_name', 'tail_name', 'score']
        if 'kg_score' in best_preds.columns:
            display_cols.append('kg_score')
        if 'meta_score' in best_preds.columns:
            display_cols.append('meta_score')

        logger.info("Top %s predicted SIMILAR_TO relationships:", top_k)
        logger.info("%s", best_preds[display_cols])
        return best_preds

    def write_predictions(self, predictions, top_k=10, reset_existing=False, batch_size=500):
        """Write top-k predicted SIMILAR_TO relationships to Neo4j with score metadata."""
        if predictions is None or predictions.empty:
            logger.warning("No predictions to write.")
            return 0

        required_columns = {'head_label', 'tail_label', 'score'}
        if not required_columns.issubset(set(predictions.columns)):
            raise ValueError(f"Predictions must include columns: {sorted(required_columns)}")

        preds = predictions.copy()
        preds['head_label'] = preds['head_label'].astype(str)
        preds['tail_label'] = preds['tail_label'].astype(str)
        for optional_col in ['rank', 'kg_score', 'kg_score_norm', 'meta_score']:
            if optional_col not in preds.columns:
                preds[optional_col] = None
        preds = preds[preds['head_label'] != preds['tail_label']]
        preds = (
            preds.sort_values('score', ascending=False)
            .groupby('head_label', as_index=False, group_keys=False)
            .head(top_k)
        )

        records = preds.to_dict('records')
        if reset_existing: # Remove all existing SIMILAR_TO relationships created by KGEmbedder before writing new ones.
            self._run_query("MATCH ()-[r:SIMILAR_TO {source: 'kg_embedding'}]->() DELETE r")

        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            self._run_query(
                """
                UNWIND $batch AS row
                MATCH (h:Game {appid: toInteger(row.head_label)})
                MATCH (t:Game {appid: toInteger(row.tail_label)})
                MERGE (h)-[r:SIMILAR_TO]->(t)
                SET
                    r.score = row.score,
                    r.source = 'kg_embedding',
                    r.rank = row.rank,
                    r.kg_score = row.kg_score,
                    r.kg_score_norm = row.kg_score_norm,
                    r.meta_score = row.meta_score
                """,
                {'batch': batch},
            )

        logger.info("Wrote %s SIMILAR_TO relationships to Neo4j", len(records))
        return len(records)

    def kg_embedding_recommendations(self, appid, top_k=10):
        """ Get KG embedding recommendations for a given appid """
        query="""
            MATCH (g:Game {appid: $appid})-[r:SIMILAR_TO]->(rec:Game)
            WHERE r.source = 'kg_embedding'
            RETURN g.appid AS head_appid, rec.appid AS recommended_appid, r.score AS score
            ORDER BY r.score DESC
            LIMIT $top_k
        """
        return self._fetch_scores(query, {"appid": appid, "top_k": top_k})