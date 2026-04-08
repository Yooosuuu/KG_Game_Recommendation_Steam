from dotenv import load_dotenv
from neo4j import GraphDatabase
import os
from neo4jConnector import Neo4jConnector
import tqdm
import pandas as pd

class DatalogReasoner(Neo4jConnector):
    def __init__(self):
        super().__init__()
        
    def __enter__(self):
        self.connect()
        return self
            
    def _execute_rule(self, query):
        """ Helper function to execute a Cypher query and return the number of relationships created """
        with self.driver.session() as session:
            result = session.run(query)
            return result.consume().counters.relationships_created

    def _fetch_scores(self, query):
        """ Helper function to fetch scores from Neo4j and return as a DataFrame """
        with self.driver.session() as session:
            results = session.run(query).data()
        return pd.DataFrame(results)
    
    def _write_similar_to(self, df, batch_size=200):
        """ Write top 10 SIMILAR_TO relationships per game to Neo4j """
        records = df.to_dict('records')
        with self.driver.session() as session:
            for i in range(0, len(records), batch_size):
                batch = records[i:i+batch_size]
                session.run("""
                    UNWIND $batch AS row
                    MATCH (g1:Game {appid: row.appid1})
                    MATCH (g2:Game {appid: row.appid2})
                    MERGE (g1)-[r:SIMILAR_TO]->(g2)
                    SET r.score = row.score
                    MERGE (g2)-[r2:SIMILAR_TO]->(g1)
                    SET r2.score = row.score
                """, {"batch": batch})
        return len(records)

    def compute_genre_tag_scores(self):
        """ Rule 1 : SIMILAR_TO(X,Y) ← HAS_GENRE(X,G), HAS_GENRE(Y,G), HAS_TAG(X,T), HAS_TAG(Y,T), X≠Y
            Score : (1 + shared_genres) + (0.5 * shared_tags), min 3 shared tags """
        query = """
            MATCH (g1:Game)-[:HAS_GENRE]->(genre:Genre)<-[:HAS_GENRE]-(g2:Game)
            MATCH (g1)-[:HAS_TAG]->(tag:Tag)<-[:HAS_TAG]-(g2)
            WHERE g1.appid < g2.appid
            WITH g1, g2, count(DISTINCT genre) AS shared_genres,
                 count(DISTINCT tag) AS shared_tags
            WHERE shared_tags >= 3
            RETURN g1.appid AS appid1, g2.appid AS appid2,
                   (1 + shared_genres) + (0.5 * shared_tags) AS score
        """
        df = self._fetch_scores(query)
        print(f"[genre+tag] Candidate pairs: {len(df)}")
        return df

    def compute_developer_scores(self):
        """ Rule 2 : SIMILAR_TO(X,Y) ← DEVELOPED_BY(X,D), DEVELOPED_BY(Y,D), X≠Y
            Score : +3 """
        query = """
            MATCH (g1:Game)-[:DEVELOPED_BY]->(dev:Developer)<-[:DEVELOPED_BY]-(g2:Game)
            WHERE g1.appid < g2.appid
            WITH g1, g2, count(DISTINCT dev) AS shared_devs
            RETURN g1.appid AS appid1, g2.appid AS appid2,
                   3 AS score
        """
        df = self._fetch_scores(query)
        print(f"[developer] Candidate pairs: {len(df)}")
        return df

    def compute_publisher_scores(self):
        """ Rule 3 : SIMILAR_TO(X,Y) ← PUBLISHED_BY(X,P), PUBLISHED_BY(Y,P), X≠Y
            Score : +1 """
        query = """
            MATCH (g1:Game)-[:PUBLISHED_BY]->(pub:Publisher)<-[:PUBLISHED_BY]-(g2:Game)
            WHERE g1.appid < g2.appid
            RETURN g1.appid AS appid1, g2.appid AS appid2,
                   1 AS score
        """
        df = self._fetch_scores(query)
        print(f"[publisher] Candidate pairs: {len(df)}")
        return df

    def compute_coplayed_scores(self):
        """ Rule 4 : CO_PLAYED(X,Y) ← PLAYED(U,X), PLAYED(U,Y), common_users >= 5
            Score : common_users / (total_plays_X + total_plays_Y) * 2 -> normalized score between 0 and 2 """
        query = """
            MATCH (u:User)-[:PLAYED]->(g1:Game)
            MATCH (u)-[:PLAYED]->(g2:Game)
            WHERE g1.appid < g2.appid
            WITH g1, g2, count(u) AS common_users
            WHERE common_users >= 5
            MATCH (:User)-[:PLAYED]->(g1)
            WITH g1, g2, common_users, count(*) AS plays_g1
            MATCH (:User)-[:PLAYED]->(g2)
            WITH g1, g2, common_users, plays_g1, count(*) AS plays_g2
            RETURN g1.appid AS appid1, g2.appid AS appid2,
                toFloat(common_users) / (plays_g1 + plays_g2 - common_users) * 2 AS score
        """
        df = self._fetch_scores(query)
        print(f"[coplayed] Candidate pairs: {len(df)}")
        return df

    def _merge_and_top10(self, dfs):
        """ Merge all score DataFrames, sum scores per pair, keep top 10 per game """
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined[combined['score'] >= 1.0]
        
        # Sum scores for pairs that appear in multiple rules
        combined = (combined.groupby(['appid1', 'appid2'], as_index=False)
                            .agg({'score': 'sum'}))
        
        # Top 10 per game (considering both appid1 and appid2)
        top10_left = (combined.sort_values('score', ascending=False)
                              .groupby('appid1')
                              .head(10))
        top10_right = (combined.sort_values('score', ascending=False)
                               .groupby('appid2')
                               .head(10))
        
        top10 = pd.concat([top10_left, top10_right]).drop_duplicates(
            subset=['appid1', 'appid2']
        )
        print(f"Total pairs after top10 filter: {len(top10)}")
        return top10
    
    def enforce_top10(self):
        """ Keep only top 10 SIMILAR_TO per game, delete the rest """
        query = """
            MATCH (g:Game)-[r:SIMILAR_TO]->()
            WITH g, r ORDER BY r.score DESC
            WITH g, collect(r) AS rels
            FOREACH (r IN rels[10..] | DELETE r)
        """
        with self.driver.session() as session:
            session.run(query)
        print("Top 10 enforced per game")
    
    def reset_similar_to(self):
        """ Delete all SIMILAR_TO relationships """
        with self.driver.session() as session:
            result = session.run("MATCH ()-[r:SIMILAR_TO]->() DELETE r")
            print(f"SIMILAR_TO relationships deleted")

    def apply_all_rules(self):
        """ Compute all scores, merge, keep top 10 per game, write to Neo4j """
        print("Computing Datalog rule scores...")
        
        dfs = []
        rules = [
            self.compute_genre_tag_scores,
            self.compute_developer_scores,
            self.compute_publisher_scores,
            self.compute_coplayed_scores,
        ]
        for func in tqdm.tqdm(rules, desc="Computing scores"):
            df = func()
            if not df.empty:
                dfs.append(df)
        
        print("Merging scores and filtering top 10...")
        top10 = self._merge_and_top10(dfs)
        
        print("Writing SIMILAR_TO relationships to Neo4j...")
        created = self._write_similar_to(top10)
        self.enforce_top10()
        print(f"Total SIMILAR_TO written: {created}")
        self.print_graph_summary()