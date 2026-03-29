# Knowledge Graph Construction for Game Recommendations with independant functions + class for Knowledge Graph construction with neo4j
import pandas as pd
from neo4j import GraphDatabase
import data_preparation as dp
from dotenv import load_dotenv
import os
import tqdm

class KnowledgeGraphBuilder:
    def __init__(self, merged_df):
        self.merged_df = merged_df
        
    def create_neo4j_graph(self):
        """ Initialize Neo4j driver and create constraints for uniqueness """
        load_dotenv("../.env")
        uri = os.environ["NEO4J_URI"]
        user = os.environ["NEO4J_USERNAME"]
        password = os.environ["NEO4J_PASSWORD"]
        driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver = driver

    def close(self):
        """ Close the Neo4j driver connection """
        self.driver.close()
        
    def __enter__(self):
        """ Context manager entry point to initialize the graph connection and create constraints """
        self.create_neo4j_graph()
        self.create_constraints()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """ Context manager exit point to ensure the graph connection is closed """
        self.close()
        
    def print_graph_summary(self):
        with self.driver.session() as session:
            node_count = session.run("MATCH (n) RETURN count(n)").single()[0]
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r)").single()[0]
            print(f"Nodes: {node_count}, Relationships: {rel_count}")

    def create_constraints(self):
        """ Create uniqueness constraints for Game, Genre, Tag, Developer, Publisher and User nodes in Neo4j """
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (g:Game) REQUIRE g.appid IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (g2:Genre) REQUIRE g2.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Tag) REQUIRE t.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Developer) REQUIRE d.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Publisher) REQUIRE p.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.ID IS UNIQUE")
            
    def _batch_insert(self, query, data_list, batch_size=100):
        """ Helper function to insert data in batches to Neo4j using UNWIND for efficiency """
        with self.driver.session() as session:
            for i in range(0, len(data_list), batch_size):
                batch = data_list[i:i+batch_size]
                session.run(query, {"batch": batch})
    
    def create_game_nodes(self):
        games_data = self.merged_df.df[['appid', 'name', 'release_date', 'price', 'metacritic_score', 'user_score', 'average_playtime_forever', 'estimated_owners', 'positive', 'negative', 'pct_pos_total']].drop_duplicates(subset='appid').to_dict('records')
        self._batch_insert("UNWIND $batch AS row MERGE (g:Game {appid: row.appid}) SET g += row", games_data)
    
    def create_genre_nodes(self):
        unique_genres = (self.merged_df.df['genres']
                            .explode()
                            .dropna()
                            .unique())
        genres_data = [{"name": g} for g in unique_genres if g]
        self._batch_insert(
            "UNWIND $batch AS row MERGE (g:Genre {name: row.name})",
            genres_data
        )
        
    def create_tag_nodes(self):
        unique_tags = (self.merged_df.df['tags']
                            .explode()
                            .dropna()
                            .unique())
        tags_data = [{"name": t} for t in unique_tags if t]
        self._batch_insert(
            "UNWIND $batch AS row MERGE (t:Tag {name: row.name})",
            tags_data
        )

    def create_developer_nodes(self):
        unique_developers = (self.merged_df.df['developers']
                                .explode()
                                .dropna()
                                .unique())
        developers_data = [{"name": d} for d in unique_developers if d]
        self._batch_insert(
            "UNWIND $batch AS row MERGE (d:Developer {name: row.name})",
            developers_data
        )

    def create_publisher_nodes(self):
        unique_publishers = (self.merged_df.df['publishers']
                                .explode()
                                .dropna()
                                .unique())
        publishers_data = [{"name": p} for p in unique_publishers if p]
        self._batch_insert(
            "UNWIND $batch AS row MERGE (p:Publisher {name: row.name})",
            publishers_data
        )

    def create_user_nodes(self):
        users_data = self.merged_df.df[['ID']].drop_duplicates(subset='ID').to_dict('records')
        self._batch_insert("UNWIND $batch AS row MERGE (u:User {ID: row.ID})", users_data)

    def create_all_nodes(self):
        """ Create all nodes in the graph by calling individual node creation methods (Extract + taken from merged_df + build a dict list (data to insert) + use _batch_insert) """
        print("Creating nodes...")
        functions = [
            self.create_game_nodes,
            self.create_genre_nodes,
            self.create_tag_nodes,
            self.create_developer_nodes,
            self.create_publisher_nodes,
            self.create_user_nodes
        ]
        for _ in tqdm.tqdm(range(len(functions)), desc="Nodes"):
            functions[_]()
            
    def create_has_genre(self):
        data = (self.merged_df.df[['appid', 'genres']]
                    .drop_duplicates(subset='appid')
                    .explode('genres')
                    .dropna(subset=['genres']))
        relationships_data = data.rename(columns={'genres': 'genre'}).to_dict('records')
        self._batch_insert(
            """
            UNWIND $batch AS row
            MATCH (g:Game {appid: row.appid}), (gen:Genre {name: row.genre})
            MERGE (g)-[:HAS_GENRE]->(gen)
            """,
            relationships_data
        )

    def create_has_tag(self):
        data = (self.merged_df.df[['appid', 'tags']]
                    .drop_duplicates(subset='appid')
                    .explode('tags')
                    .dropna(subset=['tags']))
        relationships_data = data.rename(columns={'tags': 'tag'}).to_dict('records')
        self._batch_insert(
            """
            UNWIND $batch AS row
            MATCH (g:Game {appid: row.appid}), (t:Tag {name: row.tag})
            MERGE (g)-[:HAS_TAG]->(t)
            """,
            relationships_data
        )
    
    def create_developed_by(self):
        data = (self.merged_df.df[['appid', 'developers']]
                    .drop_duplicates(subset='appid')
                    .explode('developers')
                    .dropna(subset=['developers']))
        relationships_data = data.rename(columns={'developers': 'developer'}).to_dict('records')
        self._batch_insert(
            """
            UNWIND $batch AS row
            MATCH (g:Game {appid: row.appid}), (d:Developer {name: row.developer})
            MERGE (g)-[:DEVELOPED_BY]->(d)
            """,
            relationships_data
        )
    
    def create_published_by(self):
        data = (self.merged_df.df[['appid', 'publishers']]
                    .drop_duplicates(subset='appid')
                    .explode('publishers')
                    .dropna(subset=['publishers']))
        relationships_data = data.rename(columns={'publishers': 'publisher'}).to_dict('records')
        self._batch_insert(
            """
            UNWIND $batch AS row
            MATCH (g:Game {appid: row.appid}), (p:Publisher {name: row.publisher})
            MERGE (g)-[:PUBLISHED_BY]->(p)
            """,
            relationships_data
        )
    
    def create_played(self):
        """ Create PLAYED relationships between User and Game nodes with hours of playtime data as properties """
        user_data = self.merged_df.df[['ID', 'appid', 'hours']].dropna(subset=['hours'])
        relationships_data = user_data.to_dict('records')
        self._batch_insert(
            """
            UNWIND $batch AS row
            MATCH (u:User {ID: row.ID}), (g:Game {appid: row.appid})
            MERGE (u)-[r:PLAYED]->(g)
            SET r.hours = row.hours
            """,
            relationships_data
        )
    
    def create_all_relationships(self):
        """ Create all relationships in the graph by calling individual relationship creation methods """
        print("Creating relationships...")
        functions = [
            self.create_has_genre,
            self.create_has_tag,
            self.create_developed_by,
            self.create_published_by,
            self.create_played
        ]
        for _ in tqdm.tqdm(range(len(functions)), desc="Relationships"):
            functions[_]()