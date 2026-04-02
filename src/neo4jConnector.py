import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

class Neo4jConnector:
    """Base class for Neo4j connection management"""
    
    def __init__(self):
        self.driver = None
    
    def connect(self):
        load_dotenv("../.env")
        uri = os.environ["NEO4J_URI"]
        user = os.environ["NEO4J_USERNAME"]
        password = os.environ["NEO4J_PASSWORD"]
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        if self.driver:
            self.driver.close()
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
    
    def print_graph_summary(self):
        with self.driver.session() as session:
            node_count = session.run("MATCH (n) RETURN count(n)").single()[0]
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r)").single()[0]
            print(f"Nodes: {node_count}, Relationships: {rel_count}")