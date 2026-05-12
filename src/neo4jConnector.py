import os
from dotenv import load_dotenv, find_dotenv
from neo4j import GraphDatabase

class Neo4jConnector:
    """Base class for Neo4j connection management"""
    
    def __init__(self, driver=None):
        self.driver = driver
    
    def connect(self):
        env_path = find_dotenv()
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv()
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
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
        # Use helper _run_query to standardize reads and simplify callers
        node_res = self._run_query("MATCH (n) RETURN count(n) AS node_count")
        rel_res = self._run_query("MATCH ()-[r]->() RETURN count(r) AS rel_count")
        node_count = node_res[0].get('node_count') if node_res else 0
        rel_count = rel_res[0].get('rel_count') if rel_res else 0
        print(f"Nodes: {node_count}, Relationships: {rel_count}")

    def _run_query(self, query, params=None):
        """Run a Cypher query and return a list of dict records.

        This helper centralizes Neo4j access so callers don't need to open sessions
        or repeatedly call `session.run()`. It returns an empty list for write
        queries that don't return records.
        """
        with self.driver.session() as session:
            result = session.run(query, params or {})
            return [dict(r) for r in result]