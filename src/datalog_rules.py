from dotenv import load_dotenv
from neo4j import GraphDatabase
import os
from neo4jConnector import Neo4jConnector

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