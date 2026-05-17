import pandas as pd
import datalog_rules as dr
import kg_embedding as kge
import streamlit as st
from neo4jConnector import Neo4jConnector

"""Interface module for the Steam Game Recommendation system.
This module was for msot part written by AI with some adjustments to match the project structure and ensure compatibility with the Neo4jConnector and other components. It defines a Streamlit-based interface that allows users to select a game and view recommendations based on both Datalog reasoning and KG embedding, as well as a fused view combining both methods.
prompt : 
Given my code, what technology would you recommend for the interface? I was thinking of creating a fairly basic interface (which could be improved later) that, when given a game name as input, outputs a list of games we might like based on the knowledge graph I’ve implemented. I would need three different views for the three recommendation methods (data log rule, embedding, and a mix of both).
"""

# Cache functions to optimize performance in Streamlit interface and allow autocomplete search for game names.
@st.cache_data
def load_all_games():
    """Load all game names from Neo4j and cache the result to optimize performance in Streamlit interface."""
    query = "MATCH (g:Game) RETURN g.name AS name ORDER BY g.name"
    with Neo4jConnector() as conn:
        result = conn._run_query(query, {})
    return sorted([r['name'] for r in result])

# Cache the Neo4j driver to reuse across the Streamlit session and avoid reconnecting on every query.
@st.cache_resource
def get_neo4j_driver():
    """Cache the Neo4j driver for the entire Streamlit session - Written by AI """
    from dotenv import load_dotenv, find_dotenv
    import os
    from neo4j import GraphDatabase
    
    env_path = find_dotenv()
    if env_path:
        load_dotenv(env_path)
    else:
        load_dotenv()
    
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    
    return GraphDatabase.driver(uri, auth=(user, password))

class KG_Interface(Neo4jConnector):
    def __init__(self):
        super().__init__()
        self.driver = get_neo4j_driver()  # Use the cached driver
        self.datalog_reasoner = dr.DatalogReasoner(driver=self.driver)
        self.kg_embedder = kge.KGEmbedder(driver=self.driver)

    def run(self):
        """ Written by AI """
        st.title("Steam Game Recommendation", anchor="top", text_alignment="center")
        st.set_page_config(page_title="Steam Game Recommender", layout="wide")

        game_names = load_all_games()

        game_name_input = st.selectbox(
            "Select a game to get recommendations:",
            game_names,
            index=None,
            placeholder="Type to search...",
            help="Start typing to search for a game. Select one from the dropdown to see recommendations.",
        )

        if game_name_input is None:
            st.info("Choose a game to see recommendations.")
            return

        try:
            appid = self.convert_name_to_appid(game_name_input)
            game_name = self.convert_appid_to_name(appid)

            if not game_name:
                st.error("Game not found in Neo4j. Please check the appid and try again.")
                return

            st.subheader(f"Recommendations for: {game_name} (appid: {appid})", text_alignment="center")

            datalog_df = self.datalog_reasoner.datalog_recommendations_per_game(appid, top_k=10)
            embedding_df = self.kg_embedder.kg_embedding_recommendations(appid, top_k=10)
            fused_df = self.reciprocal_rank_fusion(appid, top_k=10, k=10)

            self.display_three_views(datalog_df, embedding_df, fused_df)

        except ValueError:
            st.error("Invalid game name. Please enter a valid game name.")

    
    def convert_appid_to_name(self, appid):
        """ Convert an appid to a game name using Neo4j """
        query = "MATCH (g:Game {appid: $appid}) RETURN g.name AS name"
        result = self._run_query(query, {"appid": appid})
        return result[0]['name'] if result else None
    
    def convert_name_to_appid(self, name):
        """ Convert a game namee to an appid using Neo4j """
        query = "MATCH (g:Game {name: $name}) RETURN g.appid AS appid"
        result = self._run_query(query, {"name": name})
        return result[0]['appid'] if result else None
    
    def reciprocal_rank_fusion(self, appid, top_k=10, k=10):
        """ Apply Reciprocal Rank Fusion to combine Datalog and KG embedding recommendations 
            Written by AI and then adjusted to handle edge cases where one or both recommendation sources may be empty, and to normalize scores before fusion.
        """
        datalog_df = self.datalog_reasoner.datalog_recommendations_per_game(appid, top_k=top_k*2)
        embedding_df = self.kg_embedder.kg_embedding_recommendations(appid, top_k=top_k*2)
        
        # Handle cases where one or both recommendation sources are empty
        if datalog_df.empty and embedding_df.empty:
            st.warning("No recommendations found from either Datalog reasoning or KG embedding.")
            return pd.DataFrame(columns=['head_appid', 'recommended_appid', 'rrf_score', 'sources'])
        
        if datalog_df.empty:
            st.warning("No recommendations found from Datalog reasoning. Showing KG embedding recommendations only.")
            embedding_df['sources'] = 'kg_embedding'
            # Normalize score to 0-1 range
            min_score = embedding_df['score'].min()
            max_score = embedding_df['score'].max()
            embedding_df['rrf_score'] = (embedding_df['score'] - min_score) / (max_score - min_score + 1e-8)
            return embedding_df[['head_appid', 'recommended_appid', 'rrf_score', 'sources']].head(top_k)
        
        if embedding_df.empty:
            st.warning("No recommendations found from KG embedding. Showing Datalog reasoning recommendations only.")
            datalog_df['sources'] = 'datalog'
            # Normalize score to 0-1 range
            min_score = datalog_df['score'].min()
            max_score = datalog_df['score'].max()
            datalog_df['rrf_score'] = (datalog_df['score'] - min_score) / (max_score - min_score + 1e-8)
            return datalog_df[['head_appid', 'recommended_appid', 'rrf_score', 'sources']].head(top_k)
        
        # Both sources available - continue with RRF fusion
        datalog_df['source'] = 'datalog'
        embedding_df['source'] = 'kg_embedding'
        
        datalog_df['score'] = (datalog_df['score'] - datalog_df['score'].min()) / (datalog_df['score'].max() - datalog_df['score'].min() + 1e-8)
        embedding_df['score'] = (embedding_df['score'] - embedding_df['score'].min()) / (embedding_df['score'].max() - embedding_df['score'].min() + 1e-8)
        
        datalog_df['rank'] = datalog_df['score'].rank(ascending=False, method='min')
        embedding_df['rank'] = embedding_df['score'].rank(ascending=False, method='min')
        
        combined_df = pd.concat([datalog_df, embedding_df], ignore_index=True)
        combined_df['rrf_score'] = 1 / (k + combined_df['rank'])
        combined_df['sources'] = combined_df['source']        
        combined_df = combined_df.groupby(['head_appid', 'recommended_appid'], as_index=False).agg({
            'rrf_score': 'sum',
            'sources': lambda x: ','.join(set(x))
        })
        fused_df = combined_df.sort_values('rrf_score', ascending=False).head(top_k)
        
        return fused_df
    
    def display_three_views(self, datalog_df, embedding_df, fused_df):
        """ Written by AI and then adjusted """
        datalog_view = datalog_df.copy()
        embedding_view = embedding_df.copy()
        fused_view = fused_df.copy()

        for view in [datalog_view, embedding_view, fused_view]:
            if 'recommended_appid' in view.columns:
                view['Recommended Game'] = view['recommended_appid'].apply(self.convert_appid_to_name)
            if 'recommended_appid' in view.columns:
                view.drop(columns=['recommended_appid'], inplace=True)
            if 'head_appid' in view.columns:
                view.drop(columns=['head_appid'], inplace=True)

        st.subheader("Fusion",
                     help="This view shows the top recommendations after applying Reciprocal Rank Fusion to combine signals from both Datalog reasoning and KG embedding. The 'sources' column indicates which method(s) contributed to each recommendation, and 'rrf_score' reflects the combined relevance score.")
        cols = ['sources', 'Recommended Game', 'rrf_score'] if {'sources', 'rrf_score'}.issubset(fused_view.columns) else fused_view.columns.tolist()
        st.dataframe(fused_view[cols].head(10))
        
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Datalog",
                         help="This view shows the top recommendations based solely on Datalog reasoning rules applied to the knowledge graph. The 'score' column reflects the relevance of each recommendation according to the Datalog rules.")
            cols = ['Recommended Game', 'score'] if 'score' in datalog_view.columns else datalog_view.columns.tolist()
            st.dataframe(datalog_view[cols].head(10))

        with col2:
            st.subheader("Embedding",
                         help="This view shows the top recommendations based on semantic embeddings derived from the knowledge graph. The 'score' column reflects the similarity of each recommendation to the input query.")
            cols = ['Recommended Game', 'score'] if 'score' in embedding_view.columns else embedding_view.columns.tolist()
            st.dataframe(embedding_view[cols].head(10))
    
    
if __name__ == "__main__":
    app = KG_Interface()
    app.run()