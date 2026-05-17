import data_preparation as dp
import kg_construction as kg
import datalog_rules as dr
import kg_embedding as kge
import interface
import logging


if __name__ == "__main__":
    
    # Configuration flags
    RUN_DATA_PREPARATION = False
    RUN_MERGE_PREPARATION = False
    RUN_KG_CONSTRUCTION = False
    RUN_DATALOG_RULES = False
    RUN_KG_EMBEDDING = False
    RUN_INTERFACE = True
    
    # Initialize shared variables
    merged_df = None
    steam_games = None
    user_data = None
    
    ##### DATA PREPARATION #####
    if RUN_DATA_PREPARATION:
        steam_games = dp.SteamDataFrame().load('../data/cleaned_steam_games.csv')
        user_data = dp.SteamDataFrame().load('../data/cleaned_user_data.csv')    
        merged_df = dp.SteamDataFrame().load('../data/merged_user_game_data.csv')
        
        for col in ['genres', 'tags', 'developers', 'publishers']:
            merged_df.parse_list_column(col)
        
    
    ##### MERGE PREPARATION #####
    if RUN_MERGE_PREPARATION or (RUN_DATA_PREPARATION and merged_df is None):
        matches = dp.match_games(steam_games.df, user_data.df)
        merged_df = (dp.SteamDataFrame(dp.merge_datasets(steam_games.df, user_data.df, matches))
                                        .delete_columns(columns=['name_user_df', 'name_games'])
                                        .deduplicate_appid()
                                        .filter_min_hours()
                                        .filter_play_only())
        merged_df.save('../data/merged_user_game_data.csv')    
    
    
    ##### KG CONSTRUCTION #####
    if RUN_KG_CONSTRUCTION:
        # Requires RUN_DATA_PREPARATION to be true !
        with kg.KnowledgeGraphBuilder(merged_df.filter_users_by_game_count(min_games=3)) as kg_builder:
            kg_builder.create_all_nodes()
            kg_builder.create_all_relationships()
            kg_builder.print_graph_summary()


    ##### DATALOG REASONING #####
    if RUN_DATALOG_RULES:
        with dr.DatalogReasoner() as rule_applier:
            rule_applier.reset_similar_to()
            rule_applier.apply_all_rules()
            rule_applier.print_graph_summary()

        
    ##### KG Embedding and Prediction #####
    if RUN_KG_EMBEDDING:
        logging.basicConfig(level=logging.INFO)
        training = False  # Set to False to skip training and load existing model
        write_to_neo4j = True  # Set to False to skip writing predictions to Neo4j
        with kge.KGEmbedder() as embedder:
            if training:
                embedder.train(force_retrain=True, pykeen_model='RotatE', create_inverse_triples=True)
                preds = embedder.predict_similar_to(top_k=10)
            else:
                embedder.load_model()
                preds = embedder.load_predictions("../data/similar_games.csv")
            embedder.show_best_predictions(preds, top_k=60)
            if write_to_neo4j:
                embedder.write_predictions(preds)
    
    ##### INTERFACE #####
    if RUN_INTERFACE:
        app = interface.KG_Interface()
        app.run()