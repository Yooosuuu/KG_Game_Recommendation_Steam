import data_preparation as dp
import kg_construction as kg
import datalog_rules as dr

if __name__ == "__main__":
    steam_games = dp.SteamDataFrame().load('../data/cleaned_steam_games.csv')
    user_data = dp.SteamDataFrame().load('../data/cleaned_user_data.csv')    
    merged_df = dp.SteamDataFrame().load('../data/merged_user_game_data.csv')
    
    for col in ['genres', 'tags', 'developers', 'publishers']:
        merged_df.parse_list_column(col)

    # with kg.KnowledgeGraphBuilder(merged_df.filter_users_by_game_count(min_games=3)) as kg_builder:
    #     kg_builder.create_all_nodes()
    #     kg_builder.create_all_relationships()
    #     kg_builder.print_graph_summary()
    
    with dr.DatalogReasoner() as rule_applier:
        rule_applier.reset_similar_to()
        rule_applier.apply_all_rules()
        rule_applier.print_graph_summary()

    # matches = dp.match_games(steam_games.df, user_data.df)
    # merged_df = (dp.SteamDataFrame(dp.merge_datasets(steam_games.df, user_data.df, matches))
    #                                 .delete_columns(columns=['name_user_df', 'name_games'])
    #                                 .deduplicate_appid()
    #                                 .filter_min_hours()
    #                                 .filter_play_only())
    # merged_df.save('../data/merged_user_game_data.csv')
    
    # games_per_user = merged_df.count_games_per_user()
    # print(f"Total users: {merged_df.count_users()}")
    # print(f"Total games: {merged_df.count_games()}")
    # print(f"Average games per user: {games_per_user['game_count'].mean():.2f}")

    # for min_g in [2, 3, 4]:
    #     temp = dp.SteamDataFrame(merged_df.df.copy())
    #     temp.filter_users_by_game_count(min_games=min_g)
    #     gpu = temp.count_games_per_user()
    #     print(f"Min {min_g} games → {gpu.shape[0]} users, {temp.count_games()} distinct games")