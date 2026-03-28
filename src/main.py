import data_preparation as dp

if __name__ == "__main__":
    steam_games = dp.SteamDataFrame().load('../data/cleaned_steam_games.csv')
    user_data = dp.SteamDataFrame().load('../data/cleaned_user_data.csv')
    merged_df = dp.SteamDataFrame().load('../data/merged_user_game_data.csv')
    
    # matches = dp.match_games(steam_games.df, user_data.df)
    # merged_df = (dp.SteamDataFrame(dp.merge_datasets(steam_games.df, user_data.df, matches))
    #                                 .delete_columns(columns=['name_user_df', 'name_games'])
    #                                 .deduplicate_appid()
    #                                 .filter_min_hours()
    #                                 .filter_play_only())
    # merged_df.save('../data/merged_user_game_data.csv')
    
    print(f"Total users: {merged_df.count_users()}")
    print(f"Total games: {merged_df.count_games()}")
    print(f"Average games per user: {merged_df.count_games_per_user()['game_count'].mean():.2f}")
    print(f"Nombre de joueurs avec au moins 2 jeux distincts : {merged_df.count_games_per_user()[merged_df.count_games_per_user()['game_count'] >= 2].shape[0]}")
    print(f"Total games dont les joueurs ont au moins 2 jeux : {merged_df.filter_users_by_game_count(min_games=2).count_games()}")
    print(f"Nombre de joueurs avec au moins 3 jeux distincts : {merged_df.count_games_per_user()[merged_df.count_games_per_user()['game_count'] >= 3].shape[0]}")
    print(f"Total games dont les joueurs ont au moins 3 jeux : {merged_df.filter_users_by_game_count(min_games=3).count_games()}")
    print(f"Nombre de joueurs avec au moins 4 jeux distincts : {merged_df.count_games_per_user()[merged_df.count_games_per_user()['game_count'] >= 4].shape[0]}")
    print(f"Total games dont les joueurs ont au moins 4 jeux : {merged_df.filter_users_by_game_count(min_games=4).count_games()}")