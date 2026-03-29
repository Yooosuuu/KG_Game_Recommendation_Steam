""" Data preparation functions """
import ast
import json
import os
import re
import pandas as pd
from rapidfuzz import process, distance
from tqdm import tqdm


def normalize_game_name(name):
    """ Lowercase, strip, remove special characters for matching """
    if pd.isna(name):
        return ""
    return re.sub(r'[^a-z0-9\s]', '', str(name).lower().strip())


def match_games(steam_games_df, steam_user_df,
                col_games='name', col_user='game', threshold=98):
    """ Match game names from user data to Steam games using fuzzy matching. """
    names_userdf = steam_user_df[col_user].unique().tolist()
    names_games = steam_games_df[col_games].tolist()

    print("Normalizing game names...")
    normalized_userdf = [normalize_game_name(n) for n in tqdm(names_userdf, desc="User games")]
    normalized_games = [normalize_game_name(n) for n in tqdm(names_games, desc="Steam games")]

    print("Computing similarity matrix...")
    matrix = process.cdist(normalized_userdf, normalized_games,
                            scorer=distance.Indel.normalized_similarity,
                            workers=-1)

    results = []
    for i, row in enumerate(tqdm(matrix, desc="Matching games")):
        best_idx = row.argmax()
        best_score = row[best_idx] * 100
        if best_score >= threshold:
            results.append({
                'name_user_df': names_userdf[i],
                'name_games': names_games[best_idx],
                'score': best_score
            })

    return pd.DataFrame(results)


def merge_datasets(steam_games_df, steam_user_df, matches_df,
                   col_user='game', col_games='name'):
    """ Merge the two datasets based on matched game names """
    steam_user_df_merged = steam_user_df.merge(
        matches_df[['name_user_df', 'name_games']],
        left_on=col_user,
        right_on='name_user_df',
        how='inner'
    )
    return steam_user_df_merged.merge(
        steam_games_df,
        left_on='name_games',
        right_on=col_games,
        how='inner'
    )


class SteamDataFrame:
    """
    Class to encapsulate a Steam dataset and provide methods for transformation and analysis.
    """

    def __init__(self, df=None):
        self.df = df

    # ── I/O ──────────────────────────────────────────────────────────────────

    def load(self, file_path, encoding='utf-8'):
        """ Load csv with encoding fallback """
        try:
            self.df = pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            self.df = pd.read_csv(file_path, encoding='latin-1')
        return self

    def save(self, file_path):
        """ Save dataframe to csv file """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.df.to_csv(file_path, index=False)
        return self

    # ── Inspection ───────────────────────────────────────────────────────────

    def describe(self):
        """ Quick overview of the dataframe """
        print(f"Shape: {self.df.shape}")
        print(f"Missing values:\n{self.df.isnull().sum()}")
        print(f"Dtypes:\n{self.df.dtypes}")
        return self

    def count_users(self, user_col='ID'):
        """ Count number of distinct users """
        return self.df[user_col].nunique()

    def count_games(self, game_col='game'):
        """ Count number of distinct games """
        return self.df[game_col].nunique()

    def count_games_per_user(self, user_col='ID', game_col='game'):
        """ Count number of distinct games per user """
        return self.df.groupby(user_col)[game_col].nunique().reset_index(
        ).rename(columns={game_col: 'game_count'})
        
    def count_nb_player_specific_game(self, game_name, game_col='game', user_col='ID'):
        """ Count number of distinct players for a specific game """
        return self.df[self.df[game_col] == game_name][user_col].nunique()
    

    # ── Transformations ──────────────────────────────────────────────────────

    def delete_columns(self, columns):
        """ Delete specified columns from dataframe """
        self.df = self.df.drop(columns=columns, errors='ignore')
        return self
    
    def parse_list_column(self, column_name):
        def parse(val):
            if pd.isna(val):
                return []
            try:
                result = ast.literal_eval(str(val))
                if isinstance(result, dict):
                    return list(result.keys())
                if isinstance(result, list):
                    return result
            except (ValueError, SyntaxError):
                return [x.strip() for x in str(val).split(',')]
        self.df[column_name] = self.df[column_name].apply(parse)
        return self

    def filter_users_by_game_count(self, min_games=3,
                                    user_col='ID', game_col='game'):
        """ Keep only users with at least min_games distinct games """
        game_counts = self.count_games_per_user(user_col, game_col)
        valid_users = game_counts[
            game_counts['game_count'] >= min_games
        ][user_col]
        self.df = self.df[self.df[user_col].isin(valid_users)]
        return self

    def filter_to_matched_games(self, matches_df,
                                 game_col='game', match_col='name_user_df'):
        """ Keep only rows whose game appears in the matches dataframe.
            Call this after match_games() to restrict users to matched games """
        matched_games = matches_df[match_col].unique()
        self.df = self.df[self.df[game_col].isin(matched_games)]
        return self
    
    def filter_play_only(self, action_col='action'):
        """ Keep only 'play' rows, drop 'purchase' rows """
        self.df = self.df[self.df[action_col] == 'play']
        return self

    def filter_min_hours(self, min_hours=0.1, hours_col='hours'):
        """ Remove rows where playtime is too low to be meaningful """
        self.df = self.df[self.df[hours_col] >= min_hours]
        return self

    def deduplicate_appid(self, user_col='ID', appid_col='appid'):
        """ Keep only the row with the most hours when a user has 
            the same game matched to multiple appids """
        self.df = (self.df.sort_values('hours', ascending=False)
                        .drop_duplicates(subset=[user_col, appid_col])
                        .reset_index(drop=True))
        return self

    def keep_action(self, action_col='action', action_value='play'):
        """ Keep only rows with a specific action (default: 'play').
            Useful to drop 'purchase' rows and keep only playtime data """
        self.df = self.df[self.df[action_col] == action_value]
        return self

    def rename_columns(self, mapping):
        """ Rename columns using a dict {old_name: new_name} """
        self.df = self.df.rename(columns=mapping)
        return self

    def reset_index(self):
        """ Reset dataframe index """
        self.df = self.df.reset_index(drop=True)
        return self