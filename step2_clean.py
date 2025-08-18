import pandas as pd
import ast
import os

# Step 1: Load and merge all CSVs in 'files/' directory
folder_path = "files"
csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")]

df_list = [pd.read_csv(file) for file in csv_files]
df = pd.concat(df_list, ignore_index=True)

df = df.drop_duplicates(subset='battle_id')

# Step 2: Restructure to one row per player
rows = []

for _, row in df.iterrows():
    for player_num in [1, 2]:
        try:
            deck = ast.literal_eval(row[f"player_{player_num}_deck"])
            stats = ast.literal_eval(row[f"player_{player_num}_stats"])
            rows.append({
                "battle_id": row["battle_id"],
                "result": row['game_result'],
                "player_num": player_num,
                "player_deck": deck,
                "avg_elixir": stats.get("avg_elixir"),
                "four_card_cycle": stats.get("four_card_cycle"),
                "elixir_leaked": stats.get("elixir_leaked"),
                "tower_hp": stats.get("tower_hp"),
            })
        except Exception as e:
            print(f"Skipping row due to error: {e}")
            continue


# Step 3: Create and display new DataFrame
restructured_df = pd.DataFrame(rows)
restructured_df['winner'] = restructured_df['player_num'].apply(lambda x: 1 if x == 1 else 0)

def get_opponent_tower_hp(group):
    return group['tower_hp'].iloc[::-1].values  # flip the order

restructured_df['opp_tower_hp'] = restructured_df.groupby('battle_id')['tower_hp'].transform(lambda x: x.iloc[::-1].values)
restructured_df['dmg_dif'] = restructured_df['tower_hp'] - restructured_df['opp_tower_hp']
restructured_df.to_csv("merged_restructured_battles.csv", index=False)

df = pd.read_csv("merged_restructured_battles.csv")  # Update path if needed

# Master set to hold unique cards
unique_cards = set()

for deck_str in df["player_deck"]:
    try:
        deck_list = ast.literal_eval(deck_str)
        unique_cards.update(deck_list)
    except Exception as e:
        print(f"Skipping deck due to error: {e}")

# Convert to sorted list if needed
card_list = sorted(unique_cards)

#print("Unique cards:", card_list)
print(f"Total unique cards: {len(card_list)}")

# Optional: save to file
with open("unique_cards.txt", "w") as f:
    for card in card_list:
        f.write(card + "\n")


