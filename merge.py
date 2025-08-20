import pandas as pd
import os

# File paths
file_loss = 'outputs/rules_predicting_LOSS.csv'
file_win = 'outputs/rules_predicting_WIN.csv'
output_file = 'master_rules.csv'
OUTDIR = 'outputs'

# Read the CSV files
df_loss = pd.read_csv(file_loss)
df_win = pd.read_csv(file_win)

# Combine the dataframes
df_master = pd.concat([df_loss, df_win], ignore_index=True)

print(f"Master file created: {output_file}")

rules_master_out = os.path.join(OUTDIR, output_file)
df_master.to_csv(rules_master_out, index=False)