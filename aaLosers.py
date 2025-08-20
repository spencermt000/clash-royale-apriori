import os
import ast
import argparse
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# -----------------------------
# CLI ARGS
# -----------------------------
parser = argparse.ArgumentParser(description="Association analysis for Clash Royale decks → winning")
parser.add_argument("--csv", default="data/merged_restructured_battles.csv", help="Path to input CSV with columns: player_deck, winner")
parser.add_argument("--min_support", type=float, default=0.01, help="Minimum support for apriori (0-1)")
parser.add_argument("--min_confidence", type=float, default=0.4, help="Minimum confidence for association rules (0-1)")
parser.add_argument("--min_lift", type=float, default=1.0, help="Minimum lift for association rules")
parser.add_argument("--min_cardinality", type=int, default=2, help="Minimum itemset length (e.g., 2 for pairs)")
parser.add_argument("--max_cardinality", type=int, default=4, help="Maximum itemset length (e.g., 4 for quads)")
parser.add_argument("--top", type=int, default=50, help="Top N rows to print for each output table")
parser.add_argument("--outdir", default="outputs", help="Directory to write CSV outputs")
args = parser.parse_args()

IN_CSV = args.csv
OUTDIR = args.outdir
os.makedirs(OUTDIR, exist_ok=True)

# -----------------------------
# LOAD & CLEAN
# -----------------------------
raw_df = pd.read_csv(IN_CSV)
if "player_deck" not in raw_df.columns or "winner" not in raw_df.columns:
    raise ValueError("CSV must contain columns: 'player_deck' and 'winner'")

# Parse stringified Python lists like "['zap-ev1', 'executioner-ev1', ...]"
def parse_deck(cell):
    if pd.isna(cell):
        return []
    if isinstance(cell, list):
        return [str(x).strip() for x in cell]
    try:
        parsed = ast.literal_eval(cell)
        if isinstance(parsed, (list, tuple)):
            return [str(x).strip() for x in parsed]
        # Fallback: just split on comma if it's a plain string
        return [c.strip().strip("'\"") for c in str(cell).strip("[]").split(",") if c.strip()]
    except Exception:
        return [c.strip().strip("'\"") for c in str(cell).strip("[]").split(",") if c.strip()]

# Build transactions: list of card strings per row
trans_cards = raw_df["player_deck"].apply(parse_deck)

# Also build transactions with an outcome label to mine rules that predict WIN
# We'll use strings 'OUTCOME=WIN' and 'OUTCOME=LOSS' so we can filter consequents easily.
trans_with_outcome = []
for cards, win in zip(trans_cards, raw_df["winner"].fillna(0).astype(int)):
    outcome = "OUTCOME=WIN" if win == 1 else "OUTCOME=LOSS"
    # Ensure unique items per transaction (apriori typically assumes sets)
    items = list(dict.fromkeys([c for c in cards if c]))
    items.append(outcome)
    trans_with_outcome.append(items)

# -----------------------------
# ONE-HOT ENCODING (with outcome)
# -----------------------------
te = TransactionEncoder()
onehot = te.fit(trans_with_outcome).transform(trans_with_outcome)
X = pd.DataFrame(onehot, columns=te.columns_)

# Helper: ensure singletons exist in frequent itemsets with correct supports
def ensure_singletons(itemsets_df: pd.DataFrame, X_df: pd.DataFrame) -> pd.DataFrame:
    # collect all items that appear in any itemset of size >= 2
    items_in_sets = set()
    for fs in itemsets_df["itemsets"]:
        for itm in fs:
            items_in_sets.add(itm)
    # which singletons are missing?
    existing_singletons = {next(iter(s)) for s in itemsets_df["itemsets"] if len(s) == 1}
    missing = [itm for itm in items_in_sets if itm not in existing_singletons]
    # append missing singletons with empirical supports from X
    rows = []
    for itm in missing:
        if itm in X_df.columns:
            supp = float(X_df[itm].mean())
            rows.append({"support": supp, "itemsets": frozenset([itm])})
    if rows:
        add_df = pd.DataFrame(rows)
        itemsets_df = pd.concat([itemsets_df, add_df], ignore_index=True)
    return itemsets_df

# -----------------------------
# FREQUENT ITEMSETS (with outcome in the basket)
# -----------------------------
itemsets_all = apriori(
    X,
    min_support=args.min_support,
    use_colnames=True,
    max_len=args.max_cardinality,
)
itemsets_all["length"] = itemsets_all["itemsets"].apply(len)

itemsets_all = ensure_singletons(itemsets_all, X)
itemsets_all["length"] = itemsets_all["itemsets"].apply(len)

# -----------------------------
# ASSOCIATION RULES → Predicting WIN
#   IMPORTANT: association_rules needs singleton supports available for
#   antecedents/consequents. Do NOT drop 1-item itemsets before calling it.
# -----------------------------
# Keep rules whose consequent is exactly {'OUTCOME=LOSS'}
rules = rules[rules["consequents"].apply(lambda s: s == {"OUTCOME=LOSS"})]

rules = association_rules(itemsets_all, metric="confidence", min_threshold=args.min_confidence)

# Keep rules whose consequent is exactly {'OUTCOME=LOSS'}
rules = rules[rules["consequents"].apply(lambda s: s == {"OUTCOME=LOSS"})]

# Filter by antecedent length so that (antecedent_len + 1 == total itemset length)
# aligns with the requested min/max cardinality.
rules["antecedent_len"] = rules["antecedents"].apply(len)
min_ant = max(1, args.min_cardinality - 1)
max_ant = max(1, args.max_cardinality - 1)
rules = rules[(rules["antecedent_len"] >= min_ant) & (rules["antecedent_len"] <= max_ant)]

if args.min_lift is not None:
    rules = rules[rules["lift"] >= args.min_lift]

# Add readable columns
rules = rules.assign(
    antecedents_str=rules["antecedents"].apply(lambda s: ", ".join(sorted(s))),
    consequents_str=rules["consequents"].apply(lambda s: ", ".join(sorted(s))),
)

# Normalize support column names from mlxtend (it may use spaces rather than underscores)
if "antecedent_support" not in rules.columns and "antecedent support" in rules.columns:
    rules = rules.rename(columns={"antecedent support": "antecedent_support"})
if "consequent_support" not in rules.columns and "consequent support" in rules.columns:
    rules = rules.rename(columns={"consequent support": "consequent_support"})

# If still missing, recompute from item supports
if ("antecedent_support" not in rules.columns) or ("consequent_support" not in rules.columns):
    # build a support lookup from itemsets_all
    support_map = {fs: supp for fs, supp in zip(itemsets_all["itemsets"], itemsets_all["support"])}
    def sup_of(fs):
        return float(support_map.get(fs, float("nan")))
    if "antecedent_support" not in rules.columns:
        rules["antecedent_support"] = rules["antecedents"].apply(sup_of)
    if "consequent_support" not in rules.columns:
        rules["consequent_support"] = rules["consequents"].apply(sup_of)

# Reorder columns for readability
col_order = [
    "antecedents_str", "consequents_str", "support", "confidence", "lift", "leverage", "conviction",
    "antecedent_support", "consequent_support"
]
existing = [c for c in col_order if c in rules.columns]
rules = rules[existing].sort_values([c for c in ["lift", "confidence", "support"] if c in rules.columns], ascending=[False, False, False])

rules_out = os.path.join(OUTDIR, "rules_predicting_LOSS.csv")
rules.to_csv(rules_out, index=False)

print("\nTOP rules predicting LOSS (sorted by lift, confidence, support):")
print(rules.head(args.top).to_string(index=False))
print(f"\nSaved full rules to: {rules_out}")

# -----------------------------
# Frequent co-occurring cards among LOSERS only (no outcome label)
# -----------------------------
loss_only = raw_df.loc[raw_df["winner"].fillna(0).astype(int) == 0].copy()
trans_win = loss_only["player_deck"].apply(parse_deck).apply(lambda lst: list(dict.fromkeys([c for c in lst if c])))

te2 = TransactionEncoder()
X_win = te2.fit(trans_win).transform(trans_win)
X_win = pd.DataFrame(X_win, columns=te2.columns_)

itemsets_win = apriori(
    X_win,
    min_support=args.min_support,
    use_colnames=True,
    max_len=args.max_cardinality,
)
itemsets_win["length"] = itemsets_win["itemsets"].apply(len)
itemsets_win = itemsets_win[(itemsets_win["length"] >= args.min_cardinality) & (itemsets_win["length"] <= args.max_cardinality)]

# Nice readable version
itemsets_win = itemsets_win.assign(items_str=itemsets_win["itemsets"].apply(lambda s: ", ".join(sorted(s))))
itemsets_win = itemsets_win[["items_str", "support", "length"]].sort_values(["length", "support"], ascending=[False, False])

itemsets_out = os.path.join(OUTDIR, "frequent_itemsets_losers.csv")
itemsets_win.to_csv(itemsets_out, index=False)

print("\nTOP frequent co-occurring card sets among LOSERS only:")
print(itemsets_win.head(args.top).to_string(index=False))
print(f"\nSaved full itemsets to: {itemsets_out}")

print("\nDone.")