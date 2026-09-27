import pandas as pd
import numpy as np
import re

df = pd.read_csv('water_breaks_raw.csv')

# ---- 1. Parse dates ----
df['date'] = pd.to_datetime(df['date'], format='%Y %b %d %I:%M:%S %p', errors='coerce')
df = df.dropna(subset=['date']).copy()
df = df.sort_values('date').reset_index(drop=True)

# ---- 2. Normalize break type ----
def norm_type(t):
    if pd.isna(t): return 'Unknown'
    t = str(t).strip().lower()
    if 'service' in t:
        return 'Service Line'
    if 'main' in t:
        return 'Water Main'
    return 'Unknown'
df['break_category'] = df['type'].apply(norm_type)

# ---- 3. Normalize pipe material into consistent categories ----
def norm_material(m):
    if pd.isna(m): return np.nan
    m = str(m).strip().lower()
    if 'cast iron' in m:
        return 'Cast Iron'          # includes lead joint / MJ variants -> legacy pipe
    if 'ductile' in m:
        return 'Ductile Iron'
    if 'pvc' in m or 'sdr' in m:
        return 'PVC'
    if 'copper' in m:
        return 'Copper'
    if 'galvan' in m:
        return 'Galvanized'
    if 'transite' in m or m == 'ac/transite':
        return 'Transite (AC)'
    if 'steel' in m:
        return 'Steel'
    if 'unknown' in m:
        return np.nan
    return 'Other'

df['material_clean'] = df['pipe_material'].apply(norm_material)

# Legacy/high-risk pipe flag: Cast Iron, Galvanized, Transite, Steel are all
# materials phased out decades ago and known (per water-utility engineering
# literature) to have much higher break rates than modern Ductile Iron/PVC.
legacy = {'Cast Iron', 'Galvanized', 'Transite (AC)', 'Steel'}
df['is_legacy_material'] = df['material_clean'].apply(
    lambda m: np.nan if pd.isna(m) else (m in legacy)
)

# ---- 4. Time features ----
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['month_name'] = df['date'].dt.strftime('%b')
df['doy'] = df['date'].dt.dayofyear
df['dow'] = df['date'].dt.day_name()
df['is_weekend'] = df['date'].dt.dayofweek >= 5

# Freeze season proxy (Bloomington IN climate: hard freezes typically Nov-Mar)
df['freeze_season'] = df['month'].isin([11,12,1,2,3])

# Season label
def season(m):
    if m in (12,1,2): return 'Winter'
    if m in (3,4,5): return 'Spring'
    if m in (6,7,8): return 'Summer'
    return 'Fall'
df['season'] = df['month'].apply(season)

# ---- 5. Geographic proxy: zip code if present, else neighborhood bucket unknown ----
def extract_zip(a):
    if pd.isna(a): return np.nan
    m = re.search(r'\b(474\d{2})\b', str(a))
    return m.group(1) if m else np.nan
df['zip'] = df['address'].apply(extract_zip)

# ---- 6. Only keep real, valid records (drop the literal 'date_repaired' bad row) ----
df = df[df['address'] != 'date_repaired']

df.to_csv('water_breaks_clean.csv', index=False)
print("Rows after cleaning:", len(df))
print("Date range:", df['date'].min(), "to", df['date'].max())
print()
print(df['break_category'].value_counts())
print()
print(df['material_clean'].value_counts(dropna=False))
print()
print("Known-material rows:", df['material_clean'].notna().sum(), "/", len(df))
