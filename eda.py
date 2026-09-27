import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 130

df = pd.read_csv('water_breaks_clean.csv', parse_dates=['date'])

# ---- EDA 1: Breaks per year (trend) ----
# Exclude partial years 2013 (starts Jan1 ok) and 2026 (partial, only through Aug)
yearly = df.groupby('year').size()
fig, ax = plt.subplots(figsize=(9,5))
colors = ['#c44e52' if y == 2026 else '#4c72b0' for y in yearly.index]
ax.bar(yearly.index.astype(str), yearly.values, color=colors)
ax.set_title('Reported Water Main / Service Line Breaks per Year — Bloomington, IN')
ax.set_ylabel('Number of breaks')
ax.set_xlabel('Year')
plt.xticks(rotation=45)
ax.annotate('2026 partial\n(through Aug)', xy=(len(yearly)-1, yearly.iloc[-1]),
            xytext=(len(yearly)-4, yearly.max()*0.8),
            arrowprops=dict(arrowstyle='->'))
plt.tight_layout()
plt.savefig('figs/01_breaks_per_year.png')
plt.close()

# ---- EDA 2: Seasonality — breaks by month ----
monthly = df.groupby('month').size()
month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
fig, ax = plt.subplots(figsize=(9,5))
colors = ['#4c72b0' if m in [11,12,1,2,3] else '#8c8c8c' for m in monthly.index]
ax.bar(month_names, monthly.reindex(range(1,13), fill_value=0).values, color=colors)
ax.set_title('Total Breaks by Month (2013–2026) — Blue = Freeze-Risk Season (Nov–Mar)')
ax.set_ylabel('Number of breaks')
plt.tight_layout()
plt.savefig('figs/02_seasonality.png')
plt.close()

# ---- EDA 3: Material mix over time (era shift) ----
mat = df.dropna(subset=['material_clean'])
mat_year = mat.groupby(['year','material_clean']).size().unstack(fill_value=0)
mat_share = mat_year.div(mat_year.sum(axis=1), axis=0)
top_materials = ['Cast Iron','Ductile Iron','PVC','Copper']
fig, ax = plt.subplots(figsize=(10,5))
mat_share[top_materials].plot(kind='area', stacked=True, ax=ax, alpha=0.85,
                                color=['#c44e52','#4c72b0','#55a868','#ccb974'])
ax.set_title('Share of Breaks by Pipe Material, Over Time')
ax.set_ylabel('Share of breaks (known-material rows)')
ax.set_xlabel('Year')
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig('figs/03_material_share_over_time.png')
plt.close()

# ---- EDA 4: Legacy vs modern pipe break rate by freeze season ----
known = df.dropna(subset=['is_legacy_material']).copy()
known['is_legacy_material'] = known['is_legacy_material'].astype(bool)
ct = pd.crosstab(known['freeze_season'], known['is_legacy_material'], normalize='index')
ct.columns = ['Modern (Ductile/PVC/Copper)','Legacy (Cast Iron/Galv/Transite/Steel)']
ct.index = ['Non-freeze season','Freeze season (Nov-Mar)']
fig, ax = plt.subplots(figsize=(7,5))
ct.plot(kind='bar', stacked=True, ax=ax, color=['#55a868','#c44e52'])
ax.set_title('Material Mix of Breaks: Freeze Season vs Rest of Year')
ax.set_ylabel('Share of breaks')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('figs/04_legacy_vs_freeze.png')
plt.close()

# ---- EDA 5: Break type (main vs service line) trend — service lines only logged from ~2022 ----
type_year = df.groupby(['year','break_category']).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(9,5))
type_year.plot(kind='bar', stacked=True, ax=ax, color=['#4c72b0','#dd8452'])
ax.set_title('Water Main vs Service Line Breaks by Year')
ax.set_ylabel('Number of breaks')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('figs/05_main_vs_service.png')
plt.close()

print("EDA figures saved.")
print()
print("Legacy material share overall:", known['is_legacy_material'].mean().round(3))
print("Legacy share in freeze season:", known[known.freeze_season]['is_legacy_material'].mean().round(3))
print("Legacy share rest of year:", known[~known.freeze_season]['is_legacy_material'].mean().round(3))
print()
print("Material share 2013-2015 vs 2023-2025 (Cast Iron):")
early = mat[mat.year.between(2013,2015)]
late = mat[mat.year.between(2023,2025)]
print("  2013-2015 Cast Iron share:", (early.material_clean=='Cast Iron').mean().round(3))
print("  2023-2025 Cast Iron share:", (late.material_clean=='Cast Iron').mean().round(3))
