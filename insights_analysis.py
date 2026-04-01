"""
Генерация инсайтов для презентации CupIT.
1. Распределение высот зданий
2. Плотность застройки (расстояния между зданиями)
3. Классификация зданий по форме (панельки, ЖК, склады, гаражи)
4. Распределение площадей зданий
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 150,
})

OUT = Path("Data for presentation/insights")
OUT.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_parquet("data/features.parquet")

# Use final.csv for height (features has nulls for only_a before prediction)
df_final = pd.read_csv("data/final.csv", usecols=["comp_id", "height", "height_source", "area_m2"])
# Merge predicted heights into features
df["height_final"] = df_final["height"].values
df["height_source"] = df_final["height_source"].values

print(f"Total buildings: {len(df):,}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. РАСПРЕДЕЛЕНИЕ ВЫСОТ
# ══════════════════════════════════════════════════════════════════════════════
print("\n═══ 1. Распределение высот ═══")

h = df["height_final"].dropna()
print(f"  Зданий с высотой: {len(h):,}")
print(f"  Мин: {h.min():.1f} м, Макс: {h.max():.1f} м")
print(f"  Среднее: {h.mean():.1f} м, Медиана: {h.median():.1f} м")

# Height categories
bins_h = [0, 5, 10, 15, 20, 30, 50, 100, 500]
labels_h = ["0–5", "5–10", "10–15", "15–20", "20–30", "30–50", "50–100", "100+"]
df["height_cat"] = pd.cut(h, bins=bins_h, labels=labels_h, right=False)
h_dist = df["height_cat"].value_counts().sort_index()

print("\n  Распределение по категориям высот:")
for cat, cnt in h_dist.items():
    pct = cnt / len(h) * 100
    print(f"    {cat:>8} м: {cnt:>7,} ({pct:5.1f}%)")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Histogram
ax = axes[0]
ax.hist(h[h <= 80], bins=80, color="#2196F3", edgecolor="white", linewidth=0.3, alpha=0.85)
ax.set_xlabel("Высота здания (м)")
ax.set_ylabel("Количество зданий")
ax.set_title("Распределение высот зданий Санкт-Петербурга")
ax.axvline(h.median(), color="red", linestyle="--", linewidth=1.5, label=f"Медиана: {h.median():.1f} м")
ax.axvline(h.mean(), color="orange", linestyle="--", linewidth=1.5, label=f"Среднее: {h.mean():.1f} м")
ax.legend()
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

# Bar chart by category
ax = axes[1]
colors_h = ["#E3F2FD", "#90CAF9", "#42A5F5", "#1E88E5", "#1565C0", "#0D47A1", "#4A148C", "#B71C1C"]
bars = ax.bar(h_dist.index, h_dist.values, color=colors_h, edgecolor="white", linewidth=0.5)
ax.set_xlabel("Категория высоты (м)")
ax.set_ylabel("Количество зданий")
ax.set_title("Здания по категориям высот")
for bar, v in zip(bars, h_dist.values):
    pct = v / len(h) * 100
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
            f"{pct:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

plt.tight_layout()
plt.savefig(OUT / "01_height_distribution.png", bbox_inches="tight")
plt.close()
print("  → Сохранено: 01_height_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# 2. ПЛОТНОСТЬ ЗАСТРОЙКИ
# ══════════════════════════════════════════════════════════════════════════════
print("\n═══ 2. Плотность застройки ═══")

# density_100m = total area of buildings within 100m / area of 100m circle
density = df["density_100m"]
n_bld = df["n_buildings_100m"]

print(f"  Плотность застройки (100м радиус):")
print(f"    Среднее: {density.mean():.3f}")
print(f"    Медиана: {density.median():.3f}")
print(f"    Макс: {density.max():.3f}")
print(f"  Среднее кол-во соседей (100м): {n_bld.mean():.1f}")
print(f"  Медиана кол-во соседей (100м): {n_bld.median():.0f}")

# Density categories
bins_d = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 3.0]
labels_d = ["<5%", "5–10%", "10–20%", "20–30%", "30–50%", "50–100%", ">100%"]
df["density_cat"] = pd.cut(density, bins=bins_d, labels=labels_d, right=False)
d_dist = df["density_cat"].value_counts().sort_index()

print("\n  Плотность застройки (доля площади в радиусе 100м):")
for cat, cnt in d_dist.items():
    pct = cnt / len(df) * 100
    print(f"    {cat:>8}: {cnt:>7,} ({pct:5.1f}%)")

# Spacing — nearest neighbor via n_buildings_50m
has_close_nbr = (df["n_buildings_50m"] > 0).sum()
isolated = (df["n_buildings_50m"] == 0).sum()
print(f"\n  Зданий с соседями в 50м: {has_close_nbr:,} ({has_close_nbr/len(df)*100:.1f}%)")
print(f"  Изолированных (>50м до соседа): {isolated:,} ({isolated/len(df)*100:.1f}%)")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Density distribution
ax = axes[0]
ax.hist(density[density <= 0.8], bins=60, color="#4CAF50", edgecolor="white", linewidth=0.3, alpha=0.85)
ax.set_xlabel("Плотность застройки (100м радиус)")
ax.set_ylabel("Количество зданий")
ax.set_title("Плотность застройки вокруг зданий")
ax.axvline(density.median(), color="red", linestyle="--", linewidth=1.5, label=f"Медиана: {density.median():.2f}")
ax.legend()
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

# Number of neighbors
ax = axes[1]
ax.hist(n_bld[n_bld <= 80], bins=40, color="#FF9800", edgecolor="white", linewidth=0.3, alpha=0.85)
ax.set_xlabel("Количество зданий в радиусе 100м")
ax.set_ylabel("Количество зданий")
ax.set_title("Количество соседних зданий (100м)")
ax.axvline(n_bld.median(), color="red", linestyle="--", linewidth=1.5, label=f"Медиана: {n_bld.median():.0f}")
ax.legend()
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

plt.tight_layout()
plt.savefig(OUT / "02_density_distribution.png", bbox_inches="tight")
plt.close()
print("  → Сохранено: 02_density_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3. КЛАССИФИКАЦИЯ ЗДАНИЙ ПО ФОРМЕ
# ══════════════════════════════════════════════════════════════════════════════
print("\n═══ 3. Классификация зданий по форме ═══")

# Classification rules (mutually exclusive, applied in order):
# - Гаражи: area < 100m², n_vertices < 10
# - Современные ЖК: n_vertices > 20, area > 500m²
# - Панельки: compactness > 0.7, rectangularity > 0.8, area > 200m²
# - Склады/промышленные: compactness < 0.4, area > 200m²
# - Прочие: everything else

df["building_class"] = "Прочие"

# Garages first (small, simple)
mask_garage = (df["area_m2"] < 100) & (df["n_vertices"] < 10)
df.loc[mask_garage, "building_class"] = "Гаражи / хозпостройки"

# Modern ЖК (complex shape, large)
mask_zhk = (df["building_class"] == "Прочие") & (df["n_vertices"] > 20) & (df["area_m2"] > 500)
df.loc[mask_zhk, "building_class"] = "Современные ЖК"

# Panel buildings (compact rectangles, medium+)
mask_panel = (df["building_class"] == "Прочие") & (df["compactness"] > 0.7) & (df["rectangularity"] > 0.8) & (df["area_m2"] > 200)
df.loc[mask_panel, "building_class"] = "Панельные дома"

# Warehouses/industrial (irregular, large)
mask_warehouse = (df["building_class"] == "Прочие") & (df["compactness"] < 0.4) & (df["area_m2"] > 200)
df.loc[mask_warehouse, "building_class"] = "Склады / промышленные"

class_dist = df["building_class"].value_counts()
print("\n  Классификация зданий:")
for cls, cnt in class_dist.items():
    pct = cnt / len(df) * 100
    avg_h = df.loc[df["building_class"] == cls, "height_final"].mean()
    avg_a = df.loc[df["building_class"] == cls, "area_m2"].mean()
    print(f"    {cls:<25s}: {cnt:>7,} ({pct:5.1f}%)  avg_h={avg_h:.1f}м  avg_area={avg_a:.0f}м²")

# Per-class stats
print("\n  Детали по классам:")
for cls in ["Панельные дома", "Современные ЖК", "Склады / промышленные", "Гаражи / хозпостройки", "Прочие"]:
    sub = df[df["building_class"] == cls]
    if len(sub) == 0:
        continue
    print(f"\n    {cls}:")
    print(f"      Кол-во: {len(sub):,}")
    print(f"      Площадь: {sub['area_m2'].median():.0f} м² (медиана), {sub['area_m2'].mean():.0f} м² (среднее)")
    print(f"      Высота: {sub['height_final'].median():.1f} м (медиана), {sub['height_final'].mean():.1f} м (среднее)")
    print(f"      Compactness: {sub['compactness'].median():.3f} (медиана)")
    print(f"      Rectangularity: {sub['rectangularity'].median():.3f} (медиана)")
    print(f"      n_vertices: {sub['n_vertices'].median():.0f} (медиана)")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Pie chart
ax = axes[0]
order = ["Панельные дома", "Современные ЖК", "Склады / промышленные", "Гаражи / хозпостройки", "Прочие"]
sizes = [class_dist.get(c, 0) for c in order]
colors_cls = ["#42A5F5", "#AB47BC", "#FF7043", "#78909C", "#BDBDBD"]
explode = [0.03] * 5
wedges, texts, autotexts = ax.pie(
    sizes, labels=order, autopct="%1.1f%%", colors=colors_cls,
    explode=explode, startangle=90, pctdistance=0.8
)
for t in autotexts:
    t.set_fontsize(11)
    t.set_fontweight("bold")
ax.set_title("Классификация зданий по форме")

# Bar chart with avg height per class
ax = axes[1]
avg_heights = [df.loc[df["building_class"] == c, "height_final"].mean() for c in order]
avg_areas = [df.loc[df["building_class"] == c, "area_m2"].mean() for c in order]
x = np.arange(len(order))
w = 0.35
bars1 = ax.bar(x - w/2, avg_heights, w, label="Ср. высота (м)", color="#42A5F5", edgecolor="white")
ax2 = ax.twinx()
bars2 = ax2.bar(x + w/2, avg_areas, w, label="Ср. площадь (м²)", color="#FF9800", alpha=0.7, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels([c.replace(" / ", "\n") for c in order], fontsize=10)
ax.set_ylabel("Средняя высота (м)", color="#42A5F5")
ax2.set_ylabel("Средняя площадь (м²)", color="#FF9800")
ax.set_title("Средняя высота и площадь по классам")
ax.legend(loc="upper left")
ax2.legend(loc="upper right")

for bar, v in zip(bars1, avg_heights):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#1565C0")

plt.tight_layout()
plt.savefig(OUT / "03_building_classification.png", bbox_inches="tight")
plt.close()
print("  → Сохранено: 03_building_classification.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4. РАСПРЕДЕЛЕНИЕ ПЛОЩАДЕЙ
# ══════════════════════════════════════════════════════════════════════════════
print("\n═══ 4. Распределение площадей ═══")

area = df["area_m2"]
print(f"  Мин: {area.min():.1f} м², Макс: {area.max():.0f} м²")
print(f"  Среднее: {area.mean():.0f} м², Медиана: {area.median():.0f} м²")

bins_a = [0, 50, 100, 200, 500, 1000, 2000, 5000, 100000]
labels_a = ["<50", "50–100", "100–200", "200–500", "500–1k", "1k–2k", "2k–5k", "5k+"]
df["area_cat"] = pd.cut(area, bins=bins_a, labels=labels_a, right=False)
a_dist = df["area_cat"].value_counts().sort_index()

print("\n  Распределение площадей:")
for cat, cnt in a_dist.items():
    pct = cnt / len(df) * 100
    print(f"    {cat:>8} м²: {cnt:>7,} ({pct:5.1f}%)")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Histogram (log scale)
ax = axes[0]
ax.hist(area[area <= 5000], bins=100, color="#9C27B0", edgecolor="white", linewidth=0.3, alpha=0.85)
ax.set_xlabel("Площадь здания (м²)")
ax.set_ylabel("Количество зданий")
ax.set_title("Распределение площадей зданий")
ax.axvline(area.median(), color="red", linestyle="--", linewidth=1.5, label=f"Медиана: {area.median():.0f} м²")
ax.axvline(area.mean(), color="orange", linestyle="--", linewidth=1.5, label=f"Среднее: {area.mean():.0f} м²")
ax.legend()
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

# Bar chart by category
ax = axes[1]
colors_a = ["#F3E5F5", "#CE93D8", "#AB47BC", "#8E24AA", "#6A1B9A", "#4A148C", "#311B92", "#1A237E"]
bars = ax.bar(a_dist.index, a_dist.values, color=colors_a, edgecolor="white", linewidth=0.5)
ax.set_xlabel("Категория площади (м²)")
ax.set_ylabel("Количество зданий")
ax.set_title("Здания по категориям площадей")
for bar, v in zip(bars, a_dist.values):
    pct = v / len(df) * 100
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
            f"{pct:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

plt.tight_layout()
plt.savefig(OUT / "04_area_distribution.png", bbox_inches="tight")
plt.close()
print("  → Сохранено: 04_area_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5. СВОДНАЯ ТАБЛИЦА — для слайда
# ══════════════════════════════════════════════════════════════════════════════
print("\n═══ СВОДКА ДЛЯ ПРЕЗЕНТАЦИИ ═══")
print(f"""
╔══════════════════════════════════════════════════════════════╗
║  КЛЮЧЕВЫЕ ЦИФРЫ ПО ЗАСТРОЙКЕ САНКТ-ПЕТЕРБУРГА              ║
╠══════════════════════════════════════════════════════════════╣
║  Всего зданий:        {len(df):>10,}                          ║
║  Медианная высота:    {h.median():>10.1f} м                       ║
║  Средняя высота:      {h.mean():>10.1f} м                       ║
║  Медианная площадь:   {area.median():>10.0f} м²                      ║
║  Средняя площадь:     {area.mean():>10.0f} м²                      ║
║                                                              ║
║  Низкая застройка (<10м):  {(h < 10).sum():>7,} ({(h < 10).sum()/len(h)*100:.1f}%)             ║
║  Средняя (10-30м):         {((h >= 10) & (h < 30)).sum():>7,} ({((h >= 10) & (h < 30)).sum()/len(h)*100:.1f}%)             ║
║  Высотная (>30м):          {(h >= 30).sum():>7,} ({(h >= 30).sum()/len(h)*100:.1f}%)              ║
║                                                              ║
║  Ср. соседей в 100м:  {n_bld.mean():>10.0f}                          ║
║  Ср. плотность (100м):{density.mean():>10.1%}                       ║
╚══════════════════════════════════════════════════════════════╝
""")

print("Все графики сохранены в:", OUT)
