"""
Утилиты для pipeline обработки геоданных зданий СПб.
Changellenge Cup IT 2026 — Кейс МТС.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely import wkt
from shapely.validation import make_valid
from shapely.ops import unary_union
from shapely.geometry import MultiPolygon, Polygon
import warnings

# ============================================================
# КОНФИГ — единое место для всех параметров pipeline
# ============================================================

CONFIG = {
    # Пути
    "path_a": "./cup_it_example_src_A.csv",
    "path_b": "./cup_it_example_src_B.csv",
    "cleaned_a": "./data/cleaned_a.parquet",
    "cleaned_b": "./data/cleaned_b.parquet",
    "matched": "./data/matched.parquet",
    "unified": "./data/unified.parquet",

    # Проекция (UTM zone 36N для СПб)
    "crs_geo": "EPSG:4326",
    "crs_projected": "EPSG:32636",

    # Очистка — Источник А
    "min_area_a": 8.0,  # м², фильтрация мелких объектов

    # Очистка — Источник Б
    "spb_bbox": {  # bounding box Санкт-Петербурга
        "lon_min": 29.4,
        "lon_max": 30.8,
        "lat_min": 59.6,
        "lat_max": 60.2,
    },
    "max_height_b": 500.0,  # м, аномально высокие
    "max_height_diff": 10.0,  # м, |height - stairs*avg| порог

    # Матчинг
    "iou_threshold": 0.1,  # порог IoU для создания ребра графа
    "intersection_min_area": 1.0,  # м², минимальная площадь пересечения
}


# ============================================================
# ЗАГРУЗКА ДАННЫХ
# ============================================================

def load_source_a(path=None):
    """Загрузка Источника А в GeoDataFrame."""
    path = path or CONFIG["path_a"]
    df = pd.read_csv(path, index_col=0)
    geom_series = df["geometry"].apply(lambda w: wkt.loads(w) if pd.notna(w) else None)
    df = df.drop(columns=["geometry"])
    gdf = gpd.GeoDataFrame(df, geometry=geom_series.values, crs=CONFIG["crs_geo"])
    return gdf


def load_source_b(path=None):
    """Загрузка Источника Б в GeoDataFrame."""
    path = path or CONFIG["path_b"]
    df = pd.read_csv(path, index_col=0)
    geometry = df["wkt"].apply(lambda w: wkt.loads(w) if pd.notna(w) else None)
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=CONFIG["crs_geo"])
    gdf = gdf.drop(columns=["wkt"])
    return gdf


# ============================================================
# ОЧИСТКА
# ============================================================

def fix_geometry(gdf, label=""):
    """Исправление невалидных геометрий через make_valid."""
    invalid_mask = ~gdf.geometry.is_valid
    n_invalid = invalid_mask.sum()
    if n_invalid > 0:
        print(f"  [{label}] Невалидных геометрий: {n_invalid} -> make_valid()")
        gdf.loc[invalid_mask, "geometry"] = gdf.loc[invalid_mask, "geometry"].apply(make_valid)
    return gdf


def explode_multipolygons(gdf, label=""):
    """Разворачивание MULTIPOLYGON в отдельные строки."""
    n_multi = gdf.geometry.geom_type.eq("MultiPolygon").sum()
    if n_multi > 0:
        print(f"  [{label}] MULTIPOLYGON: {n_multi} -> explode")
        gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    return gdf


def remove_empty_geometry(gdf, label=""):
    """Удаление записей с пустой или null геометрией."""
    mask = gdf.geometry.isna() | gdf.geometry.is_empty
    n_removed = mask.sum()
    if n_removed > 0:
        print(f"  [{label}] Пустые/null геометрии: {n_removed} -> удалены")
        gdf = gdf[~mask].reset_index(drop=True)
    return gdf


def remove_zero_area(gdf, label=""):
    """Удаление полигонов с нулевой площадью."""
    mask = gdf.geometry.area == 0
    n_removed = mask.sum()
    if n_removed > 0:
        print(f"  [{label}] Zero-area полигоны: {n_removed} -> удалены")
        gdf = gdf[~mask].reset_index(drop=True)
    return gdf


def clean_source_a(gdf):
    """Полный pipeline очистки Источника А."""
    print("=== Очистка Источника А ===")
    n0 = len(gdf)

    gdf = remove_empty_geometry(gdf, "А")
    gdf = fix_geometry(gdf, "А")
    gdf = explode_multipolygons(gdf, "А")
    gdf = remove_zero_area(gdf, "А")

    # Фильтрация мелких объектов
    gdf_proj = gdf.to_crs(CONFIG["crs_projected"])
    area_m2 = gdf_proj.geometry.area
    small_mask = area_m2 < CONFIG["min_area_a"]
    n_small = small_mask.sum()
    print(f"  [А] Площадь < {CONFIG['min_area_a']} м²: {n_small} -> удалены")
    gdf = gdf[~small_mask].reset_index(drop=True)

    # Пересчитываем площадь в метрах
    gdf_proj = gdf.to_crs(CONFIG["crs_projected"])
    gdf["area_m2"] = gdf_proj.geometry.area

    # Парсинг тегов
    gdf["tag_main"] = (
        gdf["tags"]
        .astype(str)
        .str.strip("[]")
        .str.replace("'", "")
        .str.split(",")
        .str[0]
        .str.strip()
    )

    # Кластеризация отключена — ухудшает IoU для матчинга,
    # т.к. объединённые полигоны А становятся слишком большими.
    # Граф связности в 02_matching уже обрабатывает m:n отношения.

    print(f"  [А] Итого: {n0} -> {len(gdf)} ({n0 - len(gdf)} удалено)")
    return gdf


def clean_source_b(gdf):
    """Полный pipeline очистки Источника Б."""
    print("=== Очистка Источника Б ===")
    n0 = len(gdf)

    gdf = remove_empty_geometry(gdf, "Б")
    gdf = fix_geometry(gdf, "Б")
    gdf = explode_multipolygons(gdf, "Б")
    gdf = remove_zero_area(gdf, "Б")

    # Фильтрация за пределами СПб
    bbox = CONFIG["spb_bbox"]
    centroids = gdf.geometry.centroid
    in_spb = (
        (centroids.x >= bbox["lon_min"])
        & (centroids.x <= bbox["lon_max"])
        & (centroids.y >= bbox["lat_min"])
        & (centroids.y <= bbox["lat_max"])
    )
    n_outside = (~in_spb).sum()
    print(f"  [Б] За пределами bbox СПб: {n_outside} -> удалены")
    gdf = gdf[in_spb].reset_index(drop=True)

    # Восстановление пропущенных высот
    missing_h = gdf["height"].isna() | (gdf["height"] == 0)
    can_restore = missing_h & gdf["stairs"].notna() & gdf["avg_floor_height"].notna()
    n_restored = can_restore.sum()
    if n_restored > 0:
        gdf.loc[can_restore, "height"] = (
            gdf.loc[can_restore, "stairs"] * gdf.loc[can_restore, "avg_floor_height"]
        )
        print(f"  [Б] Высота восстановлена из stairs*avg: {n_restored}")

    # Аномалии высоты
    anomaly_mask = gdf["height"] > CONFIG["max_height_b"]
    n_anomaly = anomaly_mask.sum()
    if n_anomaly > 0:
        print(f"  [Б] Высота > {CONFIG['max_height_b']} м: {n_anomaly} -> удалены")
        gdf = gdf[~anomaly_mask].reset_index(drop=True)

    # Пересчитываем площадь
    gdf_proj = gdf.to_crs(CONFIG["crs_projected"])
    gdf["area_m2"] = gdf_proj.geometry.area

    print(f"  [Б] Итого: {n0} -> {len(gdf)} ({n0 - len(gdf)} удалено)")
    return gdf


# ============================================================
# КЛАСТЕРИЗАЦИЯ ВНУТРИ ИСТОЧНИКА
# ============================================================

def cluster_touching_polygons(gdf, buffer_m=1.0, label=""):
    """Объединяет касающиеся/близкие полигоны в один.

    1. Буферизует полигоны на buffer_m метров
    2. Находит пересекающиеся буферы через sjoin
    3. Строит граф касаний → connected components
    4. Объединяет каждую компоненту через unary_union
    """
    import networkx as nx

    n0 = len(gdf)
    gdf_proj = gdf.to_crs(CONFIG["crs_projected"]).copy()
    gdf_proj["_orig_idx"] = range(len(gdf_proj))

    # Буферизуем для поиска касающихся
    buffered = gdf_proj.copy()
    buffered["geometry"] = gdf_proj.geometry.buffer(buffer_m)

    # Находим пересечения буферов (= касающиеся/близкие полигоны)
    pairs = gpd.sjoin(buffered, buffered, how="inner", predicate="intersects")
    pairs = pairs[pairs.index != pairs["index_right"]]  # убираем self-joins

    # Граф касаний
    G = nx.Graph()
    G.add_nodes_from(range(len(gdf_proj)))
    for idx, row in pairs.iterrows():
        G.add_edge(idx, row["index_right"])

    components = list(nx.connected_components(G))
    n_multi = sum(1 for c in components if len(c) > 1)

    # Объединяем каждую компоненту
    records = []
    for comp in components:
        comp = list(comp)
        if len(comp) == 1:
            # Одиночный полигон — просто копируем
            records.append(gdf.iloc[comp[0]].to_dict())
        else:
            # Объединяем геометрию
            geoms = [gdf.geometry.iloc[i] for i in comp]
            merged_geom = unary_union(geoms)

            # Атрибуты от наибольшего полигона
            areas = [gdf_proj.geometry.iloc[i].area for i in comp]
            best = comp[np.argmax(areas)]
            rec = gdf.iloc[best].to_dict()
            rec["geometry"] = merged_geom
            records.append(rec)

    result = gpd.GeoDataFrame(records, crs=gdf.crs)

    # Пересчитываем площадь
    result_proj = result.to_crs(CONFIG["crs_projected"])
    result["area_m2"] = result_proj.geometry.area

    print(f"  [{label}] Кластеризация: {n0} -> {len(result)} ({n_multi} групп объединено)")
    return result.reset_index(drop=True)


# ============================================================
# МАТЧИНГ
# ============================================================

def compute_iou(geom_a, geom_b):
    """IoU (Intersection over Union) для двух полигонов."""
    try:
        if geom_a.is_empty or geom_b.is_empty:
            return 0.0
        intersection = geom_a.intersection(geom_b).area
        if intersection == 0:
            return 0.0
        union = geom_a.area + geom_b.area - intersection
        return intersection / union if union > 0 else 0.0
    except Exception:
        return 0.0


def compute_overlap_ratio(geom_a, geom_b):
    """Доля пересечения относительно меньшего полигона."""
    try:
        if geom_a.is_empty or geom_b.is_empty:
            return 0.0
        intersection = geom_a.intersection(geom_b).area
        min_area = min(geom_a.area, geom_b.area)
        return intersection / min_area if min_area > 0 else 0.0
    except Exception:
        return 0.0


# ============================================================
# СОХРАНЕНИЕ / ЗАГРУЗКА
# ============================================================

def save_gdf(gdf, path):
    """Сохранение GeoDataFrame в parquet."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Приводим object-колонки со смешанными типами к строкам
    gdf = gdf.copy()
    for col in gdf.select_dtypes(include=["object"]).columns:
        gdf[col] = gdf[col].astype(str).replace("nan", None)
    gdf.to_parquet(path)
    print(f"  Сохранено: {path} ({len(gdf)} записей)")


def load_gdf(path):
    """Загрузка GeoDataFrame из parquet."""
    return gpd.read_parquet(path)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

from scipy.spatial import cKDTree

def extract_geometric_features(gdf_proj):
    """Extract geometric features from projected GeoDataFrame.

    Returns DataFrame with columns:
    - area, perimeter, n_vertices
    - compactness, complexity_ratio
    - convex_area, convexity, rectangularity
    """
    geom = gdf_proj.geometry

    area = geom.area
    perimeter = geom.length
    n_vertices = geom.apply(
        lambda g: len(g.exterior.coords) if g.geom_type == 'Polygon'
        else sum(len(p.exterior.coords) for p in g.geoms)
    )

    convex = geom.convex_hull
    convex_area = convex.area

    mbr = geom.apply(
        lambda g: g.minimum_rotated_rectangle
        if hasattr(g, 'minimum_rotated_rectangle') else g.envelope
    )
    mbr_area = mbr.apply(lambda g: g.area)

    features = pd.DataFrame({
        'area': area,
        'perimeter': perimeter,
        'n_vertices': n_vertices,
        'compactness': (4 * np.pi * area) / (perimeter ** 2 + 1e-10),
        'complexity_ratio': perimeter / (np.sqrt(area) + 1e-10),
        'convex_area': convex_area,
        'convexity': area / (convex_area + 1e-10),
        'rectangularity': area / (mbr_area + 1e-10),
    }, index=gdf_proj.index)

    return features


def extract_contextual_features(gdf_proj, radii=(50, 100, 250, 500, 1000)):
    """Extract contextual features: neighbor stats at multiple radii."""
    centroids = np.column_stack([gdf_proj.geometry.centroid.x, gdf_proj.geometry.centroid.y])
    areas = gdf_proj.geometry.area.values
    tree = cKDTree(centroids)

    all_features = {}

    for r in radii:
        counts = np.zeros(len(gdf_proj))
        mean_areas = np.zeros(len(gdf_proj))
        max_areas = np.zeros(len(gdf_proj))
        total_areas = np.zeros(len(gdf_proj))

        neighbors = tree.query_ball_tree(tree, r)

        for i, nbrs in enumerate(neighbors):
            nbrs = [j for j in nbrs if j != i]
            if len(nbrs) > 0:
                nbr_areas = areas[nbrs]
                counts[i] = len(nbrs)
                mean_areas[i] = nbr_areas.mean()
                max_areas[i] = nbr_areas.max()
                total_areas[i] = nbr_areas.sum()

        all_features[f'n_buildings_{r}m'] = counts
        all_features[f'mean_area_{r}m'] = mean_areas
        all_features[f'max_area_{r}m'] = max_areas
        all_features[f'density_{r}m'] = total_areas / (np.pi * r ** 2)

    return pd.DataFrame(all_features, index=gdf_proj.index)


def extract_spatial_lag_features(gdf_proj, target_col, k_values=(5, 10, 20, 50)):
    """Spatial lag features: weighted average of target for k nearest neighbors."""
    centroids = np.column_stack([gdf_proj.geometry.centroid.x, gdf_proj.geometry.centroid.y])
    target = gdf_proj[target_col].values
    tree = cKDTree(centroids)

    features = {}

    for k in k_values:
        lag = np.full(len(gdf_proj), np.nan)
        distances, indices = tree.query(centroids, k=k+1)

        for i in range(len(gdf_proj)):
            nbr_idx = indices[i, 1:]
            nbr_vals = target[nbr_idx]
            valid = ~np.isnan(nbr_vals)
            if valid.any():
                nbr_dists = distances[i, 1:][valid]
                nbr_vals_valid = nbr_vals[valid]
                weights = 1.0 / (nbr_dists + 1e-10)
                lag[i] = np.average(nbr_vals_valid, weights=weights)

        features[f'height_lag_k{k}'] = lag

    return pd.DataFrame(features, index=gdf_proj.index)


# ============================================================
# METRICS
# ============================================================

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def compute_height_metrics(y_true, y_pred):
    """Compute all height prediction metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    errors = np.abs(y_true - y_pred)
    pct_under_3m = (errors < 3).mean() * 100
    pct_under_5m = (errors < 5).mean() * 100
    p95_error = np.percentile(errors, 95)

    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'pct_under_3m': pct_under_3m,
        'pct_under_5m': pct_under_5m,
        'p95_error': p95_error,
        'n_samples': len(y_true),
    }
