"""
Экспорт обогащённых GeoJSON для web-карты.
Добавляет zone_type и рекомендации по БС к каждому зданию.
"""

import json
import numpy as np
import pandas as pd
import geopandas as gpd


# === Zone classification ===

def classify_zones(height, h_max_k10, height_lag_k10, density_100m):
    """Classify RF zone for each building based on neighborhood context."""
    conditions = [
        (h_max_k10 > 25) | (height > 25),           # highrise
        density_100m < 0.05,                          # rural
        (density_100m < 0.15) & (height_lag_k10 < 10),  # suburban
    ]
    choices = ['highrise', 'urban', 'suburban', 'rural']
    # np.select: first match wins; default = 'urban'
    # reorder: highrise first, then rural, then suburban, rest = urban
    return np.select(conditions, ['highrise', 'rural', 'suburban'], default='urban')


def compute_bs_recommendations(zone, height):
    """Compute BS placement heights per zone type."""
    macro_h = np.where(
        zone == 'highrise', np.clip(height + 3, 30, 50),
        np.where(zone == 'urban', np.clip(height + 8, 20, 25),
        np.where(zone == 'suburban', np.clip(height + 10, 15, 25),
        np.clip(height + 20, 30, 60))))

    small_cell_h = np.where(
        zone == 'highrise', np.clip(height * 0.6, 16, 24),
        np.where(zone == 'urban', np.clip(height * 0.5, 10, 13),
        np.where(zone == 'suburban', np.clip(height * 0.8, 10, 12),
        np.clip(height + 5, 15, 20))))

    pico_h = np.where(
        zone == 'highrise', np.clip(height * 0.5, 18, 22),
        np.where(zone == 'urban', np.clip(height * 0.7, 12, 15),
        np.full_like(height, np.nan)))

    pico_needed = np.isin(zone, ['highrise', 'urban'])

    return (
        np.round(macro_h, 1),
        np.round(small_cell_h, 1),
        np.round(pico_h, 1),
        pico_needed,
    )


def main():
    print("Loading data...")
    gdf = gpd.read_parquet('data/final.parquet')
    feat = pd.read_parquet('data/features.parquet',
                           columns=['height_lag_k10', 'h_max_k10', 'density_100m'])

    # Row-aligned — assign directly
    gdf['h_max_k10'] = feat['h_max_k10'].values
    gdf['height_lag_k10'] = feat['height_lag_k10'].values
    gdf['density_100m'] = feat['density_100m'].values

    print("Classifying zones...")
    gdf['zone_type'] = classify_zones(
        gdf['height'].values,
        gdf['h_max_k10'].values,
        gdf['height_lag_k10'].values,
        gdf['density_100m'].values,
    )

    macro_h, sc_h, pico_h, pico_needed = compute_bs_recommendations(
        gdf['zone_type'].values, gdf['height'].values)
    gdf['macro_h'] = macro_h
    gdf['small_cell_h'] = sc_h
    gdf['pico_h'] = pico_h
    gdf['pico_needed'] = pico_needed

    # Zone distribution
    print("\nZone distribution:")
    print(gdf['zone_type'].value_counts())

    # Load districts
    with open('web/data/districts.json') as f:
        districts = json.load(f)

    # Export properties
    export_cols = [
        'geometry', 'height', 'match_type', 'height_source',
        'tag_main', 'area_m2',
        'zone_type', 'macro_h', 'small_cell_h', 'pico_h', 'pico_needed',
    ]

    # Compute centroids once
    centroids = gdf.geometry.centroid

    for d in districts:
        lat, lon, r = d['lat'], d['lon'], d['r']
        mask = (
            (centroids.x >= lon - r) & (centroids.x <= lon + r) &
            (centroids.y >= lat - r) & (centroids.y <= lat + r)
        )
        subset = gdf.loc[mask, export_cols].copy()

        # Round numeric props
        subset['height'] = subset['height'].round(1)
        subset['area_m2'] = subset['area_m2'].round(0).astype(int)
        # Convert pico_h NaN to None for JSON null
        subset['pico_h'] = subset['pico_h'].where(subset['pico_needed'], other=None)
        # Boolean to Python bool
        subset['pico_needed'] = subset['pico_needed'].astype(bool)

        out_path = f"web/data/{d['id']}.geojson"
        subset.to_file(out_path, driver='GeoJSON', coordinate_precision=6)
        print(f"  {d['id']}: {len(subset)} buildings -> {out_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
