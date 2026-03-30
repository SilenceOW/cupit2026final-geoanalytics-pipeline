**Changellenge Cup IT 2026 Final | Единая модель высотности зданий**

Построение единой модели высотности зданий Санкт-Петербурга для планирования телекоммуникационной сети.

## Quick Start

```bash
# 1. Клонировать репозиторий
git clone <repo-url>
cd "CupIT Final"

# 2. Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 3. Установить зависимости
pip install -r requirements.txt
brew install libomp  # macOS only, для XGBoost

# 4. Запустить pipeline последовательно
jupyter notebook
# Открыть и запустить (Restart & Run All) в порядке:
#   01_cleaning.ipynb
#   02_matching.ipynb
#   03_height_model.ipynb
#   04_validation.ipynb
#   05_visualization.ipynb
```

## Pipeline

### 01_cleaning.ipynb — Очистка данных
- Исправление невалидных геометрий (`make_valid`, explode MULTIPOLYGON)
- Фильтрация: площадь < 8 м$^2$ (мусор оцифровки), здания за пределами СПб
- Восстановление пропущенных высот через `stairs * avg_floor_height`
- **Вход:** `cup_it_example_src_A.csv`, `cup_it_example_src_B.csv`
- **Выход:** `data/cleaned_a.parquet`, `data/cleaned_b.parquet`

### 02_matching.ipynb — Сопоставление зданий
Graph-based entity resolution для геообъектов:
- **Проход 1:** R-tree spatial index → IoU + overlap ratio (для пересекающихся полигонов)
- **Проход 2:** Centroid proximity matching (для сдвинутых полигонов, < 20м)
- Граф связности (NetworkX) → Connected components → Разрезка больших компонент (> 20 узлов)
- Приоритет Источника Б (высота привязана к геометрии)
- Постобработка: доматчивание only_B соседей, удаление дублей
- **Выход:** `data/unified.parquet` (~197K зданий, 82% с высотой)

### 03_height_model.ipynb — Предсказание высот
- 73 признака: геометрические (8), контекстуальные (20 на 5 масштабах), spatial lag (4), категориальные + пространственные (37+), target encoding (3)
- Ensemble: 0.7 * XGBoost + 0.3 * RandomForest
- Метрики (full features): MAE 0.33м, RMSE 2.2м, R$^2$ 0.98
- Метрики (only_A, реальный use case): MAE ~3.0м, 70% зданий < 3м ошибки
- **Выход:** `data/final.parquet`, `data/final.csv`, `data/shp/buildings_spb.shp`

### 04_validation.ipynb — Валидация
- GKH cross-check: MAE 3.62м (наша высота vs этажность ЖКХ * 3м)
- Адресная валидация: 88% совпадение улиц между источниками
- Sanity checks: 100% покрытие, разумный диапазон высот [2, 462]м

### 05_visualization.ipynb — Визуализация
- Интерактивная карта высот (folium) с реальными полигонами зданий
- Тепловая карта высотной застройки (> 20м)
- Статистика по районам, инсайты для RF-планирования

## Output Formats

| Файл | Формат | Описание |
|------|--------|----------|
| `data/final.parquet` | GeoParquet | Основной датасет, быстрая загрузка в Python |
| `data/final.csv` | CSV (WKT) | Универсальный формат, геометрия в колонке `wkt` |
| `data/shp/buildings_spb.shp` | Shapefile | Для импорта в RF-планировщики (Atoll, Planet EV) |

## Key Parameters

Все параметры настраиваются в ячейке КОНФИГ в начале каждого notebook.

| Параметр | Значение | Обоснование |
|----------|----------|-------------|
| `min_area_a` | 8 м$^2$ | Индустрия: 20-40 м$^2$. Консервативный порог |
| `iou_threshold` | 0.1 | Протестировано 0.01-0.5 на 4 районах |
| `overlap_threshold` | 0.3 | Ловит маленький Б внутри большого А |
| `centroid_max_dist` | 20 м | Сдвиг между источниками до 20м |
| `max_component_size` | 20 | При < 12 ломает реальные здания |
| Ensemble weights | 0.7 XGB / 0.3 RF | Протестировано 5 комбинаций |

## Project Structure

```
.
├── README.md                  # Этот файл
├── requirements.txt           # Зависимости
├── utils.py                   # Общие функции
├── 01_cleaning.ipynb          # Очистка данных
├── 02_matching.ipynb          # Сопоставление зданий
├── 03_height_model.ipynb      # Модель высот
├── 04_validation.ipynb        # Валидация
├── 05_visualization.ipynb     # Визуализация
├── eda_analysis.ipynb         # Разведочный анализ
├── pipeline_report.pdf        # Отчёт по pipeline
├── case_docs/                 # Описание кейса и статьи
├── data/                      # Промежуточные данные (генерируются pipeline)
│   ├── cleaned_a.parquet
│   ├── cleaned_b.parquet
│   ├── unified.parquet
│   ├── final.parquet
│   ├── final.csv
│   ├── shp/buildings_spb.shp
│   ├── xgb_model.pkl
│   └── rf_model.pkl
└── cup_it_example_src_A.csv   # Исходные данные (не в репо)
    cup_it_example_src_B.csv
```

## References

1. Nature 2024 — Inferring Building Height from Footprint Morphology (74.5M buildings, XGBoost MAE 0.91m)
2. ESA 2024 — Many-to-Many Polygon Matching a la Jaccard (ILP for building matching)
3. SIGMOD 2025 — 3dSAGER: Geospatial Entity Resolution (F1 92.8%)
4. MDPI 2022 — Spatial Autocorrelation in ML Models (spatial lag features)
5. MDPI 2024 — Building Height Extraction via Spatial Clustering + Random Forest
