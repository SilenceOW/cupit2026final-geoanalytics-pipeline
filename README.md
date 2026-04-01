# Выше крыши: Единая модель высотности зданий Санкт-Петербурга

**Changellenge Cup IT 2026 | Кейс МТС | Финал | 3 место**

Построение единой модели высотности 196,807 зданий Санкт-Петербурга из двух разнородных геоисточников для планирования телекоммуникационной сети МТС.

> [Презентация решения (PDF)](MTS%20Solution.pdf)

---

## Команда DDBuilder

| Участник | Роль | Вуз / Программа |
|----------|------|-----------------|
| **Александр Корнилович** | Product Manager, Business Analyst | МФТИ & МШУ Сколково, «Управление инновациями в бизнесе», 2 курс |
| **Андрей Колесников** | Data Scientist, ML Engineer | ФПМИ МФТИ, «Анализ данных в экономике» (совместно с РЭШ), 2 курс |
| **Арина Рыбакова** | Presentation Designer, Product Manager | МШУ Сколково, «Управление и предпринимательство», 1 курс |
| **Герман Иванов** | Data Engineer, Data Analyst | ВШЭ, «Экономика и анализ данных», 1 курс |
| **Игорь Карташов** | Data Scientist, ML Engineer | ИТМО, «Инженерия искусственного интеллекта», 2 курс |

---

## О кейсе

МТС — крупнейший оператор связи в России. Для планирования сети (выбор мест для базовых станций, расчёт зон покрытия) необходима точная информация о высоте зданий и их геометрии. Проблема: ни один открытый источник не даёт полной и достоверной картины — данные содержат ошибки, пропуски и противоречат друг другу.

**Задача:** объединить два набора геоданных по СПб (Источник А — 171,454 здания, Источник Б — 161,076 зданий), очистить их, сопоставить объекты между источниками и построить единую модель высотности, включая предсказание высоты для зданий, где она неизвестна.

**Подзадачи:**
1. Анализ качества данных и очистка
2. Сопоставление зданий из разных источников (entity resolution)
3. Алгоритм выбора наиболее достоверной высоты
4. ML-модель предсказания высоты для зданий без данных
5. Валидация результатов
6. Визуализация и инсайты для RF-планирования

---

## Наше решение

### Ключевые результаты

| Метрика | Значение |
|---------|----------|
| Всего зданий в unified dataset | **196,807** |
| С высотой из Источника Б | 160,750 (82%) |
| ML-предсказанных | 36,057 (18%) |
| MAE на hold-out (20%) | **2.21 м** |
| R² на hold-out | **0.890** |
| % ошибок < 3 м | 78% |
| % ошибок < 5 м | 88% |
| OSM cross-check (30,737 зданий) | MAE 3.13 м |
| GKH cross-check (24,599 зданий) | MAE 3.66 м |
| Адресная валидация | 88.3% совпадение |

### Архитектура подхода

```
Источник А (171K)  ──┐                                      ┌── Валидация (GKH + OSM + адресная)
                     ├── Очистка ── Matching (граф) ── ML ──┤
Источник Б (161K)  ──┘     ↓            ↓            ↓      └── Web-визуализация + Shapefile
                      make_valid    2 прохода:     XGBoost
                      explode       IoU+overlap    85 фичей
                      bbox filter   centroid       log1p target
                      8 м² порог    NetworkX       5 пространственных масштабов
```

### Что отличает наше решение

- **Двухпроходный matching** — IoU + overlap ratio (для пересекающихся) + centroid proximity (для сдвинутых до 20 м) + постобработка доматчивания соседей
- **85 признаков** на 5 пространственных масштабах (50/100/250/500/1000 м), spatial lag IDW, tag-specific lag, size-aware features
- **Обнаружение и удаление target leak** — `stairs` (63% importance) и `gkh_floor_max` были явно исключены из модели как прямые прокси целевой переменной
- **4 независимых валидации** — hold-out, GKH (Реформа ЖКХ), OSM (OpenStreetMap, 30K зданий), адресная проверка
- **30+ задокументированных экспериментов** в `experiments/`
- **Web-приложение** с режимом RF-планирования (рекомендации по типу сот)
- **Shapefile** для импорта в Atoll / Planet EV

---

## Pipeline

### 01_cleaning.ipynb — Очистка данных
- Исправление невалидных геометрий (`make_valid`, explode MultiPolygon)
- Фильтрация: площадь < 8 м², здания за пределами СПб (bbox)
- Восстановление пропущенных высот через `stairs * avg_floor_height`
- **Вход:** `cup_it_example_src_A.csv`, `cup_it_example_src_B.csv`
- **Выход:** `data/cleaned_a.parquet`, `data/cleaned_b.parquet`

### 02_matching.ipynb — Сопоставление зданий
- **Проход 1:** R-tree spatial index -> IoU >= 0.1 OR overlap >= 0.3
- **Проход 2:** Centroid proximity < 20 м, area ratio >= 0.3
- Граф связности (NetworkX) -> connected components -> разрезка компонент > 20 узлов
- Приоритет геометрии Источника Б (высота привязана к полигону)
- Постобработка: доматчивание only_B соседей через 2 м buffer
- **Выход:** `data/unified.parquet` (196,807 зданий, 82% с высотой)

### 03_height_model.ipynb — Предсказание высот
- 85 признаков: геометрические (8), контекстуальные (20), spatial lag (4), категориальные + size-aware (49), target encoding (3), match type (1)
- XGBoost (800 деревьев, depth=10, log1p target)
- Предсказано 36,057 высот, диапазон [2.8, 92.6] м
- **Выход:** `data/final.parquet`, `data/final.csv`, `data/shp/buildings_spb.shp`, `data/xgb_model.pkl`

### 04_validation.ipynb — Валидация
- GKH cross-check: наша высота vs этажность ЖКХ * 3 м (24,599 зданий)
- OSM cross-check: наша высота vs `building:levels` * 3 м (30,737 зданий)
- Адресная валидация: 88.3% совпадение улиц между источниками
- Sanity checks: 100% покрытие, диапазон [2, 462] м, 20 зданий > 100 м

### 05_visualization.ipynb — Визуализация
- Интерактивные карты высот по районам (folium)
- Тепловая карта высотной застройки (> 20 м)
- Карты matching inspection для 4 районов
- Статистика по районам, инсайты для RF-планирования

---

## Quick Start

```bash
# 1. Клонировать репозиторий
git clone https://github.com/SilenceOW/cupit2026final-geoanalytics-pipeline.git
cd cupit2026final-geoanalytics-pipeline

# 2. Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 3. Установить зависимости
pip install -r requirements.txt
brew install libomp  # macOS only, для XGBoost

# 4. Положить исходные данные в корень проекта
#    cup_it_example_src_A.csv
#    cup_it_example_src_B.csv

# 5. Запустить pipeline последовательно
jupyter notebook
# Открыть и запустить (Restart & Run All) в порядке:
#   01_cleaning.ipynb
#   02_matching.ipynb
#   03_height_model.ipynb
#   04_validation.ipynb
#   05_visualization.ipynb
```

---

## Выходные форматы

| Файл | Формат | Описание |
|------|--------|----------|
| `data/final.parquet` | GeoParquet | Основной датасет |
| `data/final.csv` | CSV + WKT | Универсальный формат |
| `data/shp/buildings_spb.shp` | Shapefile | Для RF-планировщиков (Atoll, Planet EV) |
| `web/index.html` | HTML/JS | Интерактивная карта с режимом RF-планирования |

---

## Структура проекта

```
.
├── README.md                    # Описание проекта
├── MTS Solution.pdf             # Презентация решения
├── requirements.txt             # Зависимости (14 пакетов)
├── utils.py                     # Общие функции и CONFIG
│
├── 01_cleaning.ipynb            # Очистка данных
├── 02_matching.ipynb            # Сопоставление зданий
├── 03_height_model.ipynb        # Модель высот (XGBoost, 85 фичей)
├── 04_validation.ipynb          # Валидация (GKH + OSM + адресная)
├── 05_visualization.ipynb       # Визуализация и инсайты
├── eda_analysis.ipynb           # Разведочный анализ данных
│
├── export_rf_geojson.py         # Экспорт данных для web-карты
├── insights_analysis.py         # Генерация графиков для презентации
└── web/                         # Интерактивная карта (Leaflet.js)
    ├── index.html
    └── data/                    # GeoJSON по районам + RF grid

```

---

## Ключевые параметры

Все параметры настраиваются в ячейке `КОНФИГ` каждого ноутбука и в `CONFIG` dict в `utils.py`.

| Параметр | Значение | Обоснование |
|----------|----------|-------------|
| `min_area_a` | 8 м² | Сохраняет реальные трансформаторные будки (~6-9 м²) |
| `iou_threshold` | 0.1 | Протестировано 0.01-0.5 на 4 районах |
| `overlap_threshold` | 0.3 | Ловит маленький Б внутри большого А |
| `centroid_max_dist` | 20 м | Измеренный макс. сдвиг между источниками |
| `max_component_size` | 20 | При < 12 ломает реальные здания |
| XGBoost | 800 trees, depth=10, lr=0.03 | 30+ экспериментов |
| Target transform | log1p(height) | Снижает OSM MAE на 21% |

---

## References

1. Nature 2024 — Inferring Building Height from Footprint Morphology
2. ESA 2024 — Many-to-Many Polygon Matching a la Jaccard
3. SIGMOD 2025 — 3dSAGER: Geospatial Entity Resolution
4. MDPI 2022 — Spatial Autocorrelation in ML Models
5. MDPI 2024 — Building Height Extraction via Spatial Clustering + Random Forest
