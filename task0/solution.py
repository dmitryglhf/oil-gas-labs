import marimo

__generated_with = "0.19.4"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path

    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", 500)
    pd.options.display.max_colwidth = 250
    pd.set_option("display.float_format", lambda x: "%.2f" % x)

    import numpy as np

    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    from scipy.spatial import Voronoi
    from scipy.spatial.distance import squareform

    from tslearn.metrics import dtw
    from tslearn.clustering import TimeSeriesKMeans

    from tqdm.autonotebook import tqdm

    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    DATA_DIR = Path(__file__).parent / "COSTA MODEL"
    return (
        DATA_DIR,
        StandardScaler,
        TimeSeriesKMeans,
        Voronoi,
        dendrogram,
        dtw,
        linkage,
        matplotlib,
        np,
        pd,
        pdist,
        plt,
        silhouette_score,
        sns,
        squareform,
        tqdm,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Лабораторная работа: Кластеризация профилей добычи COSTA

    **Задачи:**
    1. Прокластеризовать временные ряды датасета COSTA используя DTW
    2. Определить оптимальное количество кластеров
    3. Визуализировать кластеры на карте
    4. Проанализировать пространственное распределение кластеров
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Загрузка данных

    Используем данные из листа **Appraisal Wells** - это разведочные скважины (HW-*),
    для которых есть координаты в файле Well Heads.
    """)
    return


@app.cell
def _(DATA_DIR, pd):
    # Загрузка данных добычи из Excel
    _path = DATA_DIR / "5 - Production" / "COSTA Synthetic Production Monthly.xlsx"
    _df_raw = pd.read_excel(_path, sheet_name='Appraisal Wells', header=3)

    # Переименовываем колонки
    _df_raw.columns = ['well', 'date', 'gor', 'gas_rate', 'oil_rate', 'watercut', 'water_rate', 'bhp']

    # Убираем строку с заголовками (первая строка содержит названия)
    df = _df_raw[_df_raw['well'] != 'Well'].copy()

    # Преобразуем типы данных
    df['date'] = pd.to_datetime(df['date'])
    df['oil_rate'] = pd.to_numeric(df['oil_rate'])
    df['water_rate'] = pd.to_numeric(df['water_rate'])
    df['gas_rate'] = pd.to_numeric(df['gas_rate'])
    df['gor'] = pd.to_numeric(df['gor'])
    df['watercut'] = pd.to_numeric(df['watercut'])
    df['bhp'] = pd.to_numeric(df['bhp'])

    print(f"Загружено записей: {len(df)}")
    print(f"Количество скважин: {len(df['well'].unique())}")
    print(f"Скважины: {sorted(df['well'].unique(), key=lambda x: int(x.split('-')[1]))}")
    df.head()
    return (df,)


@app.cell
def _(DATA_DIR, pd):
    # Загрузка координат скважин
    _path_coords = DATA_DIR / "4 - Building 3D Geo-Model" / "1 - Well Heads" / "COSTA Well Heads.prn"
    coords = pd.read_csv(_path_coords, sep=r'\s+', skiprows=1,
                         names=['well', 'x', 'y', 'kb', 'td'])
    print(f"Загружено координат для {len(coords)} скважин")
    coords.head()
    return (coords,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Визуализация профилей добычи
    """)
    return


@app.cell
def _(df, plt):
    # Визуализация профилей добычи нефти (Oil Rate)
    column = 'oil_rate'
    plt.figure(figsize=(15, 5))
    for _well in df['well'].unique():
        _well_data = df[df['well'] == _well]
        plt.plot(_well_data['date'], _well_data[column], label=_well)

    plt.legend(loc='upper center', frameon=True, bbox_to_anchor=(0.5, -0.15), ncol=6)
    plt.grid(ls='--')
    plt.xlabel('Date')
    plt.ylabel('Oil Rate (bbl/day)')
    plt.title('Профили добычи нефти по скважинам COSTA')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    return (column,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Подготовка данных для кластеризации

    Создаём pivot-таблицу и нормализуем данные
    """)
    return


@app.cell
def _(column, df):
    # Создаём pivot-таблицу: строки - даты, столбцы - скважины
    data = df.pivot_table(index='date', columns='well', values=column).bfill()
    print(f"Размер данных: {data.shape}")
    data.head()
    return (data,)


@app.cell
def _(StandardScaler, data):
    # Нормализация данных
    _scaler = StandardScaler()
    data_scaled = data.copy()
    _scaler.fit(data_scaled)
    data_scaled[data_scaled.columns] = _scaler.transform(data_scaled)
    data_scaled.head()
    return (data_scaled,)


@app.cell
def _(data_scaled, plt):
    # Визуализация нормализованных профилей
    plt.figure(figsize=(15, 5))
    for _col in data_scaled.columns:
        plt.plot(data_scaled.index, data_scaled[_col], label=_col)

    plt.legend(loc='upper center', frameon=True, bbox_to_anchor=(0.5, -0.2), ncol=6)
    plt.grid(ls='--')
    plt.xlabel('Date')
    plt.ylabel('Normalized Oil Rate')
    plt.title('Нормализованные профили добычи')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Определение оптимального числа кластеров

    Используем метод локтя (Elbow Method) и коэффициент силуэта (Silhouette Score)
    """)
    return


@app.cell
def _(TimeSeriesKMeans, data, data_scaled, plt, silhouette_score, tqdm):
    # Определение оптимального числа кластеров
    _distortions = []
    _silhouette_scores = []
    _n_range = range(2, min(10, len(data.columns)))

    for _k in tqdm(_n_range, desc="Подбор числа кластеров"):
        _model = TimeSeriesKMeans(
            n_clusters=_k,
            metric="dtw",
            n_jobs=-1,
            max_iter=10,
            random_state=42,
        )
        _model.fit(data.T)
        _distortions.append(_model.inertia_)
        _silhouette_scores.append(silhouette_score(data_scaled.T, _model.labels_))

    # Графики
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 4))

    _axes[0].plot(list(_n_range), _distortions, "bx-")
    _axes[0].set_xlabel("Количество кластеров (k)")
    _axes[0].set_ylabel("Distortion (Inertia)")
    _axes[0].set_title("Метод локтя (Elbow Method)")
    _axes[0].grid(True)

    _axes[1].plot(list(_n_range), _silhouette_scores, "rx-")
    _axes[1].set_xlabel("Количество кластеров (k)")
    _axes[1].set_ylabel("Silhouette Score")
    _axes[1].set_title("Коэффициент силуэта")
    _axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    # Выводим лучшее значение силуэта
    best_k = list(_n_range)[_silhouette_scores.index(max(_silhouette_scores))]
    print(f"\nОптимальное число кластеров по силуэту: {best_k}")
    print(f"Максимальный Silhouette Score: {max(_silhouette_scores):.3f}")
    return (best_k,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Кластеризация с оптимальным числом кластеров

    **Обоснование выбора метода:**
    - DTW (Dynamic Time Warping) используется, так как он учитывает временные сдвиги в профилях добычи
    - Скважины могут иметь схожую динамику, но с разным временем запуска
    - Silhouette Score выбран как метрика, так как он оценивает компактность и разделимость кластеров
    """)
    return


@app.cell
def _(TimeSeriesKMeans, best_k, data):
    # Финальная кластеризация
    n_clusters = best_k
    final_model = TimeSeriesKMeans(
        n_clusters=n_clusters,
        metric='dtw',
        n_jobs=-1,
        max_iter=100,
        random_state=42
    )
    final_model.fit(data.T)
    print(f"Кластеризация завершена с {n_clusters} кластерами")
    return final_model, n_clusters


@app.cell
def _(final_model, n_clusters, plt):
    # Визуализация центроидов кластеров
    plt.figure(figsize=(12, 4))
    for _cluster_num in range(n_clusters):
        plt.plot(final_model.cluster_centers_[_cluster_num, :, 0].T,
                 label=f'Кластер {_cluster_num}', linewidth=2)
    plt.title('Центроиды кластеров (характерные профили)')
    plt.xlabel('Временной шаг')
    plt.ylabel('Oil Rate')
    plt.legend()
    plt.grid(True)
    plt.show()
    return


@app.cell
def _(data, final_model):
    # Присваиваем кластеры скважинам
    _df_clusters = data.T.copy()
    _df_clusters['cluster'] = final_model.predict(_df_clusters)
    cluster_dict = dict(_df_clusters['cluster'])
    print("Распределение скважин по кластерам:")
    print(_df_clusters['cluster'].value_counts().sort_index())
    print("\nСкважины и их кластеры:")
    for _well, _cluster in sorted(cluster_dict.items(), key=lambda x: (x[1], x[0])):
        print(f"  {_well}: кластер {_cluster}")
    return (cluster_dict,)


@app.cell
def _(cluster_dict, df):
    # Добавляем информацию о кластере в основной датафрейм
    df['cluster'] = df['well'].map(cluster_dict)
    df.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Визуализация профилей по кластерам
    """)
    return


@app.cell
def _(column, df, n_clusters, plt):
    # Визуализация профилей добычи по кластерам
    _colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    plt.figure(figsize=(15, 5))

    for _cluster_idx in range(n_clusters):
        _df_cluster = df[df['cluster'] == _cluster_idx]
        for _well in _df_cluster['well'].unique():
            _well_data = _df_cluster[_df_cluster['well'] == _well]
            plt.plot(_well_data['date'], _well_data[column],
                     label=_well, color=_colors[_cluster_idx % len(_colors)])

    plt.legend(loc='upper center', frameon=True, bbox_to_anchor=(0.5, -0.2), ncol=6)
    plt.grid(ls='--')
    plt.xlabel('Date')
    plt.ylabel('Oil Rate (bbl/day)')
    plt.title('Профили добычи по кластерам (цвет = кластер)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Построение кластерной карты (Clustermap) и дендрограммы
    """)
    return


@app.cell
def _(data, dtw, pd, pdist, plt, sns, squareform):
    # Построение матрицы расстояний DTW и кластерной карты
    _series = data.copy().T.to_numpy()
    distances = pdist(_series, dtw)
    _distance_matrix = pd.DataFrame(
        data=squareform(distances),
        index=data.columns,
        columns=data.columns
    )

    # Clustermap
    _cg = sns.clustermap(_distance_matrix, cmap='viridis_r', cbar_pos=None, figsize=(10, 8))
    _cg.fig.suptitle('Кластерная карта DTW-расстояний', size=14, y=1.02)
    _cg.ax_heatmap.set_xlabel('Скважина', size=10)
    _cg.ax_heatmap.set_ylabel('Скважина', size=10)
    plt.setp(_cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    plt.show()
    return (distances,)


@app.cell
def _(data, dendrogram, distances, linkage, plt):
    # Построение дендрограммы
    plt.figure(figsize=(12, 5))
    _links = linkage(distances, method="average", metric="euclidean", optimal_ordering=True)
    plt.title("Иерархическая кластеризация профилей добычи (дендрограмма)")
    plt.xlabel("Скважина", fontsize=10)
    plt.ylabel("DTW-расстояние")

    dendrogram(_links, color_threshold=2, leaf_font_size=10,
               labels=data.columns, leaf_rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Визуализация кластеров на карте

    Объединяем данные о кластерах с координатами скважин
    """)
    return


@app.cell
def _(cluster_dict, coords, pd):
    # Создаём датафрейм с кластерами и координатами
    _wells_clusters = pd.DataFrame({
        'well': list(cluster_dict.keys()),
        'cluster': list(cluster_dict.values())
    })

    # Объединяем с координатами
    map_data = coords.merge(_wells_clusters, on='well', how='inner')
    print(f"Скважин с координатами и кластерами: {len(map_data)}")
    map_data
    return (map_data,)


@app.cell
def _(map_data, n_clusters, plt):
    # Простая визуализация кластеров на карте
    plt.figure(figsize=(12, 8))

    _colors_map = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    _markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

    for _cluster_id in range(n_clusters):
        _cluster_data = map_data[map_data['cluster'] == _cluster_id]
        plt.scatter(_cluster_data['x'], _cluster_data['y'],
                    c=_colors_map[_cluster_id % len(_colors_map)],
                    marker=_markers[_cluster_id % len(_markers)],
                    s=200, label=f'Кластер {_cluster_id}',
                    edgecolors='black', linewidth=1.5)

    # Подписи скважин
    for _, _row in map_data.iterrows():
        plt.annotate(_row['well'], xy=(_row['x'], _row['y']),
                     xytext=(5, 5), textcoords='offset points', fontsize=9)

    plt.xlabel('X (Easting)', fontsize=12)
    plt.ylabel('Y (Northing)', fontsize=12)
    plt.title('Пространственное распределение кластеров скважин COSTA', fontsize=14)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    # Добавляем статистику
    _stats_text = f"Всего скважин: {len(map_data)}\nКластеров: {n_clusters}"
    plt.text(0.98, 0.02, _stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Визуализация с диаграммой Вороного
    """)
    return


@app.cell
def _(matplotlib, np):
    def discrete_cmap(N, base_cmap=None):
        """Создаёт дискретную цветовую карту из N цветов"""
        base = matplotlib.colormaps.get_cmap(base_cmap)
        color_list = base(np.linspace(0, 1, N))
        cmap_name = base.name + str(N)
        return matplotlib.colors.LinearSegmentedColormap.from_list(cmap_name, color_list, N)

    def voronoi_finite_polygons_2d(vor, radius=None):
        """Преобразует бесконечные регионы Вороного в конечные полигоны"""
        if vor.points.shape[1] != 2:
            raise ValueError('Requires 2D input')
        new_regions = []
        new_vertices = vor.vertices.tolist()
        center = vor.points.mean(axis=0)
        if radius is None:
            radius = np.ptp(vor.points, axis=0).max() * 2
        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))
        for p1, region in enumerate(vor.point_region):
            vertices = vor.regions[region]
            if all((v >= 0 for v in vertices)):
                new_regions.append(vertices)
                continue
            ridges = all_ridges[p1]
            new_region = [v for v in vertices if v >= 0]
            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = (v2, v1)
                if v1 >= 0:
                    continue
                t = vor.points[p2] - vor.points[p1]
                t = t / np.linalg.norm(t)
                n = np.array([-t[1], t[0]])
                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[v2] + direction * radius
                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())
            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]
            new_regions.append(new_region.tolist())
        return (new_regions, np.asarray(new_vertices))
    return discrete_cmap, voronoi_finite_polygons_2d


@app.cell
def _(
    Voronoi,
    discrete_cmap,
    map_data,
    matplotlib,
    n_clusters,
    np,
    plt,
    voronoi_finite_polygons_2d,
):
    # Визуализация с диаграммой Вороного
    _fig_vor = plt.figure(figsize=(14, 10))
    _ax_vor = _fig_vor.gca()

    # Координаты
    _coordinates = np.array(map_data[['x', 'y']])
    _vor = Voronoi(_coordinates)
    _regions, _vertices = voronoi_finite_polygons_2d(_vor)

    # Нормализация цветов
    _minima = map_data['cluster'].min()
    _maxima = map_data['cluster'].max()
    _norm = matplotlib.colors.Normalize(vmin=_minima, vmax=_maxima, clip=True)
    _mapper = matplotlib.cm.ScalarMappable(norm=_norm, cmap=matplotlib.cm.tab10)

    # Рисуем полигоны Вороного
    for _idx, _region in enumerate(_regions):
        _polygon = _vertices[_region]
        _x_poly = np.append(_polygon[:, 0], _polygon[0, 0])
        _y_poly = np.append(_polygon[:, 1], _polygon[0, 1])
        plt.plot(_x_poly, _y_poly, 'k', linewidth=0.5)
        plt.fill(*zip(*_polygon), alpha=0.6,
                 color=_mapper.to_rgba(map_data['cluster'].iloc[_idx]))

    # Скважины
    _scatter = plt.scatter(map_data['x'], map_data['y'],
                          c=map_data['cluster'],
                          cmap=discrete_cmap(n_clusters, 'tab10'),
                          marker='o', s=150, edgecolors='black', linewidth=2)

    # Подписи
    for _, _row_vor in map_data.iterrows():
        plt.annotate(_row_vor['well'], xy=(_row_vor['x'], _row_vor['y']),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=9, fontweight='bold')

    plt.xlabel('X (Easting)', fontsize=12)
    plt.ylabel('Y (Northing)', fontsize=12)
    plt.title('Карта кластеров с диаграммой Вороного', fontsize=14)
    plt.colorbar(_scatter, label='Кластер', ticks=range(n_clusters))
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Анализ пространственного распределения кластеров
    """)
    return


@app.cell
def _(map_data, n_clusters, np):
    # Количественная оценка пространственной кластеризации
    from scipy.spatial.distance import cdist

    print("=" * 60)
    print("АНАЛИЗ ПРОСТРАНСТВЕННОГО РАСПРЕДЕЛЕНИЯ КЛАСТЕРОВ")
    print("=" * 60)

    # Для каждого кластера вычисляем среднее расстояние между скважинами
    _spatial_stats = []
    for _cluster_id in range(n_clusters):
        _cluster_wells = map_data[map_data['cluster'] == _cluster_id]
        if len(_cluster_wells) > 1:
            _coords = _cluster_wells[['x', 'y']].values
            _distances = cdist(_coords, _coords)
            _mean_dist = _distances[np.triu_indices(len(_coords), k=1)].mean()
            _max_dist = _distances.max()
        else:
            _mean_dist = 0
            _max_dist = 0

        _spatial_stats.append({
            'cluster': _cluster_id,
            'n_wells': len(_cluster_wells),
            'mean_internal_distance': _mean_dist,
            'max_internal_distance': _max_dist,
            'wells': list(_cluster_wells['well'])
        })

    for _stat in _spatial_stats:
        print(f"\nКластер {_stat['cluster']}:")
        print(f"  Количество скважин: {_stat['n_wells']}")
        print(f"  Скважины: {', '.join(_stat['wells'])}")
        print(f"  Среднее расстояние между скважинами: {_stat['mean_internal_distance']:.0f} м")
        print(f"  Максимальное расстояние: {_stat['max_internal_distance']:.0f} м")

    # Общая метрика: средневзвешенное внутрикластерное расстояние
    _total_wells = sum(s['n_wells'] for s in _spatial_stats)
    _weighted_avg = sum(s['mean_internal_distance'] * s['n_wells'] for s in _spatial_stats) / _total_wells
    print(f"\n{'=' * 60}")
    print(f"Средневзвешенное внутрикластерное расстояние: {_weighted_avg:.0f} м")
    print(f"{'=' * 60}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. Выводы

    В ходе выполнения лабораторной работы была проведена кластеризация профилей добычи нефти 17 разведочных скважин синтетического месторождения COSTA с использованием метода Dynamic Time Warping. Исходные данные представляют собой результаты гидродинамического моделирования в симуляторе CMG IMEX на 30-летний период прогнозной эксплуатации с помесячной дискретизацией. Для определения оптимального числа кластеров применялся коэффициент силуэта в сочетании с методом локтя, что позволило выявить три устойчивые группы скважин. Значение Silhouette Score составило 0.32, что соответствует хорошему качеству кластеризации и свидетельствует о наличии реальной структуры в данных.

    Результаты кластеризации выявили три группы скважин с принципиально различными характеристиками добычи. Первый кластер объединил четыре высокопродуктивные скважины со средним дебитом около 3640 баррелей в сутки, второй кластер включил восемь скважин с низкой продуктивностью и средним дебитом порядка 38 баррелей в сутки, а третий кластер сформировался из пяти скважин со средним уровнем добычи около 1140 баррелей в сутки. Такое распределение отражает естественную геологическую неоднородность карбонатного резервуара и различия в фильтрационно-ёмкостных свойствах пластов в зонах расположения скважин.

    Пространственный анализ показал, что кластеры характеризуются различной степенью географической компактности. Наиболее сконцентрированным оказался кластер высокопродуктивных скважин со средним внутрикластерным расстоянием около 14.5 километров, тогда как скважины с низкой продуктивностью распределены по значительно большей территории со средним расстоянием порядка 30 километров. Средневзвешенное внутрикластерное расстояние для всего месторождения составило 23.5 километра, что указывает на отсутствие строгой пространственной локализации кластеров и подтверждает влияние локальных геологических факторов на продуктивность отдельных скважин.

    Проведённый анализ демонстрирует эффективность метода DTW для кластеризации временных рядов добычи, поскольку он учитывает не только абсолютные значения дебитов, но и динамику их изменения во времени. Полученные результаты могут быть использованы для оптимизации стратегии разработки месторождения, выявления зон с повышенным потенциалом добычи и планирования мероприятий по интенсификации притока на малопродуктивных скважинах. Для более детального анализа пространственной связи между кластерами и геологическими характеристиками резервуара рекомендуется дополнительно рассмотреть корреляцию с данными сейсморазведки и петрофизическими параметрами.
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
