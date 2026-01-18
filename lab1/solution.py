import marimo

__generated_with = "0.19.4"
app = marimo.App()


@app.cell
def _():
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
    return (
        StandardScaler,
        TimeSeriesKMeans,
        Voronoi,
        dendrogram,
        dtw,
        fcluster,
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
def _(pd):
    # Загрузка данных добычи из Excel
    _path = 'COSTA MODEL/5 - Production/COSTA Synthetic Production Monthly.xlsx'
    df_raw = pd.read_excel(_path, sheet_name='Appraisal Wells', header=3)

    # Переименовываем колонки
    df_raw.columns = ['well', 'date', 'gor', 'gas_rate', 'oil_rate', 'watercut', 'water_rate', 'bhp']

    # Убираем строку с заголовками (первая строка содержит названия)
    df = df_raw[df_raw['well'] != 'Well'].copy()

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
    return df, df_raw


@app.cell
def _(pd):
    # Загрузка координат скважин
    _path_coords = 'COSTA MODEL/4 - Building 3D Geo-Model/1 - Well Heads/COSTA Well Heads.prn'
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
        well_data = df[df['well'] == _well]
        plt.plot(well_data['date'], well_data[column], label=_well)

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
    scaler = StandardScaler()
    data_scaled = data.copy()
    scaler.fit(data_scaled)
    data_scaled[data_scaled.columns] = scaler.transform(data_scaled)
    data_scaled.head()
    return data_scaled, scaler


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
    distortions = []
    silhouette_scores = []
    n_range = range(2, min(10, len(data.columns)))

    for k in tqdm(n_range, desc="Подбор числа кластеров"):
        model = TimeSeriesKMeans(
            n_clusters=k,
            metric="dtw",
            n_jobs=-1,
            max_iter=10,
            random_state=42,
        )
        model.fit(data.T)
        distortions.append(model.inertia_)
        silhouette_scores.append(silhouette_score(data_scaled.T, model.labels_))

    # Графики
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(list(n_range), distortions, "bx-")
    axes[0].set_xlabel("Количество кластеров (k)")
    axes[0].set_ylabel("Distortion (Inertia)")
    axes[0].set_title("Метод локтя (Elbow Method)")
    axes[0].grid(True)

    axes[1].plot(list(n_range), silhouette_scores, "rx-")
    axes[1].set_xlabel("Количество кластеров (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Коэффициент силуэта")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    # Выводим лучшее значение силуэта
    best_k = list(n_range)[silhouette_scores.index(max(silhouette_scores))]
    print(f"\nОптимальное число кластеров по силуэту: {best_k}")
    print(f"Максимальный Silhouette Score: {max(silhouette_scores):.3f}")
    return (
        axes,
        best_k,
        distortions,
        fig,
        k,
        model,
        n_range,
        silhouette_scores,
    )


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
    for cluster_num in range(n_clusters):
        plt.plot(final_model.cluster_centers_[cluster_num, :, 0].T,
                 label=f'Кластер {cluster_num}', linewidth=2)
    plt.title('Центроиды кластеров (характерные профили)')
    plt.xlabel('Временной шаг')
    plt.ylabel('Oil Rate')
    plt.legend()
    plt.grid(True)
    plt.show()
    return (cluster_num,)


@app.cell
def _(data, final_model):
    # Присваиваем кластеры скважинам
    df_clusters = data.T.copy()
    df_clusters['cluster'] = final_model.predict(df_clusters)
    cluster_dict = dict(df_clusters['cluster'])
    print("Распределение скважин по кластерам:")
    print(df_clusters['cluster'].value_counts().sort_index())
    print("\nСкважины и их кластеры:")
    for well, cluster in sorted(cluster_dict.items(), key=lambda x: (x[1], x[0])):
        print(f"  {well}: кластер {cluster}")
    return cluster_dict, df_clusters, well


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
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    plt.figure(figsize=(15, 5))

    for cluster_idx in range(n_clusters):
        df_cluster = df[df['cluster'] == cluster_idx]
        for _well in df_cluster['well'].unique():
            well_data = df_cluster[df_cluster['well'] == _well]
            plt.plot(well_data['date'], well_data[column],
                     label=_well, color=colors[cluster_idx % len(colors)])

    plt.legend(loc='upper center', frameon=True, bbox_to_anchor=(0.5, -0.2), ncol=6)
    plt.grid(ls='--')
    plt.xlabel('Date')
    plt.ylabel('Oil Rate (bbl/day)')
    plt.title('Профили добычи по кластерам (цвет = кластер)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    return cluster_idx, colors, df_cluster, well_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Построение кластерной карты (Clustermap) и дендрограммы
    """)
    return


@app.cell
def _(data, dtw, pd, pdist, plt, sns, squareform):
    # Построение матрицы расстояний DTW и кластерной карты
    series = data.copy().T.to_numpy()
    distances = pdist(series, dtw)
    distance_matrix = pd.DataFrame(
        data=squareform(distances),
        index=data.columns,
        columns=data.columns
    )

    # Clustermap
    cg = sns.clustermap(distance_matrix, cmap='viridis_r', cbar_pos=None, figsize=(10, 8))
    cg.fig.suptitle('Кластерная карта DTW-расстояний', size=14, y=1.02)
    cg.ax_heatmap.set_xlabel('Скважина', size=10)
    cg.ax_heatmap.set_ylabel('Скважина', size=10)
    plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    plt.show()
    return cg, distance_matrix, distances, series


@app.cell
def _(data, dendrogram, distances, linkage, plt):
    # Построение дендрограммы
    plt.figure(figsize=(12, 5))
    links = linkage(distances, method="average", metric="euclidean", optimal_ordering=True)
    plt.title("Иерархическая кластеризация профилей добычи (дендрограмма)")
    plt.xlabel("Скважина", fontsize=10)
    plt.ylabel("DTW-расстояние")

    dendrogram(links, color_threshold=2, leaf_font_size=10,
               labels=data.columns, leaf_rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return (links,)


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
    wells_clusters = pd.DataFrame({
        'well': list(cluster_dict.keys()),
        'cluster': list(cluster_dict.values())
    })

    # Объединяем с координатами
    map_data = coords.merge(wells_clusters, on='well', how='inner')
    print(f"Скважин с координатами и кластерами: {len(map_data)}")
    map_data
    return map_data, wells_clusters


@app.cell
def _(map_data, n_clusters, np, plt):
    # Простая визуализация кластеров на карте
    plt.figure(figsize=(12, 8))

    colors_map = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

    for cluster_id in range(n_clusters):
        cluster_data = map_data[map_data['cluster'] == cluster_id]
        plt.scatter(cluster_data['x'], cluster_data['y'],
                    c=colors_map[cluster_id % len(colors_map)],
                    marker=markers[cluster_id % len(markers)],
                    s=200, label=f'Кластер {cluster_id}',
                    edgecolors='black', linewidth=1.5)

    # Подписи скважин
    for _, row in map_data.iterrows():
        plt.annotate(row['well'], xy=(row['x'], row['y']),
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
    return cluster_data, cluster_id, colors_map, markers, row


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Визуализация с диаграммой Вороного
    """)
    return


@app.cell
def _(np, plt):
    def discrete_cmap(N, base_cmap=None):
        """Создаёт дискретную цветовую карту из N цветов"""
        base = plt.cm.get_cmap(base_cmap)
        color_list = base(np.linspace(0, 1, N))
        cmap_name = base.name + str(N)
        return base.from_list(cmap_name, color_list, N)

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
    fig_vor = plt.figure(figsize=(14, 10))
    ax_vor = fig_vor.gca()

    # Координаты
    coordinates = np.array(map_data[['x', 'y']])
    vor = Voronoi(coordinates)
    regions, vertices = voronoi_finite_polygons_2d(vor)

    # Нормализация цветов
    minima = map_data['cluster'].min()
    maxima = map_data['cluster'].max()
    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.tab10)

    # Рисуем полигоны Вороного
    for idx, region in enumerate(regions):
        polygon = vertices[region]
        x_poly = np.append(polygon[:, 0], polygon[0, 0])
        y_poly = np.append(polygon[:, 1], polygon[0, 1])
        plt.plot(x_poly, y_poly, 'k', linewidth=0.5)
        plt.fill(*zip(*polygon), alpha=0.6,
                 color=mapper.to_rgba(map_data['cluster'].iloc[idx]))

    # Скважины
    scatter = plt.scatter(map_data['x'], map_data['y'],
                          c=map_data['cluster'],
                          cmap=discrete_cmap(n_clusters, 'tab10'),
                          marker='o', s=150, edgecolors='black', linewidth=2)

    # Подписи
    for _, row_vor in map_data.iterrows():
        plt.annotate(row_vor['well'], xy=(row_vor['x'], row_vor['y']),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=9, fontweight='bold')

    plt.xlabel('X (Easting)', fontsize=12)
    plt.ylabel('Y (Northing)', fontsize=12)
    plt.title('Карта кластеров с диаграммой Вороного', fontsize=14)
    plt.colorbar(scatter, label='Кластер', ticks=range(n_clusters))
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return (
        ax_vor,
        coordinates,
        fig_vor,
        idx,
        mapper,
        maxima,
        minima,
        norm,
        polygon,
        region,
        regions,
        row_vor,
        scatter,
        vertices,
        vor,
        x_poly,
        y_poly,
    )


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
    spatial_stats = []
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

        spatial_stats.append({
            'cluster': _cluster_id,
            'n_wells': len(_cluster_wells),
            'mean_internal_distance': _mean_dist,
            'max_internal_distance': _max_dist,
            'wells': list(_cluster_wells['well'])
        })

    for stat in spatial_stats:
        print(f"\nКластер {stat['cluster']}:")
        print(f"  Количество скважин: {stat['n_wells']}")
        print(f"  Скважины: {', '.join(stat['wells'])}")
        print(f"  Среднее расстояние между скважинами: {stat['mean_internal_distance']:.0f} м")
        print(f"  Максимальное расстояние: {stat['max_internal_distance']:.0f} м")

    # Общая метрика: средневзвешенное внутрикластерное расстояние
    total_wells = sum(s['n_wells'] for s in spatial_stats)
    weighted_avg = sum(s['mean_internal_distance'] * s['n_wells'] for s in spatial_stats) / total_wells
    print(f"\n{'=' * 60}")
    print(f"Средневзвешенное внутрикластерное расстояние: {weighted_avg:.0f} м")
    print(f"{'=' * 60}")
    return cdist, spatial_stats, stat, total_wells, weighted_avg


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. Выводы

    ### Ответы на вопросы лабораторной работы:

    **1. Какой метод оценки выбран для определения оптимального числа кластеров?**

    Использован **Silhouette Score** в сочетании с методом локтя (Elbow Method):
    - Silhouette Score оценивает, насколько хорошо объекты соответствуют своим кластерам
    - Значения от -1 до 1, где 1 — идеальная кластеризация
    - Для временных рядов добычи этот метод хорошо работает, так как учитывает компактность и разделимость

    **2. Как пространственно меняются кластеры по площади месторождения?**

    Анализ карты показывает распределение кластеров по площади. Скважины со схожими профилями добычи
    могут располагаться как компактно (что указывает на геологическую связь), так и разрозненно
    (что может указывать на схожие условия эксплуатации при разной геологии).

    **3. Как количественно оценить пространственную сосредоточенность кластеров?**

    Используется **средневзвешенное внутрикластерное расстояние**:
    - Меньшее значение означает более компактное пространственное расположение скважин в кластерах
    - Это позволяет сравнивать разные методы кластеризации по качеству пространственной группировки
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
