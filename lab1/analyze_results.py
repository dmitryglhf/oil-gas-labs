"""
Скрипт для анализа результатов кластеризации COSTA
"""
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import dtw
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Пути к данным
DATA_DIR = Path(__file__).parent / "COSTA MODEL"
PRODUCTION_PATH = DATA_DIR / "5 - Production" / "COSTA Synthetic Production Monthly.xlsx"
COORDS_PATH = DATA_DIR / "4 - Building 3D Geo-Model" / "1 - Well Heads" / "COSTA Well Heads.prn"

print("=" * 70)
print("АНАЛИЗ РЕЗУЛЬТАТОВ КЛАСТЕРИЗАЦИИ COSTA")
print("=" * 70)

# 1. Загрузка данных
print("\n1. ЗАГРУЗКА ДАННЫХ")
print("-" * 50)

df_raw = pd.read_excel(PRODUCTION_PATH, sheet_name='Appraisal Wells', header=3)
df_raw.columns = ['well', 'date', 'gor', 'gas_rate', 'oil_rate', 'watercut', 'water_rate', 'bhp']
df = df_raw[df_raw['well'] != 'Well'].copy()
df['date'] = pd.to_datetime(df['date'])
df['oil_rate'] = pd.to_numeric(df['oil_rate'])

coords = pd.read_csv(COORDS_PATH, sep=r'\s+', skiprows=1, names=['well', 'x', 'y', 'kb', 'td'])

print(f"Загружено записей: {len(df)}")
print(f"Количество скважин: {len(df['well'].unique())}")
print(f"Скважины: {sorted(df['well'].unique(), key=lambda x: int(x.split('-')[1]))}")
print(f"Диапазон дат: {df['date'].min().strftime('%Y-%m-%d')} - {df['date'].max().strftime('%Y-%m-%d')}")
print(f"Количество временных точек: {df.groupby('well').size().iloc[0]}")

# 2. Подготовка данных
print("\n2. ПОДГОТОВКА ДАННЫХ")
print("-" * 50)

data = df.pivot_table(index='date', columns='well', values='oil_rate').bfill()
print(f"Размер матрицы данных: {data.shape} (временные точки x скважины)")

scaler = StandardScaler()
data_scaled = data.copy()
data_scaled[data_scaled.columns] = scaler.fit_transform(data_scaled)

# 3. Определение оптимального числа кластеров
print("\n3. ОПРЕДЕЛЕНИЕ ОПТИМАЛЬНОГО ЧИСЛА КЛАСТЕРОВ")
print("-" * 50)

silhouette_scores = []
inertias = []
n_range = range(2, min(10, len(data.columns)))

for k in n_range:
    model = TimeSeriesKMeans(n_clusters=k, metric="dtw", n_jobs=-1, max_iter=10, random_state=42)
    model.fit(data.T)
    inertias.append(model.inertia_)
    silhouette_scores.append(silhouette_score(data_scaled.T, model.labels_))
    print(f"  k={k}: Silhouette={silhouette_scores[-1]:.4f}, Inertia={inertias[-1]:.2f}")

best_k = list(n_range)[silhouette_scores.index(max(silhouette_scores))]
print(f"\nОптимальное число кластеров: {best_k}")
print(f"Максимальный Silhouette Score: {max(silhouette_scores):.4f}")

# 4. Финальная кластеризация
print("\n4. РЕЗУЛЬТАТЫ КЛАСТЕРИЗАЦИИ")
print("-" * 50)

final_model = TimeSeriesKMeans(n_clusters=best_k, metric='dtw', n_jobs=-1, max_iter=100, random_state=42)
final_model.fit(data.T)

cluster_dict = dict(zip(data.columns, final_model.labels_))

print(f"Распределение скважин по кластерам:")
for cluster_id in range(best_k):
    wells_in_cluster = [w for w, c in cluster_dict.items() if c == cluster_id]
    print(f"  Кластер {cluster_id}: {len(wells_in_cluster)} скважин - {', '.join(sorted(wells_in_cluster, key=lambda x: int(x.split('-')[1])))}")

# 5. Анализ DTW-расстояний
print("\n5. АНАЛИЗ DTW-РАССТОЯНИЙ")
print("-" * 50)

series = data.T.to_numpy()
distances = pdist(series, dtw)
dist_matrix = squareform(distances)

print(f"Минимальное DTW-расстояние: {distances.min():.2f}")
print(f"Максимальное DTW-расстояние: {distances.max():.2f}")
print(f"Среднее DTW-расстояние: {distances.mean():.2f}")

# Найти наиболее похожие и различные пары
dist_df = pd.DataFrame(dist_matrix, index=data.columns, columns=data.columns)
min_idx = np.unravel_index(np.argmin(dist_matrix + np.eye(len(dist_matrix)) * 1e10), dist_matrix.shape)
max_idx = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)

print(f"Наиболее похожие скважины: {data.columns[min_idx[0]]} и {data.columns[min_idx[1]]} (расстояние: {dist_matrix[min_idx]:.2f})")
print(f"Наиболее различные скважины: {data.columns[max_idx[0]]} и {data.columns[max_idx[1]]} (расстояние: {dist_matrix[max_idx]:.2f})")

# 6. Пространственный анализ
print("\n6. ПРОСТРАНСТВЕННЫЙ АНАЛИЗ КЛАСТЕРОВ")
print("-" * 50)

wells_clusters = pd.DataFrame({'well': list(cluster_dict.keys()), 'cluster': list(cluster_dict.values())})
map_data = coords.merge(wells_clusters, on='well', how='inner')

print(f"Скважин с координатами: {len(map_data)}")

for cluster_id in range(best_k):
    cluster_wells = map_data[map_data['cluster'] == cluster_id]
    if len(cluster_wells) > 1:
        coords_arr = cluster_wells[['x', 'y']].values
        dists = cdist(coords_arr, coords_arr)
        mean_dist = dists[np.triu_indices(len(coords_arr), k=1)].mean()
        max_dist = dists.max()

        # Центроид кластера
        centroid_x = cluster_wells['x'].mean()
        centroid_y = cluster_wells['y'].mean()
    else:
        mean_dist = 0
        max_dist = 0
        centroid_x = cluster_wells['x'].iloc[0] if len(cluster_wells) > 0 else 0
        centroid_y = cluster_wells['y'].iloc[0] if len(cluster_wells) > 0 else 0

    print(f"\nКластер {cluster_id}:")
    print(f"  Количество скважин: {len(cluster_wells)}")
    print(f"  Центроид: X={centroid_x:.0f}, Y={centroid_y:.0f}")
    print(f"  Среднее расстояние между скважинами: {mean_dist:.0f} м")
    print(f"  Максимальное расстояние: {max_dist:.0f} м")

# Общая метрика пространственной компактности
total_wells = len(map_data)
weighted_avg = sum(
    cdist(map_data[map_data['cluster'] == c][['x', 'y']].values,
          map_data[map_data['cluster'] == c][['x', 'y']].values)[np.triu_indices(len(map_data[map_data['cluster'] == c]), k=1)].mean()
    * len(map_data[map_data['cluster'] == c])
    for c in range(best_k) if len(map_data[map_data['cluster'] == c]) > 1
) / total_wells

print(f"\nСредневзвешенное внутрикластерное расстояние: {weighted_avg:.0f} м")

# 7. Статистика по профилям добычи
print("\n7. СТАТИСТИКА ПО ПРОФИЛЯМ ДОБЫЧИ")
print("-" * 50)

for cluster_id in range(best_k):
    wells_in_cluster = [w for w, c in cluster_dict.items() if c == cluster_id]
    cluster_data = df[df['well'].isin(wells_in_cluster)]

    print(f"\nКластер {cluster_id}:")
    print(f"  Средний дебит нефти: {cluster_data['oil_rate'].mean():.1f} bbl/day")
    print(f"  Макс. дебит нефти: {cluster_data['oil_rate'].max():.1f} bbl/day")
    print(f"  Мин. дебит нефти: {cluster_data['oil_rate'].min():.1f} bbl/day")
    print(f"  Стд. отклонение: {cluster_data['oil_rate'].std():.1f} bbl/day")

# 8. Проверка качества кластеризации
print("\n8. ОЦЕНКА КАЧЕСТВА КЛАСТЕРИЗАЦИИ")
print("-" * 50)

final_silhouette = silhouette_score(data_scaled.T, final_model.labels_)
print(f"Silhouette Score: {final_silhouette:.4f}")

if final_silhouette > 0.5:
    quality = "ОТЛИЧНОЕ"
elif final_silhouette > 0.25:
    quality = "ХОРОШЕЕ"
elif final_silhouette > 0:
    quality = "УДОВЛЕТВОРИТЕЛЬНОЕ"
else:
    quality = "ПЛОХОЕ"

print(f"Качество кластеризации: {quality}")
print(f"  (>0.5 - отличное, 0.25-0.5 - хорошее, 0-0.25 - удовлетворительное, <0 - плохое)")

print("\n" + "=" * 70)
print("АНАЛИЗ ЗАВЕРШЁН")
print("=" * 70)
