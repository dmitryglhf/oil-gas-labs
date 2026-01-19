import marimo

__generated_with = "0.19.4"
app = marimo.App()


@app.cell
def _():
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # увеличим количество отображаемых столбцов
    pd.set_option("display.max_columns", None)
    # увеличим количество отображаемых рядов
    pd.set_option("display.max_rows", 500)
    # увеличим ширину столбцов
    pd.options.display.max_colwidth = 250

    # отключаем экспоненциальное представление
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
    # Кластеризация профилей добычи
    """)
    return


@app.cell
def _(pd):
    # Define the path to the data file
    _path = 'data/data.txt'
    df = pd.read_csv(_path, delimiter='\t')
    # Load the data file into a pandas DataFrame. The delimiter is set to tab character as the file is a TSV (Tab-separated values) file
    df.columns = [x.lower() for x in df.columns]
    df['well'] = df['well'].str.lower().str.replace('well', '')
    # Convert all column names to lower case to maintain consistency and ease of access
    df['date'] = pd.to_datetime(df['date'])
    # Replace any occurrence of the string "well" in the 'well' column with an empty string and convert the values to lower case
    # Convert the 'date' column to datetime format for easier manipulation in future operations
    # Display the first 5 rows of the DataFrame for a quick overview of the data
    df.head()
    return (df,)


@app.cell
def _(df):
    # This block of code operates on a DataFrame named 'df' which contains data about wells.

    # Count the total number of unique wells in the DataFrame
    print(len(df["well"].unique()))

    # Output the names of the unique wells
    print(df["well"].unique())
    return


@app.cell
def _(df, plt):
    # 'qoil' is the column we're interested in plotting
    column = 'qoil'
    plt.figure(figsize=(15, 5))
    # Create a new figure with specified size
    for _i, _col in enumerate(df['well'].unique()):
        plt.plot(df[df['well'] == _col][column], label=_col)
    # Loop over each unique well in the dataframe
    _leg = plt.legend(loc='upper center', frameon=True, bbox_to_anchor=(0.5, -0.15), ncol=8)
    plt.grid(ls='--')  # Plot the 'qoil' values for the current well
    plt.xlabel('Index')
    plt.ylabel(column)
    # Set up a legend to identify the wells in the plot
    # Add a grid to the plot with dashed lines (ls="--")
    plt.show()
    return


@app.cell
def _(df):
    df.head(1)
    return


@app.cell
def _(df):
    # Defines the column name that we are interested in
    column_1 = 'qoil'
    data = df.pivot_table(index='date', columns='well', values=column_1).fillna(method='bfill')
    # Pivots the DataFrame to create a new DataFrame
    # The new DataFrame has "date" as its index, "well" as its columns, and the values of the specified "column" as its values
    # The 'fillna' method is used to fill any NA/NaN values using the 'bfill' method, which propagates the next valid observation backward to the previous valid
    data.head()
    return column_1, data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Визуализируем профили добычи
    """)
    return


@app.cell
def _(column_1, data, plt):
    ls = ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '--', '--', '--', '--', '--', '--', '--', '--', '--']
    plt.figure(figsize=(15, 5))
    for _i, _col in enumerate(data.columns):
        plt.plot(data[_col], label=_col, ls=ls[_i])
    _leg = plt.legend(loc='upper center', frameon=True, bbox_to_anchor=(0.5, -0.3), ncol=8)
    plt.grid(ls='--')
    plt.xlabel('Date')
    plt.ylabel(column_1)
    plt.xticks(rotation=90)
    plt.show()
    return (ls,)


@app.cell
def _(StandardScaler, data):
    # Instantiate StandardScaler, a preprocessing module that standardizes features by removing the mean and scaling to unit variance
    scaler = StandardScaler()

    # Make a copy of the original DataFrame to avoid changing the original data
    data_scaled = data.copy()

    # Compute the mean and standard deviation based on the training data
    scaler.fit(data_scaled)

    # Perform standardization by centering and scaling
    data_scaled[data_scaled.columns] = scaler.transform(data_scaled)

    data_scaled.head()
    return (data_scaled,)


@app.cell
def _(data_scaled, ls, plt):
    plt.figure(figsize=(15, 5))
    for _i, _col in enumerate(data_scaled.columns):
        plt.plot(data_scaled.index, data_scaled[_col], label=_col, linestyle=ls[_i])
    _leg = plt.legend(loc='upper center', frameon=True, bbox_to_anchor=(0.5, -0.3), ncol=8)
    plt.grid(ls='--')
    plt.xlabel('Date')
    plt.xticks(rotation=90)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Можете ли мы выделить скважины которые по вашей "экспертной" оценке попадают в отдельные кластера?
    - На какие свойства профилей вы опирались при определении кластеров?
    <br>
    <br>
    Порстроение кластермапа
    """)
    return


@app.cell
def _(TimeSeriesKMeans, data, data_scaled, plt, silhouette_score, tqdm):
    # Initialize empty lists to store distortions and silhouette scores
    distortions = []
    silhouette = []

    # Specify the range of 'k' values we want to test (number of clusters)
    n = range(1, 10)

    # Loop over each 'k' value
    for k in tqdm(n):
        # Initialize a TimeSeriesKMeans model
        model = TimeSeriesKMeans(
            n_clusters=k,  # number of clusters
            metric="dtw",  # use dynamic time warping distance
            n_jobs=1,  # number of parallel jobs
            max_iter=10,  # maximum number of iterations
            random_state=0,  # random state for reproducibility
        )
        # Fit the model to the data
        model.fit(data.T)
        # Calculate and append the model's inertia (sum of squared distances of samples to their closest cluster center)
        distortions.append(model.inertia_)

        # Calculate and append silhouette score (measure of how similar an object is to its own cluster compared to other clusters)
        # for k > 1, since silhouette score is not defined for a single cluster
        if k > 1:
            silhouette.append(silhouette_score(data_scaled.T, model.labels_))

    # Plot the distortions as a function of 'k'
    plt.figure(figsize=(10, 4))
    plt.plot(n, distortions, "bx-")
    plt.xlabel("k")
    plt.ylabel("Distortion")
    plt.title("Elbow Method")
    plt.show()

    # Plot the silhouette scores as a function of 'k'
    plt.figure(figsize=(10, 4))
    plt.plot(n[1:], silhouette, "bx-")
    plt.xlabel("k")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette")
    plt.show()
    return


@app.cell
def _(TimeSeriesKMeans, data):
    # Number of clusters to be generated
    n_clusters = 3
    model_1 = TimeSeriesKMeans(n_clusters=n_clusters, metric='dtw', n_jobs=-1, max_iter=100, random_state=0)
    # Initialize the Time Series K-Means model
    # n_clusters: the number of clusters to form
    # metric: the distance metric to be used, in this case, Dynamic Time Warping (DTW)
    # n_jobs: the number of CPU cores to use for computation, 1 means use one core (consider setting to -1 for using all cores)
    # max_iter: maximum number of iterations of the k-means algorithm for a single run
    # random_state: determines random number generation for centroid initialization, using an int will guarantee the same results across different calls
    # Fit the model to the data
    # Here the data is transposed (.T) because the TimeSeriesKMeans expects the time dimension to be the last
    model_1.fit(data.T)
    return model_1, n_clusters


@app.cell
def _(model_1, n_clusters, plt):
    # Create a new figure with specified size
    plt.figure(figsize=(12, 4))
    for cluster_number in range(n_clusters):
    # Iterate over each cluster
        plt.plot(model_1.cluster_centers_[cluster_number, :, 0].T, label=str(cluster_number))
    plt.title('Cluster centroids')  # Plot the centroid of each cluster
    plt.legend()  # The cluster centers are assumed to be 3D with shape (n_clusters, n_features, n_samples)
    # Set the title of the plot
    # Add a legend to the plot
    # Display the plot
    plt.show()  # We're extracting the first feature across all samples for each cluster and plotting that
    return


@app.cell
def _(data, model_1):
    # The transpose operation is used to swap the rows and columns of 'data'.
    df_transformed = data.T
    df_transformed['cluster'] = model_1.predict(df_transformed)
    # Use the trained model to predict the cluster for each instance in the dataset.
    # The results are added as a new column 'cluster' in the dataframe.
    d = dict(df_transformed['cluster'])
    # Convert the 'cluster' column into a dictionary.
    # The keys in the dictionary are the index values from the dataframe,
    # and the values in the dictionary are the corresponding values in the 'cluster' column.
    d
    return d, df_transformed


@app.cell
def _(d, df):
    # Map the 'well' column of the DataFrame 'df' to a new column 'cluster' using the dictionary 'd'
    df["cluster"] = df["well"].map(d)

    # Display the updated DataFrame 'df'
    df
    return


@app.cell
def _(df, plt):
    # Define a list of colors to be used for different clusters
    color = ['C0', 'C1', 'C2', 'C3']
    plt.figure(figsize=(15, 5))
    for j, _cluster in enumerate(df['cluster'].unique()):
        df_cluster = df[df['cluster'] == _cluster]
    # Iterate over each unique cluster in the DataFrame
        for _i, _well in enumerate(df_cluster['well'].unique()):
            df_well = df_cluster[df_cluster['well'] == _well]  # Create a subset DataFrame for each cluster
            plt.plot(df_well['date'], df_well['qoil'], label=_well, color=color[j % len(color)])
    _leg = plt.legend(loc='upper center', frameon=True, bbox_to_anchor=(0.5, -0.2), ncol=8)
    plt.grid(ls='--')  # Iterate over each unique well within the current cluster
    plt.xlabel('Date')
    plt.ylabel('qoil')
    plt.xticks(rotation=90)
    # Set the legend at the specified location with a box around it
    # Set the labels for the x and y axes
    plt.show()  # Plot the qoil over time for each well, assigning a unique color and label  # Use modulo operation to prevent index error if there are more clusters than colors
    return


@app.cell
def _(data, dtw, pd, pdist, plt, sns, squareform):
    # Set the size for the labels on the plot
    size = 10
    series = data.copy().T.to_numpy()
    # Transpose the original data and convert it to a numpy array
    # This is done because the pdist function expects rows to represent observations (wells in this case)
    # and columns to represent variables (time points in this case)
    distances = pdist(series, dtw)
    distance_matrix = pd.DataFrame(data=squareform(distances), index=data.columns, columns=data.columns)
    # Compute the pairwise distances between all wells using the Dynamic Time Warping (DTW) method
    # DTW is a measure used to compute the distance between two temporal sequences, which may vary in speed
    cg = sns.clustermap(distance_matrix, cmap='viridis_r', cbar_pos=None)
    cg.fig.suptitle('$Q_{oil}\\ clustermap$', size=size)
    # Convert the condensed distance matrix returned by pdist into a square matrix
    # and create a DataFrame using the well names as the row and column labels
    plt.ylabel('Distance')
    _ax = cg.ax_heatmap
    _ax.set_xlabel('Well', size=size)
    _ax.set_ylabel('Well', size=size)
    # Create a clustermap (a type of heatmap) of the distance matrix
    # viridis_r is a sequential colormap (reversed)
    # cbar_pos=None removes the colorbar
    cg.fig.subplots_adjust(top=0.92)
    plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    plt.show()
    return distances, series


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Построение дендрограммы
    """)
    return


@app.cell
def _(data, dendrogram, distances, linkage, plt):
    plt.figure(figsize=(10, 4))

    # Perform hierarchical clustering using the linkage function
    # distances: A precomputed matrix of distances between points (assumed to be provided)
    # method="average": The linkage method to use. 'average' linkage computes the average distance between clusters
    # metric="euclidean": The distance metric to use. In this case, Euclidean distance
    # optimal_ordering=True: This reorders the linkage matrix so that the pairwise distances between successive leaves are minimal
    links = linkage(distances, method="average", metric="euclidean", optimal_ordering=True)

    plt.title(r"$Q_{oil}$" + " hierarchical clustering dendrogram")

    plt.xlabel("Well", fontsize=10)
    plt.ylabel("Distance")

    # Generate the dendrogram plot
    # links: The hierarchical clustering encoded as a linkage matrix
    # color_threshold=2: The color threshold for the dendrogram. All the descendent links deeper than color_threshold are colored the same
    # leaf_font_size=10: The font size for the leaf labels
    # labels=data.columns: The labels for the leaves (assumed to be the columns of a DataFrame 'data')
    # leaf_rotation=0: Rotates the leaf labels. Here, it's set to 0, so the labels are not rotated
    dn = dendrogram(
        links, color_threshold=2, leaf_font_size=10, labels=data.columns, leaf_rotation=0
    )

    plt.grid()
    plt.show()
    return (links,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Совпадают ли ваши "экспертные" кластера с тем, что получилось путем кластеризации с помощью алгоритма DTW?
    - Какие скважины не совпадают с вашей оценкой?
    <br>
    <br>
    Создадим датафрейм с именем скажины и ее принадлежность к кластеру
    """)
    return


@app.cell
def _(data, fcluster, links, np, pd, plt, series):
    well_names_list = np.asarray(data.columns)
    final_well_list = []
    final_cluster_list = []

    def visualize_clusters(series, links, num_clusters, num_series_to_draw=3, make_gray=False):
        results = fcluster(links, num_clusters, criterion='maxclust')
        s = pd.Series(results)
        clusters = s.unique()
        ncols = 2
        nrows = np.ceil(len(clusters) / float(ncols)).astype(int)
        count = 0
        f, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(15, 4 * nrows))
        for row in range(nrows):
            for _col in range(ncols):
                _ax = axs[row, _col] if nrows > 1 else axs[_col]
                cluster_idx = row * ncols + _col
                if cluster_idx >= len(clusters):
                    _ax.axis('off')
                    continue
                cluster_indices = s[s == cluster_idx + 1].index
                num_series_to_select = min(num_series_to_draw, len(cluster_indices))
                cluster_indices = np.random.choice(cluster_indices, num_series_to_select, replace=False)
                well_list = well_names_list[sorted(cluster_indices)].tolist()
                cluster_list = [count] * len(well_list)
                series_to_plot = series[sorted(cluster_indices), :]
                plt.suptitle('$Q_{oil}$' + ' clusters', fontsize=20)
                for _i, y in enumerate(series_to_plot):
                    color = 'gray' if make_gray else None
                    _ax.plot(y, color=color, alpha=0.75, label=well_list[_i])
                    _ax.set_title(str(count) + ' cluster', fontsize=15)
                    _ax.get_xaxis().set_ticklabels([])
                    _ax.get_yaxis().set_ticklabels([])
                final_well_list.extend(well_list)
                final_cluster_list.extend(cluster_list)
                count = count + 1
                _ax.grid(True)
                _ax.legend()
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
    num_clusters = 3
    num_series_to_draw = 100
    make_gray = False
    visualize_clusters(np.array(series), links, num_clusters, num_series_to_draw, make_gray)
    return


@app.cell
def _(df_transformed, n_clusters, np, plt):
    def plot_cluster_tickers(current_cluster):
        fig, _ax = plt.subplots(int(np.ceil(current_cluster.shape[0] / 4)), 4, figsize=(15, 3 * int(np.ceil(current_cluster.shape[0] / 4))))  # The number of subplots is determined by the number of rows in the current_cluster dataframe.
        fig.autofmt_xdate(rotation=45)  # The layout is arranged in a 4-column grid, with the number of rows determined by dividing the number of data points by 4 and rounding up.
        _ax = _ax.reshape(-1)
        for index, (_, row) in enumerate(current_cluster.iterrows()):
            _ax[index].plot(row.iloc[1:-1])
            _ax[index].set_title(f'{row.well}\n{row.cluster}')
            plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()  # The figure size is determined by the number of subplots
    for _cluster in range(n_clusters):
        print(f'Cluster number: {_cluster}')
    # Iterate over the clusters
        plot_cluster_tickers(df_transformed[df_transformed.cluster == _cluster].reset_index())  # Rotates the x-axis labels 45 degrees for better visibility  # Reshape the axis object to a 1D array, so we can iterate over it  # Iterate over the rows of the dataframe  # Plot the row data, excluding the first and last columns  # Set the title for each subplot  # Rotate the x-axis labels 90 degrees for better visibility  # Adjusts subplot params so that subplots are nicely fit in the figure  # Display the figure  # For each cluster, filter the dataframe for rows that belong to the current cluster, reset the index, and plot the tickers
    return


@app.cell
def _(df_transformed):
    df_transformed
    return


@app.cell
def _(df_transformed):
    # Reset the index of the DataFrame `df_transformed`
    # After resetting, the index becomes a new column in the DataFrame
    df_transformed_reset = df_transformed.reset_index()

    # Select the "well" and "cluster" columns from the DataFrame `df_transformed_reset`
    wells_and_clusters = df_transformed_reset[["well", "cluster"]]
    wells_and_clusters
    return (wells_and_clusters,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Визуализуруем полученные профили добычи
    """)
    return


@app.cell
def _(column_1, df, plt, wells_and_clusters):
    ymin = df[column_1].min()
    ymax = df[column_1].max()
    plt.figure(figsize=(15, 10))
    plt.suptitle('WOPR in the clusters')
    nrows = 3
    ncols = 1
    index = 1
    for _cluster in wells_and_clusters['cluster'].unique():
        for _well in wells_and_clusters[wells_and_clusters['cluster'] == _cluster]['well'].unique():
            plt.subplot(nrows, ncols, index)
            well_data = df[df['well'] == _well]
            plt.plot(well_data['date'], well_data[column_1], label=_well)
            plt.ylim(ymin, ymax)
        plt.title(_cluster)
        plt.legend(loc='lower left')
        plt.grid()
        index = index + 1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Какие скважины выбиваются из общего паттерна кластера?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Загрузим датафрейм с ккординатами скважин
    """)
    return


@app.cell
def _(pd):
    _path = 'data/'
    data_1 = pd.read_excel(f'{_path}coordinates.xlsx')
    data_1['well'] = data_1['well'].astype('str')
    data_1['well'] = data_1['well'].str.lower()
    data_1
    return (data_1,)


@app.cell
def _(wells_and_clusters):
    wells_and_clusters["well"] = wells_and_clusters["well"].astype(str)

    wells_and_clusters
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Объединим 2 датафрейма на колнку 'Well'
    """)
    return


@app.cell
def _(data_1, wells_and_clusters):
    data_2 = data_1.merge(wells_and_clusters, on='well', how='left')
    data_2
    return (data_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Визуализируем карту с полученными кластерами
    """)
    return


@app.cell
def _(np, plt):
    def discrete_cmap(N, base_cmap=None):
        """
        Create an N-bin discrete colormap from the specified input map

        Parameters
        ----------
        N : int
            The number of colors needed in the colormap.
        base_cmap : str, optional
            The base colormap to use. This should be a name of a colormap recognized by matplotlib.

        Returns
        -------
        newmap : Colormap
            A new colormap with N colors spanning the range of base_cmap.
        """
        base = plt.cm.get_cmap(base_cmap)
        color_list = base(np.linspace(0, 1, N))
        cmap_name = base.name + str(N)
        return base.from_list(cmap_name, color_list, N)

    def voronoi_finite_polygons_2d(vor, radius=None):
        """
        Reconstruct infinite Voronoi regions in a 2D diagram to finite
        regions.

        Parameters
        ----------
        vor : scipy.spatial.Voronoi instance
            The input Voronoi diagram.
        radius : float, optional
            The distance to 'points at infinity'. If not provided, the maximum point-to-point
            distance within the input points is used.

        Returns
        -------
        regions : list of lists
            A list of regions of the Voronoi diagram. Each region is a list of indices of the
            Voronoi vertices forming the region.
        vertices : ndarray
            The coordinates of the Voronoi vertices.
        """
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
            vertices = vor.regions[region]  # Rest of the function remains unchanged
            if all((v >= 0 for v in vertices)):
                new_regions.append(vertices)
                continue
            ridges = all_ridges[p1]
            new_region = [v for v in vertices if v >= 0]
            for p2, v1, v2 in ridges:  # Reconstruct infinite regions
                if v2 < 0:
                    v1, v2 = (v2, v1)
                if v1 >= 0:
                    continue
                t = vor.points[p2] - vor.points[p1]  # finite region
                t = t / np.linalg.norm(t)
                n = np.array([-t[1], t[0]])
                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n  # Reconstruct a non-finite region
                far_point = vor.vertices[v2] + direction * radius
                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())
            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]
            new_regions.append(new_region.tolist())  # Finite ridge: already in the region
        return (new_regions, np.asarray(new_vertices))  # Compute the missing endpoint of an infinite ridge  # Tangent  # Normal  # Sort region counterclockwise
    return discrete_cmap, voronoi_finite_polygons_2d


@app.cell
def _(
    Voronoi,
    data_2,
    discrete_cmap,
    matplotlib,
    np,
    plt,
    voronoi_finite_polygons_2d,
):
    fig = plt.figure(figsize=(15, 5))
    _ax = fig.gca()
    number_of_clusters = len(data_2['cluster'].unique())
    coordinates = np.array(data_2[['X', 'Y']])
    points = coordinates
    vor = Voronoi(points)
    regions, vertices = voronoi_finite_polygons_2d(vor)
    minima = min(data_2['cluster'])
    maxima = max(data_2['cluster'])
    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.Blues)
    count = 0
    for _i in regions:
        points = vertices[_i]
        x = points[:, 0]
        y = points[:, 1]
        x = np.append(x, x[0])
        y = np.append(y, y[0])
        plt.plot(x, y, 'k', linewidth=0.5)
        plt.fill(*zip(*points), alpha=0.75, color=mapper.to_rgba(data_2['cluster'][count]))
        count = count + 1
    plt.plot(x, y, 'k', linewidth=0.5, label='Well boundary')
    plt.scatter(points[:, 0], points[:, 1], color='black', s=3)
    with open('data/boundary', 'r') as file:
        polygon_boundary = np.array([list(map(float, line.split())) for line in file])
    plt.plot(*polygon_boundary.T, color='red', linewidth=0.9, label='Model boundary')
    plt.scatter(data_2.X, data_2.Y, cmap=discrete_cmap(number_of_clusters, 'Blues'), c=data_2['cluster'], marker='o', lw=0.5, edgecolors='black', s=10, alpha=0.75, label='Well')
    for label, x, y in zip(data_2['well'], data_2['X'], data_2['Y']):
        plt.annotate(label, xy=(x, y), size=15)
    plt.axis('equal')
    plt.xlim(1295000, 1372000)
    plt.ylim(275000, 298000)
    plt.yticks(np.arange(275000, 298000, 5000))
    plt.xticks(np.arange(1295000, 1372000, 5000))
    _ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    _ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.xticks(rotation=45)
    plt.colorbar(ticks=range(number_of_clusters))
    legend = plt.legend(loc='upper right', shadow=False, fontsize='large')
    legend.get_frame()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Задание:
    - Прокластеризуйте временные ряды датасета СOSTA (https://www.researchgate.net/publication/358582459_The_Design_of_an_Open-Source_Carbonate_Reservoir_Model, https://www.researchgate.net/publication/358903422_An_Open_Access_Carbonate_Reservoir_Model, https://researchportal.hw.ac.uk/en/datasets/costa-model-hierarchical-carbonate-reservoir-benchmarking-case-st) используя подхъходящий метод. Вы не ограничены DTW. Можете, например, представить одно или многомерный сигнал как разложение Фурье или набор признаков (TSFresh)
    - Попробуйте определить оптимальное количество кластеров. Какой метод оценки вы выбрали для этого типа данных и почему?
    - Визуализируйте кластера на карте
    - Как пространственно меняются кластера по площади месторождения?
    - Какой метод кластеризации выдает наиболее устойчивый результат при использовании данных различной природы?
    - Какой метод выдает объекты кластеров которые наиболее сосредоточены рядом друг с другом на карте? Как можно количественно оценить это?
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
