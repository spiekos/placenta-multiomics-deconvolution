#!/usr/bin/env python3
'''
File: outlier_clustering.py
Author: Samantha Piekos
Date: 05/08/25
Last Edited: 05/15/25
Description: This performs k-means clustering of patients based on analyte
values taken as an input CSV file. This can be include normalized analyte 
values or outlier status (-1, 0, 1). It can be any single type of multiomics
data or concatenated across types. If it does use multiomics data then early
fusion is used.

First heirarchical clustering is performed using Jaccard distance. Than the 
first n-dimensions of PCA of the input data that account for 95% of the
variance in the data is determined. k is determined using the elbow and
silhouette methods. k-means clustering is performed on dimensionality reduced
(i.e. the first n dimensions accounting for 95% variance in the data). This is
visualized in 2D and 3D plots representing the first two or three principal
components respectively. K is set by the user as they personally determine from
using the methods described above. Which cluster each patient is assigned is
recorded and saved to an output CSV file. For each cluster the number of patients
is printed. If the file provided is an outlier file than the median (IQR) outliers, 
and the analytes that are declared an outlier in X (specified as universal variable)
proportion of patients in the cluster are printed as well. A Kruskal-Wallis H-test
is used to determine if there is a statistically different number of outliers
between clusters. The number of patients belonging to each obstetric syndrome in
each cluster is plotted in a stacked bar chart. Also, the proportion of the
patients belonging to each obstetric syndrome is plotted as a stacked bar plot.
A chi-square test is used to evaluate if there is  statistical difference between 
the obstetric syndrome composition between the different clusters.

Parameters:
    input_file (CSV): Input CSV file with patients as rows and analyte
        values or outlier status as the columns.
    prefix (str): Describes what is being used as the input file and will
        be used as a prefix to generate file names of output files.
    k (int; k=5; optional): The number of clusters used in the k-means
        clustering analysis. Optional variable with the default value set
        to 5. Must be set using the 'k' flag (i.e. k=5).
    outlier (boolean; outlier=True; optional): whether or not the input_file
        are outlier values.

Returns:
    heirarchical_clustering_pdf (PDF): PDF of dendogram of heirarchical
        clustering performed using Jaccard distance saved to the output 
        directory.
    pca_component_pdf (PDF): PDF of number of plot of PCA components
        vs variance saved to the output directory.
    elbow_method_pdf (PDF): PDF of the elbow method plot for selecting
        k for k-means clustering saved to the output directory.
    silhouette_pdf (PDF): PDF of the silhouette method plot for
        selecting k for k-means clustering saved to the output directory.
    2d_k_means_clustering (PDF): PDF of the plot of each patient using
        dimensionality reduced data on the first two PCA dimensions with
        the cluster each patient belongs to (determined by k-means)
        represented by the dot color saved to the output directory.
    3d_k_means_clustering (PDF): PDF of the plot of each patient using
        dimensionality reduced data on the first three PCA dimensions with
        the cluster each patient belongs to (determined by k-means)
        represented by the dot color saved to the output directory.
    cluster_stacked_bar_graph (PDF): PDF fo the stacked bar graph of the
        number of patients that have the different obstetric syndromes in
        each k-means cluster saved to the output directory.
    cluster_proportion_stacked_bar_graph (PDF): PDF fo the stacked bar graph
        of  the proportion of each k-means cluster that is composed of each 
        obstetric syndrome saved to the output directory.
    output_file (CSV): CSV file of the cluster to which each patient was
        assigned using k-means clustering saved to the output directory.
'''

# Set up Environment
from k_means_constrained import KMeansConstrained
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from scipy.stats import chi2_contingency, fisher_exact, kruskal
from sklearn.metrics import silhouette_score
import sys


# Declare Constants
VAR_THRESHOLD = 0.95     # cumulative variance threshold for PCA
OUTLIER_MEAN_CUTOFF = 0.8
MAX_PCS = 150            # maximum number of PCs to consider
OUTPUT_DIR = './output/'
CLUSTER_MIN_SIZE=10
COLOR_DICT = {
	'Control': 'black',
	'FGR': 'cornflowerblue',
	'FGR+HDP': 'mediumorchid',
	'PE': 'pink',
	'PTD': 'goldenrod'
}


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def make_dataframes(file_input):
    """
    Read CSV and split into:
      • df_cohort: first column (e.g. metadata, Condition label)
      • df_outliers: all other numeric features
    """
    df = pd.read_csv(file_input, header=0, index_col=0)
    df_cohort = df.iloc[:, :1]
    df_outliers = df.drop(columns=['Condition'], errors='ignore')
    return df_cohort, df_outliers


def select_n_components(df_outliers, prefix, var_threshold=VAR_THRESHOLD, max_pcs=MAX_PCS):
    """
    Perform PCA, plot cumulative explained variance up to max_pcs,
    print & return the number of components needed to exceed var_threshold.
    """
    pca = PCA().fit(df_outliers)
    n_features = df_outliers.shape[1]
    max_comps = min(max_pcs, n_features)

    # cumulative variance
    xi = np.arange(1, max_comps + 1)
    cum_var = np.cumsum(pca.explained_variance_ratio_[:max_comps])

    # number of components needed
    n_components = np.argmax(cum_var >= var_threshold) + 1
    print(f"First component explaining ≥{var_threshold*100:.0f}% variance: {n_components}")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(xi, cum_var, marker='o', linestyle='--')
    plt.axhline(y=var_threshold, color='r', linestyle='-')
    plt.text(0.5, var_threshold + 0.02,
             f'{int(var_threshold*100)}% threshold', color='red', fontsize=16)
    plt.ylim(0, 1.05)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Variance Explained')
    plt.title('PCA Components vs. Cumulative Variance - ' + prefix)
    plt.xticks(np.arange(0, max_comps + 1, step=5))
    plt.grid(axis='x')
    plot_path = os.path.join(OUTPUT_DIR, f"{prefix}_n_components_vs_variance.pdf")
    plt.savefig(plot_path)
    plt.show()

    return n_components


def elbow_method(df_reduced, prefix):
    """
    Elbow plot of k-means inertia for k=2..15.
    """
    ssd = []
    K = range(2, 16)
    for k in K:
        km = KMeans(n_clusters=k, random_state=123).fit(df_reduced)
        ssd.append(km.inertia_)

    plt.figure()
    plt.plot(K, ssd, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum of Squared Distances')
    plt.title('Elbow Method For Optimal k - ' + prefix)
    plot_path = os.path.join(OUTPUT_DIR, f"{prefix}_Elbow_Method.pdf")
    plt.savefig(plot_path)
    plt.show()


def silhouette_method(df_reduced, prefix):
    """
    Silhouette score plot for k=2..15.
    """
    sil_scores = []
    K = range(2, 16)
    for k in K:
        labels = KMeans(n_clusters=k, random_state=123).fit_predict(df_reduced)
        sil_scores.append(silhouette_score(df_reduced, labels))

    plt.figure()
    plt.plot(K, sil_scores, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Method For Optimal k - ' + prefix)
    plot_path = os.path.join(OUTPUT_DIR, f"{prefix}_Silhouette_Method.pdf")
    plt.savefig(plot_path)
    plt.show()


def select_k(df, n, prefix):
    """
    Reduce data to n PCs and run elbow & silhouette analyses.
    """
    df_reduced = PCA(n_components=n, random_state=123).fit_transform(df)
    elbow_method(df_reduced, prefix)
    silhouette_method(df_reduced, prefix)


def get_variance_explained(df):
    """Print variance explained by PC1 and PC2."""
    pca2 = PCA(n_components=2, random_state=123).fit(df)
    ratios = pca2.explained_variance_ratio_
    print('Variance Explained: PC1={:.2f}%, PC2={:.2f}%'.format(ratios[0]*100, ratios[1]*100))
    return round(ratios[0]*100, 1), round(ratios[1]*100, 1)


def get_variance_explained_3D(df):
    """Print variance explained by PC1 and PC2."""
    pca3 = PCA(n_components=3, random_state=123).fit(df)
    ratios = pca3.explained_variance_ratio_
    print('Variance Explained: PC1={:.2f}%, PC2={:.2f}%, PC3={:.2f}%'.format(ratios[0]*100, ratios[1]*100, ratios[2]*100))
    return round(ratios[0]*100, 1), round(ratios[1]*100, 1), round(ratios[2]*100, 1)


def perform_k_means_clustering(df, n, k, prefix):
    """
    Run k-means on n-component PCA data, plot clusters in 2D PC space.
    """
    df_reduced_n = PCA(n_components=n, random_state=123).fit_transform(df)
    #kmeans = KMeans(n_clusters=k, random_state=123).fit(df_reduced_n)
    #labels = kmeans.labels_
    kmc = KMeansConstrained(
    	n_clusters=k,
    	size_min=CLUSTER_MIN_SIZE,
    	size_max=None,
    	random_state=123
    	)
    labels = kmc.fit_predict(df_reduced_n) 

    pc1, pc2 = get_variance_explained(df)

    # 2D visualization
    df_reduced_2 = PCA(n_components=2, random_state=123).fit_transform(df)
    plt.figure()
    plt.scatter(df_reduced_2[:, 0], df_reduced_2[:, 1], c=labels, s=100, cmap='viridis')
    #centers = kmeans.cluster_centers_
    #plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.xlabel('PC1 - ' + str(pc1) + '%')
    plt.ylabel('PC2 - ' + str(pc2) + '%')
    plot_path = os.path.join(OUTPUT_DIR, f"{prefix}_PCA_KMeans.pdf")
    plt.savefig(plot_path)
    plt.show()

    return labels


def perform_k_means_clustering_3D(df, n, k, prefix):
    """
    Run k-means on n-component PCA data, plot clusters in 3D PC space.
    """
    # Reduce to n components for clustering
    df_reduced_n = PCA(n_components=n, random_state=123).fit_transform(df)
    #kmeans = KMeans(n_clusters=k, random_state=123).fit(df_reduced_n)
    #labels = kmeans.labels_
    kmc = KMeansConstrained(
    	n_clusters=k,
    	size_min=CLUSTER_MIN_SIZE,
    	size_max=None,
    	random_state=123
    	)
    labels = kmc.fit_predict(df_reduced_n) 

    pc1, pc2, pc3 = get_variance_explained_3D(df)

    # Reduce to 3 components for visualization
    pca_3d = PCA(n_components=3, random_state=123).fit(df)
    df_reduced_3 = pca_3d.transform(df)
    #cluster_centers_3d = pca_3d.transform(kmeans.cluster_centers_)

    # 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(df_reduced_3[:, 0], df_reduced_3[:, 1], df_reduced_3[:, 2],
    	c=labels, cmap='viridis', s=100)
    # ax.scatter(cluster_centers_3d[:, 0], cluster_centers_3d[:, 1], cluster_centers_3d[:, 2],
	#            c='black', s=200, alpha=0.5)
    ax.set_xlabel('PC1 - ' + str(pc1) + '%')
    ax.set_ylabel('PC2 - ' + str(pc2) + '%')
    ax.set_zlabel('PC3 - ' + str(pc3) + '%')
    ax.set_title('K-Means Clustering - ' + prefix)
    plot_path = os.path.join(OUTPUT_DIR, f"{prefix}_PCA_KMeans_3D.pdf")
    plt.savefig(plot_path)
    plt.show()

    return labels


def make_cohort_dict(labels_list):
    """Count occurrences in a list."""
    return {lbl: labels_list.count(lbl) for lbl in set(labels_list)}


def evaluate_clusters(df, k, prefix):
    """Wrapper function: prints cluster summary, runs Kruskal-Wallis, and plots."""
    cluster_outliers = print_cluster_summary(df, k)
    run_kruskal_test_on_outliers(cluster_outliers)
    return


def print_cluster_summary(df, k):
    """Print median ± IQR for outliers and condition breakdown for each cluster."""
    total_counts = make_cohort_dict(list(df['Condition']))
    df = df.apply(lambda x: x.abs() if np.issubdtype(x.dtype, np.number) else x)  # work with binary outliers
    
    all_cluster_outliers = []

    print("=== Cluster Summary ===")
    for cluster_id in range(k):
        print(f"\nStats for Cluster {cluster_id}")
        sub = df[df['Label'] == cluster_id]
        print(f"Number of Patients in Cluster: {len(sub)}")

        outliers = sub['n_outliers']
        med = np.median(outliers)
        q1 = np.percentile(outliers, 25)
        q3 = np.percentile(outliers, 75)
        iqr = q3 - q1
        print(f"Outliers: {med:.2f} ± {iqr:.2f} (median ± IQR)")

        all_cluster_outliers.append(list(outliers))

        counts = make_cohort_dict(list(sub['Condition']))
        for cond, cnt in counts.items():
            pct = cnt * 100 / total_counts.get(cond, 1)
            print(f"{cond}: {cnt} ({pct:.1f}%)")

    return all_cluster_outliers


def run_kruskal_test_on_outliers(cluster_outliers):
    """Run Kruskal-Wallis H-test on n_outliers across clusters."""
    print("\n=== Kruskal-Wallis Test for 'n_outliers' by Cluster ===")
    valid_groups = [group for group in cluster_outliers if len(group) > 1]
    if len(valid_groups) < 2:
        print("Not enough clusters with >1 sample to perform Kruskal-Wallis test.")
    else:
        stat, p_value = kruskal(*valid_groups)
        print(f"Kruskal-Wallis H-statistic: {stat:.4f}, p-value: {p_value:.4g}")
    return


def identify_average_outliers(df, k, cutoff=OUTLIER_MEAN_CUTOFF):
    """List features with mean > cutoff for each cluster."""
    df = df.apply(lambda x: x.abs() if np.issubdtype(x.dtype, np.number) else x)  # work with binary outliers
    for cluster_id in range(k):
        print(f"Outliers Represented In >={cutoff} In Cluster {cluster_id}:")
        sub = df[df['Label'] == cluster_id].drop(columns=['Label'], errors='ignore')
        high = [col for col in sub if sub[col].mean() >= cutoff]
        print('\n'.join(high) or 'None')
        print()
    return


def perform_hierarchical_clustering(df, prefix):
    """Perform hierarchical clustering using Jaccard distance and save dendrogram."""
    binary_df = df.apply(lambda x: x.abs() if np.issubdtype(x.dtype, np.number) else x)  # work with binary outliers

    # Compute pairwise Jaccard distance
    distance_matrix = pdist(binary_df, metric='jaccard')

    # Hierarchical clustering using average linkage
    linked = linkage(distance_matrix, method='average')

    # Plot dendrogram with patient labels
    plt.figure(figsize=(25, 10))  # wider to fit all patient IDs
    dendrogram(linked,
    	labels=df.index.tolist(),
    	leaf_rotation=90,           # Rotate labels for readability
    	leaf_font_size=6,           # Shrink font size to fit
    	orientation='top',
    	distance_sort='descending',
    	show_leaf_counts=True)

    plt.title(f"{prefix} Hierarchical Clustering (Average - Jaccard)")
    plt.xlabel("Patient ID")
    plt.ylabel("Jaccard Distance")

    # Save the figure
    plot_path = os.path.join(OUTPUT_DIR, f"{prefix}_Hierarchical_Clustering.pdf")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.show()
    return


def plot_cluster_composition(df, prefix):
    """Generate stacked bar plots showing condition counts per cluster."""
    cluster_labels = sorted(df['Label'].unique())
    condition_labels = sorted(df['Condition'].unique())
    composition = pd.DataFrame(0, index=cluster_labels, columns=condition_labels, dtype=int)

    for cluster in cluster_labels:
        sub = df[df['Label'] == cluster]
        counts = sub['Condition'].value_counts()
        for cond in counts.index:
            composition.loc[cluster, cond] = counts[cond]

    colors = [COLOR_DICT.get(cond, '#cccccc') for cond in composition.columns]  # fallback gray

    composition.plot(kind='bar', stacked=True, figsize=(10, 6), color=colors)
    plt.ylabel('Number of Patients')
    plt.title('Cluster Composition by Condition')
    plt.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plot_path = os.path.join(OUTPUT_DIR, f"{prefix}_Cluster_StackedBar.pdf")
    plt.savefig(plot_path)
    plt.show()

    return


def plot_cluster_composition_proportion(df, prefix):
    """Generate stacked bar plots showing condition composition per cluster."""
    cluster_labels = sorted(df['Label'].unique())
    condition_labels = sorted(df['Condition'].unique())
    composition = pd.DataFrame(0.0, index=cluster_labels, columns=condition_labels, dtype=float)

    for cluster in cluster_labels:
        sub = df[df['Label'] == cluster]
        counts = sub['Condition'].value_counts(normalize=True)
        for cond in counts.index:
            composition.loc[cluster, cond] = counts[cond]

    colors = [COLOR_DICT.get(cond, '#cccccc') for cond in composition.columns]  # fallback gray

    composition.plot(kind='bar', stacked=True, figsize=(10, 6), color=colors)
    plt.ylabel('Proportion')
    plt.title('Cluster Composition by Condition')
    plt.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f"{prefix}_Cluster_By_Proportion_Of_Condition_StackedBar.pdf")
    plt.savefig(plot_path)
    plt.show()

    # Run and print statistical test
    test_distribution_of_cohorts_between_clusters(df)

    return


def test_distribution_of_cohorts_between_clusters(df):
    """Test if the distribution of cohorts (conditions) differs significantly between clusters using Chi-square test."""
    contingency = pd.crosstab(df['Label'], df['Condition'])
    stat, pval, dof, expected = chi2_contingency(contingency)

    print("\nChi-Square Test: Are clusters composed of significantly different proportions of cohorts?")
    print(f"Chi2 Statistic = {stat:.4f}, p-value = {pval:.4g}, Degrees of freedom = {dof}")
    if pval < 0.05:
        print("→ Significant difference in cohort composition between clusters.\n")
    else:
        print("→ No significant difference in cohort composition between clusters.\n")
    return


def main():
    ensure_output_dir()
    if len(sys.argv) < 3:
        print("Usage: script.py <input.csv> <output_prefix> [k=N]")
        sys.exit(1)

    file_input = sys.argv[1]
    prefix = sys.argv[2]
    k = 5
    outlier_file = True
    # parse optional arguments
    if len(sys.argv) > 3:
        for arg in sys.argv[3:]:
            if arg.startswith('k='):
                try:
                    k = int(arg.split('=')[1])
                except ValueError:
                    pass
            elif arg.startswith('outlier_file='):
                try:
                    outlier_file = (arg.split('=')[1])
                except ValueError:
                    pass

    df_cohort, df_outliers = make_dataframes(file_input)
    n = select_n_components(df_outliers, prefix)

    #df_reduced = PCA(n_components=n, random_state=123).fit_transform(df_outliers)
    perform_hierarchical_clustering(df_outliers, prefix)
    select_k(df_outliers, n, prefix)

    labels = perform_k_means_clustering(df_outliers, n, k, prefix)
    labels = perform_k_means_clustering_3D(df_outliers, n, k, prefix)
    print('\n')
    df_cohort['Label'] = labels
    df_cohort['n_outliers'] = df_outliers.sum(axis=1)
    df_cohort.sort_values(['Label', 'Condition', 'n_outliers'], inplace=True)
    df_cohort.to_csv(os.path.join(OUTPUT_DIR, f"{prefix}_kMeans_Clusters.csv"))
    df_outliers['Label'] = labels
    if outlier_file:
        evaluate_clusters(df_cohort, k, prefix)
        identify_average_outliers(df_outliers, k)
    plot_cluster_composition(df_cohort, prefix)
    plot_cluster_composition_proportion(df_cohort, prefix)


if __name__ == "__main__":
    main()
