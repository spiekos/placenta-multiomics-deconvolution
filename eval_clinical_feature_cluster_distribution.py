# Set up environment
import numpy as np
import pandas as pd
from scipy.stats import kruskal
import sys

# declare universal variables
MERGE_COL = 'Patient-ID'
CLUSTER_COL = 'Label'

def read_data_to_pd(file_metadata, file_input, feature_col, merge_col):
    # read in data to pandas df
    df_1 = pd.read_csv(file_metadata)
    df_2 = pd.read_csv(file_input)

        # Merge the Condition from df_1 into df_2 using 'Patient-ID' as the key
    merged_df = df_1[[merge_col, feature_col]].merge(df_2, on=merge_col, how='inner')

    return merged_df


def kruskal_test_by_label_cont(df, feature_col, cluster_col):
    # Assumes the feature is continuous
    # Group the WksGest values by Label
    grouped = df.groupby(cluster_col)[feature_col].apply(list)
    
    # Print median ± IQR for each group
    print("Median ± IQR of", feature_col, "for each cluster:")
    for label, values in grouped.items():
        values_array = np.array(values)
        median = np.median(values_array)
        q1 = np.percentile(values_array, 25)
        q3 = np.percentile(values_array, 75)
        iqr = q3 - q1
        print(f"  Label {label}: {median:.2f} ± {iqr:.2f}")
    
    # Only keep groups with more than one observation
    valid_groups = [group for group in grouped if len(group) > 1]

    # Ensure at least two groups to compare
    if len(valid_groups) < 2:
        raise ValueError("Need at least two groups with more than one observation to perform Kruskal-Wallis test.")
    
    # Perform Kruskal-Wallis test
    stat, p_value = kruskal(*valid_groups)
    
    print(f"\nKruskal-Wallis H-statistic: {stat:.4f}, p-value: {p_value:.4g}")
    return


# Main script entry point
def main():
    # parse arguments
    metadata_file = sys.argv[1]
    input_file = sys.argv[2]
    feature_col = sys.argv[3]
    analyte_type = 'Analyte'
    if len(sys.argv) > 4 and sys.argv[4].startswith('type='):
        try:
            analyte_type = sys.argv[4].split('=')[1]
        except ValueError:
            pass
    df = read_data_to_pd(metadata_file, input_file, feature_col, MERGE_COL)
    kruskal_test_by_label_cont(df, feature_col, CLUSTER_COL)


# Run the script if executed from the command line
if __name__ == "__main__":
    main()