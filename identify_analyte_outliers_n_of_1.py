#!/usr/bin/env python3
'''
File: identify_analyte_outliers_n_of_1.py
Author: Samantha Piekos
Date: 05/16/25
Description: 

@metadata_file          Filepath to the metadata for the patients.
@input_file             Filepath to the omics file of interest to perform
                        outlier analysis on.
@imputed_output_file    Filepath to save the updated omics file to replace
                        missing values with 1/2 min imputation by condition.
@normalized_output_file Filepath to save the updated omics file that has been
                        imputed and normalized on the control cohort median
                        and MAD for each analyte.
@outlier_output_file    
@type                   Optional argument whose default value is "Analyte".
                        This is used to generate plot titles and plots save
                        filepaths.
Returns:
outlier_plot            Violin and box plot PDF of the number of outliers each
                        patient has by condition
'''

# Set up environment
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statistics
import sys


# Declare constants
CONDITION_COL = 'Condition'
CONTROL_GROUP = 'Control'
MAD_THRESHOLD = 3
MERGE_COL = 'Patient-ID'
OUTLIER_PROPORTION_CUTOFF = 0.25
COLOR_DICT = {
	'Control': 'black',
	'FGR': 'cornflowerblue',
	'FGR+HDP': 'mediumorchid',
	'PE': 'pink',
	'PTD': 'goldenrod'
}
RENAME_COHORTS_DICT = {
	'Control': 'Control',
	'FGR': 'FGR',
	'FGR+hyp': 'FGR+HDP',
	'PTD': 'PTD',
	'Severe PE': 'PE'
}


def read_data_to_pd(file_metadata, file_input, condition_col, merge_col):
	# read in data to pandas df
	df_1 = pd.read_csv(file_metadata)
	df_2 = pd.read_csv(file_input)

	# rename cohorts
	df_1[condition_col] = df_1[condition_col].replace(RENAME_COHORTS_DICT)

	# Merge the Condition from df_1 into df_2 using 'Patient-ID' as the key
	merged_df = df_1[[merge_col, condition_col]].merge(df_2, on=merge_col, how='inner')

	# perform median imputation on missing values by condition
	final_df = impute_by_condition(merged_df, condition_col, imputation_type='half_min')

	# Count the number belonging into each cohort:
	print('# Cohort Sizes:')
	for condition in set(final_df[condition_col]):
		condition_df =  final_df[final_df[condition_col] == condition]
		n = len(condition_df)
		print('# ' + condition + ': ' + str(n))

	return final_df


def impute_by_condition(df, condition_col, imputation_type=min):
    """
    Perform imputation for each numeric column within each condition group.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing missing values.
        condition_col (str): Name of the column specifying the condition to group by.
        imputation_type: Type of imputation to perform. Default value is min.
                        Other excepted values are median and half_min

    Returns:
        pd.DataFrame: A new DataFrame with missing values imputed.
    """
    df_imputed = df.copy()
    numeric_cols = df.select_dtypes(include='number').columns

    for col in numeric_cols:
        if col != condition_col:
            if imputation_type == 'min':
                df_imputed[col] = df.groupby(condition_col)[col].transform(
                    lambda x: x.fillna(x.min())
                )
            elif imputation_type == 'median':
                df_imputed[col] = df.groupby(condition_col)[col].transform(
                    lambda x: x.fillna(x.median())
                )
            elif imputation_type == 'half_min':
                df_imputed[col] = df.groupby(condition_col)[col].transform(
                    lambda x: x.fillna(x.min()/2)
                )
            else:
                df_imputed[col] = df.groupby(condition_col)[col].transform(
                    lambda x: x.fillna(x.min())
                )

    return df_imputed


def normalize_and_declare_directional_outliers(
    df, control_group='Control', skip_columns=['Patient-ID', 'Condition'], mad_threshold=3.5
):
    """
    Normalize data based on the control group and declare directional outliers 
    using a modified z-score based on median and MAD.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with numeric values and metadata columns.
        control_group (str): Label of the control group in the 'Condition' column.
        skip_columns (list): Columns to exclude from normalization and outlier detection.
        mad_threshold (float): Threshold for declaring outliers.

    Returns:
        normalized_df (pd.DataFrame): DataFrame with normalized values.
        directional_outlier_df (pd.DataFrame): DataFrame with -1 (low), 0 (normal), 1 (high) per analyte.
    """
    control_mask = df['Condition'] == control_group
    normalized_df = df.copy()

    # List to hold directional outlier columns
    directional_outlier_cols = []

    for col in df.columns:
        if col not in skip_columns and np.issubdtype(df[col].dtype, np.number):
            control_values = df.loc[control_mask, col].dropna()
            median = np.median(control_values)
            mad = stats.median_abs_deviation(control_values, scale='normal')

            # Normalize using median and MAD
            if mad == 0:
                normalized_col = df[col] - median
            else:
                normalized_col = (df[col] - median) / mad

            normalized_df[col] = normalized_col

            # Directional outlier flag: -1 (low), 0 (normal), 1 (high)
            directional_col = normalized_col.apply(
                lambda x: -1 if x < -mad_threshold else (1 if x > mad_threshold else 0)
            )
            directional_col.name = col
            directional_outlier_cols.append(directional_col)

    # Combine all directional outlier columns and restore metadata columns
    directional_outlier_df = pd.concat([df[skip_columns]] + directional_outlier_cols, axis=1)

    return normalized_df, directional_outlier_df


def print_extreme_outlier_patients(outlier_df, analyte_type, cutoff=0.5):
    """
    Prints patients with more than a given percentage of outlier analytes.
    Writes all patients proportion of outliers to output csv file.
    
    Parameters:
        outlier_df (pd.DataFrame): DataFrame where rows are patients and columns are analytes,
                                   with binary outlier flags (1 = outlier, 0 = not outlier).
                                   Includes 'Patient-ID' and 'Condition' columns.
        cutoff (float): Proportion threshold to consider a patient as having extreme outliers.

    Returns:
        Nothing
    """
    print(f'# Patients with more than {cutoff * 100:.0f}% outliers:')

    # Columns to exclude from analyte count
    exclude_cols = {'Patient-ID', 'Condition'}
    analyte_cols = [col for col in outlier_df.columns if col not in exclude_cols]
    n_analytes = len(analyte_cols)

    # open file to write proportion of outliers to
    w = open('./output/proportion_' + analyte_type.lower()  + '_outliers.csv', 'w')
    w.write('Patient-ID,Condition,Proportion-Outliers\n')

    for _, row in outlier_df.iterrows():
        outlier_fraction = round(row[analyte_cols].sum() / n_analytes, 3)
        output = [row['Patient-ID'], row['Condition'], str(outlier_fraction)]
        w.write(','.join(output))
        w.write('\n')
        if outlier_fraction > cutoff:
            print('#', row['Patient-ID'], row['Condition'], outlier_fraction)

    w.close()
    return


# Counts number of outlier values per sample and prints summary statistics
def analyze_outlier_stat_diff(outlier_df, drop_columns=['Patient-ID', 'Condition']):
    dict_n_outliers = {}
    
    print('\n# Median +/- IQR Number of Outliers per Cohort:')
    cohorts = outlier_df['Condition'].unique()
    for cohort in cohorts:
        cohort_outliers = list(outlier_df[outlier_df['Condition'] == cohort].drop(drop_columns, axis=1).sum(axis=1))
        dict_n_outliers[cohort] = cohort_outliers
        median = str(int(statistics.median(cohort_outliers)))
        iqr = str(stats.iqr(cohort_outliers))
        print(f'# {cohort}: {median} +/- {iqr}')
    print('\n')

    # check if there's a statistical difference in the number of outliers between cohorts
    calc_p_value(dict_n_outliers)

    return dict_n_outliers


# Performs t-tests between each cohort and control for outlier counts
def calc_p_value(dict_n_outliers, control_group=CONTROL_GROUP):
	cohorts_outliers_list = []
	#print('# Student Two-Tail T-Test p-Values:')
	print('# Student Mann-Whitney U-test p-Values:')

	for cohort, outliers in dict_n_outliers.items():
		cohorts_outliers_list.append(outliers)
		if cohort != CONTROL_GROUP:
			#p = stats.ttest_ind(dict_n_outliers[CONTROL_GROUP], outliers)[1] # t-test
			p = stats.mannwhitneyu(dict_n_outliers[CONTROL_GROUP], outliers, alternative='two-sided')[1]
			print(f'# {cohort}: {p}')

	p = stats.kruskal(*cohorts_outliers_list)[1]
	print('\n# Kruskal–Wallis Test p-value:', p)

	return


def make_n_outlier_violin_plot(dict_outliers, color_dict, analyte_type='Analyte'):
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams.update({'font.size': 14})
    
    fig, ax = plt.subplots()

    labels = list(dict_outliers.keys())
    labels.sort()  # Ensure consistent order

    values = [dict_outliers[label] for label in labels]
    colors = [color_dict[label] for label in labels]

    # Create violin plots manually, one at a time to apply colors
    for i, (v, c) in enumerate(zip(values, colors), start=1):
        parts = ax.violinplot([v], positions=[i], showmeans=False, showmedians=False, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor(c)
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)

    # Keep boxplot default (uncolored)
    ax.boxplot(values)

    # Axis labels and title
    ax.set_title(f'Number of {analyte_type} Outliers')
    ax.set_ylabel(f'Number of {analyte_type} Outliers')
    ax.set_xticks(range(1, len(labels)+1))
    ax.set_xticklabels(labels, rotation=15)

    plt.savefig(f'./output/{analyte_type}_Outliers_Violin_Box_Plot.pdf', bbox_inches='tight')
    plt.show()
    return


# Main script entry point
def main():
    # parse arguments
    metadata_file = sys.argv[1]
    input_file = sys.argv[2]
    imputed_output_file = sys.argv[3]
    normalized_output_file = sys.argv[4]
    outlier_output_file = sys.argv[5]
    analyte_type = 'Analyte'
    if len(sys.argv) > 6 and sys.argv[6].startswith('type='):
        try:
            analyte_type = sys.argv[6].split('=')[1]
        except ValueError:
            pass

    # read in data
    df = read_data_to_pd(metadata_file, input_file, CONDITION_COL, MERGE_COL)
    df.to_csv(imputed_output_file, index=False) # write imputed and formatted data to file
    # Count the number of analytes
    print('\n# Number of ' + analyte_type + 's: ' + str(df.shape[1]-2) + '\n')
    
    # mad normalize and declare outliers
    normalized_df, outlier_df = normalize_and_declare_directional_outliers(df)
    normalized_df.to_csv(normalized_output_file, index=False) # write normalized df to output
    outlier_df.to_csv(outlier_output_file, index=False) # write outlier df to output file

    # outlier_df to binary
    outlier_df.replace(-1, 1, inplace=True)
    
    # print patient ids for patient where more than half of their values are outliers
    print_extreme_outlier_patients(outlier_df, analyte_type, cutoff=OUTLIER_PROPORTION_CUTOFF)
    
    # analyze outlier distribution difference between cohorts
    dict_n_outliers = analyze_outlier_stat_diff(outlier_df)
    
    # Generate violin plot
    make_n_outlier_violin_plot(dict_n_outliers, color_dict=COLOR_DICT, analyte_type=analyte_type)


# Run the script if executed from the command line
if __name__ == "__main__":
    main()
