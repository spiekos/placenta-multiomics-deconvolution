#!/usr/bin/env python3
'''
File: merged_datatype_csv_files.py
Author: Samantha Piekos
Date: 05/16/25
Description: This takes in a list of datafiles and then merges on a specified
column that is the same across all the files to generate a merged csv, which
is saved to provided output filepath. Data in the merged file is limited to
patients who are represented in all of the provided input files.

@output_file 		filepath to save formatted CSV file.
@input_file_list	anything in the second position onwards are filepaths to files
					that you want to merge into one formatted file limited to 
					patients whose data is represented across the files
'''

# Set up environment
import os
import pandas as pd
import sys

# Declare constants
MERGE_COL = 'Patient-ID'


def read_data_to_pd(merge_col, input_file_list):
	# read in data to pandas df
	merged_df = pd.read_csv(input_file_list[0])

	# merge all remaining dfs on merge column
	for file in input_file_list[1:]:
		df_additional = pd.read_csv(file)
		merged_df = merged_df.merge(df_additional, on=merge_col, how='inner')

	return merged_df


def limit_datasets_to_union(merged_df, merge_col, input_file_list):
	# for each individual file limit to only rows present in merged df
	for file in input_file_list:
		df = pd.read_csv(file)

		# limit the individual datatype to those rows in merged_df
		limited_df = df[df["Patient-ID"].isin(merged_df[merge_col])]

		# generate output filepath and save limited df
		output_filepath = file[:-4] + '-limited.csv'
		limited_df.to_csv(output_filepath, index=False) 

	return


# Main script entry point
def main():
	output_file = sys.argv[1]
	input_file_list = []
	for arg in sys.argv[2:]:
		input_file_list.append(arg)
	merged_df = read_data_to_pd(MERGE_COL, input_file_list)
	merged_df.to_csv(output_file, index=False) # write merged and inputed df to output file
	limit_datasets_to_union(merged_df, MERGE_COL, input_file_list)


# Run the script if executed from the command line
if __name__ == "__main__":
    main()
