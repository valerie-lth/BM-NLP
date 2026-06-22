from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
import os
import re
import argparse
from tqdm import tqdm
from data_utils import (
    clean_label,
    mass_label,
    aggressive_label,
    met_label,
    clean_txt,
    findings_sect,
    impressions_sect,
    n_tokens,
    report_sections,
    background_sect,
)
import codecs
import random

def organize_weighted_labeled_data(df, met_type, report_path):
    '''
    df: Dataframe that contains the labels for each report (Column Names: File Name[ID_reportType1.txt], Mass, Aggressive) 
    met_info_file: csv file that contains the labels for Metastasis label 
    report_path: path for where the labeled radiology reports are kept 
    '''

     # since two files are combined together, there are two same columns named this

    df = df.fillna('NA')

    # clean the annotations
    df["Mass"] = df.Mass.apply(clean_label)
    # df["N_Mass"] = df.N_Mass.apply(clean_label)
    df["Aggressive"] = df.Aggressive.apply(clean_label)
    if met_type == 'string':
        df["met_label_x"] = df.met_label_x.apply(clean_label)

    # delete all unknowns and also likelys from the annotations df
    # Likelys were deleted because we decided that we only want to use yes/no for primary brain tumour
    # df = df[(df["N_Mass"] != "unknown") & (df["Mass"] != "likely")]
    print(df.shape)

    # generate the true labels
    df["mass_label"] = df.apply(lambda row: mass_label(row), axis=1)
    df["aggressive_label"] = df.apply(lambda row: aggressive_label(row), axis=1)
    df["met_label_x"] = df.apply(lambda row: met_label(row), axis=1)

    # retrieve the radiology report text
    files = os.listdir(report_path)

    # create two new columns to put the findings and impressions into it:
    df["full_report"] = ""
    df["findings"] = ""
    df["impressions"] = ""
    df["background"] = ""
    df["impressions_back"] = ""
    df["sections"] = ""
    df["findings_tokens"] = ""
    df["impressions_tokens"] = ""
    for file in files:
        # get the patient id first from the file name
        # always be the number before first underscore
        # pid = file[: file.find("_")]
        # open and read text file
        with codecs.open(
            os.path.join(report_path, file), "r", "utf-8", errors="replace"
        ) as f:
            text = f.read()
            # concatenate all of the words together and clean text
            full_report = clean_txt(text)
            # split the text into sections:
            findings = findings_sect(full_report)
            impressions = impressions_sect(full_report)
            background = background_sect(text) # we use the unclean text because I use the newlines
            # find number of tokens each section has:
            findings_tokens = n_tokens(findings)
            impressions_tokens = n_tokens(impressions)
            # determine which sections it has
            sections = report_sections(findings, impressions)
            # add this nospace text to the csv file:
            index = df[
                df["File Name"].str.contains(file)
            ].index  # assume that there is always only one index match...
            df.loc[index, "full_report"] = full_report
            df.loc[index, "findings"] = findings
            df.loc[index, "impressions"] = impressions
            df.loc[index, "background"] = background
            df.loc[index, "impressions_back"] = background + impressions
            df.loc[index, "sections"] = sections
            df.loc[index, "findings_tokens"] = findings_tokens
            df.loc[index, "impressions_tokens"] = impressions_tokens
    full_df = df
    # full_df.to_csv(met_info_file)
    print(
        "Shape of pre-process full_df_labeled: ", full_df.shape
    )  # should be 1059 -- this is after deleting the ones we need to verify for
    full_df = full_df[
        (full_df["sections"] == "Both")
        & (full_df["impressions_tokens"].apply(pd.to_numeric) <= 512)
    ]
    print("Shape of full_df_labeled: ", full_df.shape)
    return full_df

def organize_met_labeled_data(df, report_path):
    if 'Annotate_IR' in report_path:
        df = df.rename(columns={'Metastasis':'met_label_x'})
    else:
        df = df.rename(columns={'New_Met':'met_label_x'})
    
    print(df.columns)
    df["met_label_x"] = df['met_label_x'].apply(clean_label)
    df["met_label_x"] = df.apply(lambda row: met_label(row), axis=1)
    # retrieve the radiology report text
    files = os.listdir(report_path)

    # create two new columns to put the findings and impressions into it:
    df["full_report"] = ""
    df["findings"] = ""
    df["impressions"] = ""
    df["background"] = ""
    df["impressions_back"] = ""
    df["sections"] = ""
    df["findings_tokens"] = ""
    df["impressions_tokens"] = ""
    for file in files:
        # get the patient id first from the file name
        # always be the number before first underscore
        # pid = file[: file.find("_")]
        # open and read text file
        with codecs.open(
            os.path.join(report_path, file), "r", "utf-8", errors="replace"
        ) as f:
            text = f.read()
            # concatenate all of the words together and clean text
            full_report = clean_txt(text)
            # split the text into sections:
            findings = findings_sect(full_report)
            impressions = impressions_sect(full_report)
            background = background_sect(text) # we use the unclean text because I use the newlines
            # find number of tokens each section has:
            findings_tokens = n_tokens(findings)
            impressions_tokens = n_tokens(impressions)
            # determine which sections it has
            sections = report_sections(findings, impressions)
            # add this nospace text to the csv file:
            index = df[
                df["File Name"].str.contains(file)
            ].index  # assume that there is always only one index match...
            df.loc[index, "full_report"] = full_report
            df.loc[index, "findings"] = findings
            df.loc[index, "impressions"] = impressions
            df.loc[index, "background"] = background
            df.loc[index, "impressions_back"] = background + impressions
            df.loc[index, "sections"] = sections
            df.loc[index, "findings_tokens"] = findings_tokens
            df.loc[index, "impressions_tokens"] = impressions_tokens
    full_df = df
    full_df.to_csv('Data/patient_test.csv', index=False)
    # print("Shape of full_df_labeled (pre-filter): ", full_df.shape)
    # print("report with both sections: ", full_df[(full_df["sections"] == "Both")].shape)
    # full_df = full_df[
    #     (full_df["sections"] == "Both")
    #     # & (full_df["impressions_tokens"].apply(pd.to_numeric) <= 512)
    # ]
    print("Shape of full_df_labeled: ", full_df.shape)
    
    # full_df = full_df[(full_df["sections"] != "None")]
    
    
    return full_df

def organize_labeled_data(df, met_type, report_path):
    '''
    df: Dataframe that contains the labels for each report (Column Names: File Name[ID_reportType1.txt], Mass, Aggressive) 
    met_info_file: csv file that contains the labels for Metastasis label 
    report_path: path for where the labeled radiology reports are kept 
    '''

     # since two files are combined together, there are two same columns named this

    df = df.fillna('NA')

    # clean the annotations
    df["Mass"] = df.Mass.apply(clean_label)
    # df["N_Mass"] = df.N_Mass.apply(clean_label)
    df["Aggressive"] = df.Aggressive.apply(clean_label)
    if met_type == 'string':
        df["met_label_x"] = df.met_label_x.apply(clean_label)

    # delete all unknowns and also likelys from the annotations df
    # Likelys were deleted because we decided that we only want to use yes/no for primary brain tumour
    # df = df[(df["N_Mass"] != "unknown") & (df["Mass"] != "likely")]
    print(df.shape)

    # generate the true labels
    df["mass_label"] = df.apply(lambda row: mass_label(row), axis=1)
    df["aggressive_label"] = df.apply(lambda row: aggressive_label(row), axis=1)
    df["met_label_x"] = df.apply(lambda row: met_label(row), axis=1)

    # retrieve the radiology report text
    files = os.listdir(report_path)

    # create two new columns to put the findings and impressions into it:
    df["full_report"] = ""
    df["findings"] = ""
    df["impressions"] = ""
    df["background"] = ""
    df["impressions_back"] = ""
    df["sections"] = ""
    df["findings_tokens"] = ""
    df["impressions_tokens"] = ""
    df["background_tokens"] = ""
    for file in files:
        # get the patient id first from the file name
        # always be the number before first underscore
        # pid = file[: file.find("_")]
        # open and read text file
        with codecs.open(
            os.path.join(report_path, file), "r", "utf-8", errors="replace"
        ) as f:
            text = f.read()
            # concatenate all of the words together and clean text
            full_report = clean_txt(text)
            # split the text into sections:
            findings = findings_sect(full_report)
            impressions = impressions_sect(full_report)
            background = background_sect(text) # we use the unclean text because I use the newlines
            # find number of tokens each section has:
            findings_tokens = n_tokens(findings)
            impressions_tokens = n_tokens(impressions)
            imp_back_tokens = n_tokens(background+impressions)
            # determine which sections it has
            sections = report_sections(findings, impressions)
            # add this nospace text to the csv file:
            index = df[
                df["File Name"].str.contains(file)
            ].index  # assume that there is always only one index match...
            df.loc[index, "full_report"] = full_report
            df.loc[index, "findings"] = findings
            df.loc[index, "impressions"] = impressions
            df.loc[index, "background"] = background
            df.loc[index, "impressions_back"] = background + impressions
            df.loc[index, "sections"] = sections
            df.loc[index, "findings_tokens"] = findings_tokens
            df.loc[index, "impressions_tokens"] = impressions_tokens
            df.loc[index, "background_tokens"] = imp_back_tokens
    full_df = df
    # full_df.to_csv(met_info_file)
    print(
        "Shape of pre-process full_df_labeled: ", full_df.shape
    )  # should be 1059 -- this is after deleting the ones we need to verify for
    full_df = full_df[
        (full_df["sections"] == "Both")
        & (full_df["impressions_tokens"].apply(pd.to_numeric) <= 512)
    ]
    print("Shape of full_df_labeled: ", full_df.shape)
    return full_df


def process_file(file, report_path, valid_files_set):
        """Helper function to process a single file."""
        file_path = os.path.join(report_path, file)
        try:
            with codecs.open(file_path, "r", "utf-8", errors="replace") as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading {file}: {e}")
            return None
        
        if file not in valid_files_set:
            return None

        scan_type = file[file.find("_") + 1 : file.find("_") + 3]
        full_report = clean_txt(re.sub("\s{1,}", " ", text))

        findings = findings_sect(full_report)
        impressions = impressions_sect(full_report)
        findings_tokens = n_tokens(findings)
        impressions_tokens = n_tokens(impressions)
        sections = report_sections(findings, impressions)

        return {
            "File Name": file,
            "Scan type": scan_type,
            "full_report": full_report,
            "findings": findings,
            "impressions": impressions,
            "sections": sections,
            "findings_tokens": findings_tokens,
            "impressions_tokens": impressions_tokens,
        }
def organize_unlabeled_data(report_path, input_df):
    """Process all files in report_path and create a DataFrame."""
    
    df = pd.DataFrame(
        columns=[
            "File Name",
            "Scan type",
            "full_report",
            "findings",
            "impressions",
            "sections",
            "findings_tokens",
            "impressions_tokens",
        ]
    )

    files = os.listdir(report_path)
    print(f"Total files: {len(files)}")

    valid_files_set = set(input_df["File Name"].values)  # Faster lookup

    # body_parts = {
    #     "bone", "spine", "knee", "shoulder", "abdomen", "pelvis", "chest", "neck",
    #     "hip", "pituitary", "lung", "spleen", "stomach", "adrenal glands", "pancreas",
    #     "liver", "kidney", "bladder", "prostate", "ovary", "colon", "uterus", "cervix",
    #     "hand", "arm", "finger", "toe", "knee", "wrist", "leg"
    # }

    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:  # Adjust max_workers as needed
        futures = [executor.submit(process_file, file, report_path, valid_files_set) for file in files]
        
        for future in tqdm(futures):
            result = future.result()
            if result:
                results.append(result)

    df = pd.DataFrame(results)

    # df = pd.DataFrame([r for r in results if r is not None])  # Filter out None results
    # Shuffle and reset index
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)

    return df



# two assumptions are made for this to work:
# reports have a findings and impression section
# the impression section immediately follows the findings section
# impression section is the final section


if __name__ == "__main__":
    # parse some arguments that are needed
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--output-name", type=str, default="data.csv")
    argparser.add_argument("--generate-freq", action="store_true")
    argparser.add_argument(
        "--folder-path",
        type=str,
        help="Points to where the txt files are stored corresponding to the annotations",
    )
    args = argparser.parse_args()
    
    
    # input_df = pd.read_csv('/apps/data/AHS_project/Data/ir_analysis/ir2_nonmelanoma_prediag_cns_excluded_28.csv')
    # all_reports = organize_unlabeled_data(report_path="/apps/data/AHS_project/Data/Raw Reports/deidentified", input_df=input_df)
    # print('organize completed!')
    # all_reports.to_csv('Data/ir_analysis/ir2_28_cohort.csv', index=False)
    # all_reports.to_pickle('Data/ir_analysis/ir2_28_cohort.pkl')

    input_df = pd.read_csv('/apps/data/AHS_project/Data/ir_analysis/acr_ir_nonmelanoma_prediag_excluded_28.csv')
    all_reports = organize_unlabeled_data(report_path="/apps/data/AHS_project/Data/Raw Reports/deidentified", input_df=input_df)
    print('organize completed!')
    all_reports.to_csv('Data/ir_analysis/acr_28_cohort.csv', index=False)
    all_reports.to_pickle('Data/ir_analysis/acr_28_cohort.pkl')


    
    # ir2 = pd.read_csv('/apps/data/AHS_project/Resources/ir2_weight_by_patient.csv')
    # ir2 = ir2[~ir2['New_Met'].isna()]
    # ir2 = organize_met_labeled_data(df=ir2, report_path="/apps/data/AHS_project/ACR_filtered")
    # ir2 = ir2[['ID', 'File Name', 'met_label_x']]
    # all_ir =  pd.concat([ir1, ir2], ignore_index=True)
    # all_ir.to_pickle('Data/all_ir.pkl')
    # all_ir.to_csv('Data/all_ir.csv', index=False)
   
