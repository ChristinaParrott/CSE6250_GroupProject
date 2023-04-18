########################################################################################
#    Citation - reused with modification from:
#
#    Title: patient_trajectory_prediction
#    Author: JamilProg
#    Date: 10/23/2020
#    Availability: https://github.com/JamilProg/patient_trajectory_prediction/blob/master/PyTorch_scripts/mortality_prediction/01_data_prep_mortality.py
#
########################################################################################
### THIS IS STILL A WORK IN PROGRESS AND CAN'T REPLACE THE ORIGINAL SCRIPT YET
import pickle
import csv
import copy
import sys
import os
from os.path import dirname, abspath

import pandas as pd
import numpy as np

VERBOSE = True

parent_path = dirname(dirname(dirname(abspath(__file__))))
samples_path = os.path.join(parent_path, 'generate_samples/data/output')
sys.path.insert(0, samples_path)

ADMISSIONS_FILE = os.path.join(samples_path, "ADMISSIONS_SAMPLE.csv")
DIAGNOSES_FILE = os.path.join(samples_path, "DIAGNOSES_ICD_SAMPLE.csv")

notes_path = os.path.join(parent_path, 'concept_annotation')
sys.path.insert(0, notes_path)

NOTES_FILE = os.path.join(notes_path, "post_processed_output.csv")

if not os.path.exists('data'):
    os.mkdir('data')

OUTPUT_FILE = 'data/prepared_data'

CUI_set = set()
CCS_set = set()

def get_ccs(icd_code):
    reader = csv.reader(open('ICDToCCS.csv', 'r'))
    d = dict(reader)
    if (icd_code not in d.keys()):
        return
    else:
        return d[icd_code]

def get_ICD9s_from_mimic_file(f, hadm_map):
    mimic = open(f, 'r')
    mimic.readline()
    null_icd9 = 0
    for line in mimic:
        codes = line.strip().split(',')
        id = int(codes[2])
        if (len(codes[4]) == 0):
            null_icd9 += 1
            continue

        icd9 = codes[4]
        if icd9.find("\"") != -1:
            icd9 = icd9[1:-1]
        icd9 = icd9

        if id in hadm_map:
            hadm_map[id].add(icd9)
        else:
            hadm_map[id] = set()
            hadm_map[id].add(icd9)
    for id in hadm_map.keys():
        hadm_map[id] = list(hadm_map[id])
    mimic.close()

    return hadm_map

def check_int(value):
    try:
        int(value)
        return value
    except ValueError:
        return np.NaN

def check_length(value):
    if len(value) < 5:
        return np.NaN
    return value

def split_and_convert_to_int(value):
    ids = value.strip().split(' ')
    output_list = []
    for id in ids:
        try:
            output_list.append(int(id))
            CUI_set.add(int(id))
        except:
            continue
    return output_list

def get_cui_notes():
    col_names = ["ROW_ID","SUBJECT_ID","HADM_ID","CHARTDATE","CHARTTIME","STORETIME","CATEGORY","DESCRIPTION","CGID","ISERROR","CUI_LIST"]
    notes_df = pd.read_csv(NOTES_FILE, names=col_names)

    if VERBOSE:
        print(f"notes length: {len(notes_df.index)}")

    notes_df = notes_df[notes_df["ISERROR"] != 1]

    if VERBOSE:
        print(f"notes length with error flags removed: {len(notes_df.index)}")

    # notes_df['CATEGORY'] = notes_df.apply(lambda row: row['CATEGORY'].lower().rstrip().strip('"'), axis=1)
    # notes_by_category = notes_df.groupby(['CATEGORY'])['CATEGORY'].count().reset_index(name='COUNT')
    # category_dict = dict(zip(notes_by_category.CATEGORY, notes_by_category.COUNT))
    #
    # notes_df['DESCRIPTION'] = notes_df.apply(lambda row: row['DESCRIPTION'].lower().rstrip().strip('"'), axis=1)
    # notes_by_desc = notes_df.groupby(['DESCRIPTION'])['DESCRIPTION'].count().reset_index(name='COUNT')
    # description_dict = dict(zip(notes_by_desc.DESCRIPTION, notes_by_desc.COUNT))

    notes_df['HADM_ID'] = notes_df['HADM_ID'].apply(check_int)
    notes_df = notes_df.dropna(subset=['HADM_ID'])

    if VERBOSE:
        print(f"notes length with invalid HADM_IDs removed: {len(notes_df.index)}")

    notes_df['CUI_LIST'] = notes_df.apply(lambda row: row['CUI_LIST'].strip().strip('"'), axis=1)
    notes_df['CUI_LIST'] = notes_df.apply(lambda row: row['CUI_LIST'].replace('C', ''), axis=1)

    notes_df['CUI_LIST'] = notes_df['CUI_LIST'].apply(check_length)
    notes_df = notes_df.dropna(subset=['CUI_LIST'])

    notes_df['CUI_LIST'] = notes_df['CUI_LIST'].apply(split_and_convert_to_int)

    if VERBOSE:
        print(f"notes length with invalid CUI lists removed: {len(notes_df.index)}")

    hadm_to_cui = dict(zip(notes_df.HADM_ID, notes_df.CUI_LIST))
    return hadm_to_cui


def admissions_to_dicts():
    date_parser = lambda d: pd.to_datetime(d, format='%Y-%m-%d %H:%M:%S', errors='coerce')
    admissions_df = pd.read_csv(ADMISSIONS_FILE, parse_dates=['ADMITTIME', 'DISCHTIME', 'DEATHTIME'], date_parser=date_parser)

    if VERBOSE:
        print(f"initial admissions: {len(admissions_df.index)}")

    admissions_df["SUBJECT_ID"] = admissions_df["SUBJECT_ID"].astype(int)
    admissions_df["HADM_ID"] = admissions_df["HADM_ID"].astype(int)
    admissions_df = admissions_df[admissions_df['ADMITTIME'] <= admissions_df['DISCHTIME']]

    if VERBOSE:
        print(f"admissions with valid admit/discharge dates: {len(admissions_df.index)}")

    arr_admit_date = admissions_df["ADMITTIME"].dt.to_pydatetime()
    admissions_df["ADMITTIME"] = pd.Series(arr_admit_date, dtype="object")

    arr_disch_date = admissions_df["DISCHTIME"].dt.to_pydatetime()
    admissions_df["DISCHTIME"] = pd.Series(arr_disch_date, dtype="object")

    arr_death_date = admissions_df["DEATHTIME"].dt.to_pydatetime()
    admissions_df["DEATHTIME"] = pd.Series(arr_death_date, dtype="object")

    admittime_dict = dict(zip(admissions_df.HADM_ID, admissions_df.ADMITTIME))
    discharge_dict = dict(zip(admissions_df.HADM_ID, admissions_df.DISCHTIME))

    deaths_df = admissions_df[admissions_df['DEATHTIME'].notnull()]

    deathtime_dict = dict(zip(deaths_df.SUBJECT_ID, deaths_df.DEATHTIME))

    subject_df = admissions_df.groupby('SUBJECT_ID')['HADM_ID'].apply(list).reset_index(name='HADM_LIST')
    subject_dict = dict(zip(subject_df.SUBJECT_ID, subject_df.HADM_LIST))

    return subject_dict, admittime_dict, discharge_dict, deathtime_dict

if __name__ == '__main__':
    subject_dict, admittime_dict, discharge_dict, deathtime_dict = admissions_to_dicts()

    hadm_cui = get_cui_notes()

    admits_no_note = 0

    for subject_id, subject_hadm_list in subject_dict.items():
        subject_hadm_list_copy = list(subject_hadm_list)
        for hadm_id in subject_hadm_list_copy:
            if hadm_id not in hadm_cui.keys():
                admits_no_note += 1
                del admittime_dict[hadm_id]

                subject_hadm_list.remove(hadm_id)
    if VERBOSE:
        print(f'Number of admissions without notes: {admits_no_note}')

    hadm_icd9_dict = {}

    if len(DIAGNOSES_FILE) > 0:
        hadm_icd9_dict = get_ICD9s_from_mimic_file(DIAGNOSES_FILE, hadm_icd9_dict)

    admissions_no_diagnosis = 0
    subject_invalid_admission = 0
    subject_no_admission = []
    for subject_id, hadm_list in subject_dict.items():
        hadm_list_copy = list(hadm_list)
        for admission in hadm_list_copy:
            if admission not in hadm_icd9_dict.keys():
                admissions_no_diagnosis += 1
                del admittime_dict[admission]
                hadm_list.remove(admission)
            if len(hadm_list) == 0:
                subject_invalid_admission += 1
                subject_no_admission.append(subject_id)
    for subject in subject_no_admission:
        del subject_dict[subject]

    if VERBOSE:
        print(f'Number of admissions with no diagnosis {admissions_no_diagnosis}')
        print(f'Number of subjects with invalid admission {subject_invalid_admission}')

    subject_hadm_dict = {}

    subjects_lessthan2admissions = 0
    subjects_in_map = list(subject_dict.keys())
    num_adm = 0
    for subject_id in subjects_in_map:
        subject_hadm_list = subject_dict[subject_id]
        if len(subject_hadm_list) < 2:
            subjects_lessthan2admissions += 1
            del subject_dict[subject_id]
            continue
        sorted_list = sorted([(id, admittime_dict[id], hadm_cui[id]) for id in subject_hadm_list])
        subject_dict[subject_id] = sorted_list
        num_adm += len(sorted_list)

    if VERBOSE:
        print(f'Number of subjects with less than 2 admissions {subjects_lessthan2admissions}')

    subject_hadm_dict = subject_dict

    CUI_ordered = {}
    for i, key in enumerate(CUI_set):
        CUI_ordered[key] = i

    for subject, admissions in subject_hadm_dict.items():
        for admission in admissions:
            codes_list = admission[2]
            for i in range(len(codes_list)):
                codes_list[i] = CUI_ordered[
                    codes_list[i]]

    icdtoccs = dict()
    for k, icd_values in hadm_icd9_dict.items():
        for icd9code in icd_values:
            if icd9code not in icdtoccs.keys():
                ccs_code = get_ccs(icd9code)
                icdtoccs[icd9code] = ccs_code
                CCS_set.add(ccs_code)

    subject_to_ordered_hadm_copy = copy.deepcopy(subject_hadm_dict)
    for subj, admlist in subject_to_ordered_hadm_copy.items():
        for adm in admlist:
            adm_list = list(adm)
            subject_hadm_dict[subj].remove(adm)
            ccs_list = list()
            for icdcode in hadm_icd9_dict[adm[0]]:
                ccs_list.append(icdtoccs[icdcode])
            adm_list.append(ccs_list)
            adm = tuple(adm_list)
            subject_hadm_dict[subj].append(adm)

    CUI_ordered = {}
    for i, key in enumerate(CCS_set):
        CUI_ordered[key] = i

    num_admissions = 0
    for subject, admissions in subject_hadm_dict.items():
        for admission in admissions:
            num_admissions += 1
            codes_list = admission[3]
            for i in range(len(codes_list)):
                codes_list[i] = CUI_ordered[codes_list[i]]

    subject_to_ordered_hadm_copy = copy.deepcopy(subject_hadm_dict)
    for subj, adm_list in subject_to_ordered_hadm_copy.items():
        for adm in adm_list:
            adm_list = list(adm)
            subject_hadm_dict[subj].remove(adm)
            adm_list.append(discharge_dict[adm[0]])
            adm = tuple(adm_list)
            subject_hadm_dict[subj].append(adm)

    subjects_without_deathtime = []
    for subj, _ in subject_hadm_dict.items():
        if subj not in deathtime_dict:
            subjects_without_deathtime.append(subj)
    for subj_to_del in subjects_without_deathtime:
        del subject_hadm_dict[subj_to_del]

    if VERBOSE:
        print(f'Number of subjects without a death time {len(subjects_without_deathtime)}')
        print(f'Final number of subjects in hadm dict {len(subject_hadm_dict)}')
        print(f'Final number of subjects in deathtime dict {len(deathtime_dict)}')

    pickle.dump(subject_hadm_dict, open(OUTPUT_FILE + '.npz', 'wb'), protocol=2)
    pickle.dump(deathtime_dict, open(OUTPUT_FILE + '_deathTime.npz', 'wb'), protocol=2)