########################################################################################
#    Citation - reused with modification from:
#
#    Title: patient_trajectory_prediction
#    Author: JamilProg
#    Date: 10/23/2020
#    Availability: https://github.com/JamilProg/patient_trajectory_prediction/blob/master/PyTorch_scripts/mortality_prediction/01_data_prep_mortality.py
#
########################################################################################

import pickle
from datetime import datetime
import csv
import copy
import sys
import os
from os.path import dirname, abspath

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


def get_CUINotes_from_CSV_file(f):

    mimic = open(f, 'r')
    mimic.readline()
    category_dict = {}
    description_dict = {}
    hadmToCUI = {}
    inconsistent_ids = 0
    errors = 0
    invalid_notes = 0
    for line in mimic:
        codes = line.strip().split(',')

        is_error = codes[9]

        if is_error == '1':
            errors += 1
            continue


        category = codes[6].lower().rstrip().strip('"')
        if category in category_dict:
            category_dict[category] += 1
        else:
            category_dict[category] = 1

        description = codes[7].lower().rstrip().strip('"')
        if description in description_dict:
            description_dict[description] += 1
        else:
            description_dict[description] = 1


        id = codes[2]
        if id == '':
            inconsistent_ids += 1
            continue
        else:
            try:
                id = int(id)
            except:
                continue

        CUI_vector = codes[10].strip().strip('"')

        CUI_vector = CUI_vector.replace('C', '')


        if len(CUI_vector) < 5:
            invalid_notes += 1
        else:
            if id in hadmToCUI:
                hadmToCUI[id].append(CUI_vector)
            else:
                hadmToCUI[id] = []
                hadmToCUI[id].append(CUI_vector)
    mimic.close()


    return hadmToCUI


def split_and_convertToInt(CUI_list):

    set_of_CUIcodes = set()
    for note in CUI_list:
        codes = note.strip().split(' ')
        for c in codes:
            set_of_CUIcodes.add(int(c))
            CUI_set.add(int(c))
    return list(set_of_CUIcodes)


def getCUICodes_givenAdmID():

    hadmToCUI = get_CUINotes_from_CSV_file(NOTES_FILE)

    for id, notes_list in hadmToCUI.items():
        hadmToCUI[id] = split_and_convertToInt(hadmToCUI[id])
    return hadmToCUI


def admissionsParser():

    ADMISSIONS = open(ADMISSIONS_FILE, 'r')
    ADMISSIONS.readline()
    initial_admissions = 0
    subject_dict = {}
    admittime_dict = {}
    discharge_dict = {}
    deathtime_dict = {}

    for line in ADMISSIONS:
        initial_admissions += 1
        codes = line.strip().split(',')
        subject_id = int(codes[1])
        id = int(codes[2])
        admittime = datetime.strptime(codes[3], '%Y-%m-%d %H:%M:%S')
        dischargetime = datetime.strptime(codes[4], '%Y-%m-%d %H:%M:%S')
        deathtime = None
        if codes[5]:
            deathtime = datetime.strptime(codes[5], '%Y-%m-%d %H:%M:%S')

        if admittime > dischargetime:
            continue

        admittime_dict[id] = admittime
        discharge_dict[id] = dischargetime
        if deathtime:
            deathtime_dict[subject_id] = deathtime
        if subject_id in subject_dict:
            subject_dict[subject_id].append(id)
        else:
            subject_dict[subject_id] = [id]
    ADMISSIONS.close()

    return subject_dict, admittime_dict, discharge_dict, deathtime_dict


if __name__ == '__main__':

    subject_dict, admittime_dict, discharge_dict, deathtime_dict = admissionsParser()


    hadmtoCUI = getCUICodes_givenAdmID()


    admits_no_note = 0

    for subject_id, subjectHadmList in subject_dict.items():
        subjectHadmListCopy = list(subjectHadmList)
        for hadm_id in subjectHadmListCopy:
            if hadm_id not in hadmtoCUI.keys():
                admits_no_note += 1
                del admittime_dict[hadm_id]

                subjectHadmList.remove(hadm_id)
    # print('-Number of admissions without notes: ' + str(admits_no_note))
    # print('-Number of admissions after cleaning: ' + str(len(hadmtoCUI)))
    # print('-Number of subjects after cleaning: ' + str(len(subject_dict)))


    hadm_icd9_dict = {}

    if len(DIAGNOSES_FILE) > 0:
        hadm_icd9_dict = get_ICD9s_from_mimic_file(DIAGNOSES_FILE, hadm_icd9_dict)


    admissions_no_diagnosis = 0
    subject_invalid_admission = 0
    subject_no_admission = []
    for subject_id, hadmList in subject_dict.items():
        hadmListCopy = list(hadmList)
        for admission in hadmListCopy:
            if admission not in hadm_icd9_dict.keys():
                admissions_no_diagnosis += 1
                del admittime_dict[admission]
                hadmList.remove(admission)
            if len(hadmList) == 0:
                subject_invalid_admission += 1
                subject_no_admission.append(subject_id)
    for subject in subject_no_admission:
        del subject_dict[subject]

    subject_hadm_dict = {}

    subjects_lessthan2admissions = 0
    subjects_in_map = list(subject_dict.keys())
    numAdmissions = 0
    for subject_id in subjects_in_map:
        subjectHadmList = subject_dict[subject_id]
        if len(subjectHadmList) < 1:
            subjects_lessthan2admissions += 1
            del subject_dict[subject_id]
            continue
        sortedList = sorted([(id, admittime_dict[id], hadmtoCUI[id]) for id in subjectHadmList])
        subject_dict[subject_id] = sortedList
        numAdmissions += len(sortedList)

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

    subjectTOorderedHADM_IDS_COPY = copy.deepcopy(
        subject_hadm_dict)
    for subj, admlist in subjectTOorderedHADM_IDS_COPY.items():
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

    numAdmissions = 0
    for subject, admissions in subject_hadm_dict.items():
        for admission in admissions:
            numAdmissions += 1
            codes_list = admission[3]
            for i in range(len(codes_list)):
                codes_list[i] = CUI_ordered[
                    codes_list[i]]


    subjectTOorderedHADM_IDS_COPY = copy.deepcopy(subject_hadm_dict)
    for subj, admlist in subjectTOorderedHADM_IDS_COPY.items():
        for adm in admlist:
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


    pickle.dump(subject_hadm_dict, open(OUTPUT_FILE + '.npz', 'wb'), protocol=2)
    pickle.dump(deathtime_dict, open(OUTPUT_FILE + '_deathTime.npz', 'wb'), protocol=2)