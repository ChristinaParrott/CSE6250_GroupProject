########################################################################################
#    Citation - reused with modification from:
#
#    Title: patient_trajectory_prediction
#    Author: JamilProg
#    Date: 5/9/2022
#    Availability: https://github.com/JamilProg/patient_trajectory_prediction/blob/master/concept_annotation/quickUMLS_getCUI.py
#
########################################################################################
from quickumls import QuickUMLS
import os, re, csv
from multiprocessing import Pool
import pandas as pd

# TUI list
TUI_BETA = ["T195","T123","T122","T103","T120","T200","T126","T116","T196","T131","T125","T129","T130","T121","T192","T127","T104","T114","T197","T109","T038","T034","T070","T067","T068","T069","T043","T201","T045","T041","T032","T040","T042","T039","T044","T020","T190","T049","T019","T047","T050","T033","T037","T048","T191","T046","T184","T091","T090","T017","T029","T023","T030","T031","T022","T025","T026","T018","T021","T024","T079","T203","T074","T075","T100","T011","T008","T194","T007","T012","T204","T099","T013","T004","T096","T016","T015","T001","T101","T098","T097","T014","T010","T005","T058","T060","T061"]
TUI_ALPHA = ["T020","T190","T049","T019","T047","T050","T033","T037","T048","T191","T046","T184","T038","T069","T068","T034","T070","T067","T043","T201","T045","T041","T044","T032","T040","T042","T039","T116","T195","T123","T122","T103","T120","T104","T200","T196","T126","T131","T125","T129","T130","T197","T114","T109","T121","T192","T127"]

# You may need to run the following command if this function complains about a language not being installed
# python -m spacy download en_core_web_sm
matcher = QuickUMLS(quickumls_fp='./QuickUMLS', overlapping_criteria='score', threshold=0.7, similarity_name='cosine', window=5)

# Change TUIs here to alter the list you want to use
TUIs = TUI_BETA

# Input and output path
input_chunks = "./data/chunkssmall/"
output_chunks = "./data/outputchunkssmall/"

def get_concepts(text):
    list_cui = []
    list_terms = []
    output_text = ""

    paragraphs = text.splitlines(True)
    for paragraph in paragraphs:
        matches = matcher.match(paragraph, best_match=True, ignore_syntax=False)
        concepts_output = []
        for phrase_candidate in matches:
            # Find max
            max = 0
            for candidate in phrase_candidate:
                if candidate['similarity'] > max and set(candidate['semtypes']).intersection(TUIs):
                    max = candidate['similarity']

            # Get preferred terms for that max
            list_to_write = []
            if max >= 0:
                for candidate in phrase_candidate:
                    if candidate['similarity'] == max:
                        if candidate['term'] not in list_terms:
                            if candidate['cui'] not in list_cui:
                                list_cui.append(candidate['cui'])
                                list_terms.append(candidate['term'])
                                list_to_write.append(candidate['cui'])

            concepts_output.append(list_to_write)

            for concepts in concepts_output:
                for terms in concepts:
                    terms = re.sub(r' ', '', terms)
                    output_text += terms + " "

    return output_text


def get_cui(file):
    filename = input_chunks + file
    print(f"File {filename} processing!")

    df_notes = pd.read_csv(filename)
    df_notes = df_notes.iloc[:2]
    df_notes['TEXT'] = df_notes.apply(lambda row: get_concepts(row['TEXT']), axis=1)
    df_notes['TEXT'] = '" ' + df_notes['TEXT'] + ' "'
    df_notes['HADM_ID'] = df_notes['HADM_ID'].astype(int)

    output_file = output_chunks + file + '.output'
    df_notes.to_csv(output_file, index=False, header=False, quoting=csv.QUOTE_NONE)

    print(f"File {filename} done processing!")

if __name__ == "__main__":
    pool = Pool(os.cpu_count()-2)
    pool.map(get_cui, os.listdir(input_chunks))

