import pandas as pd
import numpy as np

# Generate samples from each of the MIMIC input files, while ensuring that our samples will contain admissions
# that have corresponding clinical notes

admissions = pd.read_csv("data/input/ADMISSIONS.csv")
diagnoses = pd.read_csv("data/input/DIAGNOSES_ICD.csv")
notes = pd.read_csv("data/input/NOTEEVENTS.csv")

admissions_hadms = admissions["HADM_ID"].drop_duplicates()
notes_hadms = notes["HADM_ID"].drop_duplicates()

admissions_with_notes = pd.Series(np.intersect1d(admissions_hadms.values, notes_hadms.values)).astype(int)
admissions_with_notes = admissions_with_notes.sample(10000, random_state=6250).tolist()

admissions = admissions[admissions["HADM_ID"].isin(admissions_with_notes)]
diagnoses = diagnoses[diagnoses["HADM_ID"].isin(admissions_with_notes)]
notes = notes[notes["HADM_ID"].isin(admissions_with_notes)]

admissions.to_csv("data/output/ADMISSIONS_SAMPLE.csv", index=False)
diagnoses.to_csv("data/output/DIAGNOSES_ICD_SAMPLE.csv", index=False)
notes.to_csv("data/output/NOTEEVENTS_SAMPLE.csv", index=False)


