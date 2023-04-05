#!/usr/bin/python
import re
import os
########################################################################################
#    Citation - reused with modification from
#
#    Title: patient_trajectory_prediction
#    Author: JamilProg
#    Date: 10/26/2020
#    Availability: https://github.com/JamilProg/patient_trajectory_prediction/blob/master/concept_annotation/quickumls_processing.py
#
########################################################################################

def post_process():
    tp1 = open("temporator.csv", 'w')
    with open('./data/concatenated_output.csv') as fread1:
        for line in fread1.readlines():
            # Remove the header
            if 'ROW_ID' in line:
                continue
            # print("l",line)
            cl = re.sub(r'\n', r' ', line)
            tp1.write(cl)
    tp1.close()

    tp2 = open("temporator2.csv", 'w')
    with open("temporator.csv") as fread2:
        for line in fread2.readlines():
            cl = re.sub(r'(C\d+ *\")', r'\1\n', line)
            cl = re.sub(r'\nC', 'C', cl)
            cl = re.sub(r'\" \"', r'\" \"\n', cl)
            cl = re.sub(r'\n +', r'\n', cl)
            cl = re.sub(r' +', r' ', cl)
            cl = re.sub(r'\.0', r'', cl)
            cl = re.sub(r'"', r'', cl)
            tp2.write(cl)
    tp2.close()
    os.remove("temporator.csv")

    pattern = re.compile(r'^.*?,.*?,\d+,')
    finalfile = open("post_processed_output.csv", 'w')
    with open("temporator2.csv") as fread:
        for line in fread.readlines():
            # Empty CUI codes, we ignore it
            if line.count("\" \"") > 0 :
                continue
            # Empty line, we ignore it
            if line == " \n" or line == "\n" or line == " ":
                continue
            # Make sure that we have an HADM_ID before writing it
            if not re.match(pattern, line):
                continue
            finalfile.write(line)
    finalfile.close()
    os.remove("temporator2.csv")

if __name__ == '__main__':
    post_process()
