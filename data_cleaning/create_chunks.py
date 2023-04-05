##import statements
import os
import pandas as pd
import sys
from os.path import dirname, abspath

# avoid having to copy files around manually
parent_path = dirname(dirname(abspath(__file__)))
concept_path = os.path.join(parent_path, 'concept_annotation\\data')
sys.path.insert(0, concept_path)

note_events_path = os.path.join(concept_path, 'output.csv')
df_note_events = pd.read_csv(note_events_path)
file_size = os.stat(note_events_path).st_size / (1024 * 1024)
## chunk size should be 50, but I'm changing it to 5 for the smaller file
num_chunks = file_size // 5
rows_per_chunk = len(df_note_events.index) // num_chunks

chunks_path = os.path.join(concept_path, 'chunkssmall')
output_chunks_path = os.path.join(concept_path, 'outputchunkssmall')

if not os.path.exists(chunks_path):
    os.makedirs(chunks_path)
if not os.path.exists(output_chunks_path):
    os.makedirs(output_chunks_path)

for i, chunk in enumerate(pd.read_csv(note_events_path, chunksize=rows_per_chunk)):
    output_file = os.path.join(chunks_path, f'{i}.csv')
    chunk.to_csv(output_file, index=False)