# CSE6250_GroupProject
## How to run this project
1. Create and activate the conda environment from the environment.yml file
2. Generate Samples
   1. Inside of the generate_samples folder, create a data directory
   2. Inside of the data directory, create an input and an output folder
   3. Place the ADMISSIONS.csv, DIAGNOSES_ICD.csv, and NOTEEVENTS.csv MIMIC III datasets in the input directory
   4. Run python samples.py to generate samples, from each file, which will be placed in the samples folder
   5. The directory structure inside of your generate_samples folder should look like this
````
generate_samples
├── data
│   ├── input
│   │   ├── ADMISSIONS.csv
│   │   ├── DIAGNOSES_ICD.csv
│   │   └── NOTEEVENTS.csv
│   └── output
│       ├── ADMISSIONS_SAMPLE.csv
│       ├── DIAGNOSES_ICD_SAMPLE.csv
│       └── NOTEEVENTS_SAMPLE.csv
````
3. Clean the data
   1. Inside of the concept_annotation folder, create a data folder
   2. The next set of scripts that need to be run are in the data_cleaning folder
   3. Run python data_cleaning.py, which can take up to an hour to execute. When it is complete, it will put an output.csv file in the data folder
   4. Run python create_chunks.py, which splits the output.csv file into roughly equal sized chunks, which will be placed in a chunkssmall directory. This script also creates an outputchunkssmall directory, which will be used by the next step.
   5. When this process is complete, the directory structure of your concept_annotation folder should look like this:
````
concept_annotation
├── data
│   ├── chunkssmall  [103 entries exceeds filelimit, not opening dir]
│   ├── output.csv
└── outputchunkssmall 
````
4. Run quickUMLS
   1. Please follow the [quickUMLS installation instructions](https://github.com/Georgetown-IR-Lab/QuickUMLS)
   2. After quickUMLS has been installed, you should have a QuickUMLS folder, with the following structure:
````
QuickUMLS
├── cui-semtypes.db
│   ├── cui.leveldb  [124 entries exceeds filelimit, not opening dir]
│   └── semtypes.leveldb  [80 entries exceeds filelimit, not opening dir]
├── database_backend.flag
├── language.flag
└── umls-simstring.db  [1429 entries exceeds filelimit, not opening dir]
````
3. Place the QuickUMLS directory inside of the concept_annotation directory
4. Run python quickUMLS_getCUI.py, which can take a very long time to execute- up to 4 days
5. After this process completes, you will have several csv.output files inside of the data/outputchunsksmall directory
6. From the terminal, navigate to the concept_annotation/data folder and run the following command to concatenate all of the output files together
````
cat .outputchunkssmall/* > concatenated_output.csv
````
7. Next run python quickumls_processing.py, which will perform final data prep on the input data, and will generate a file called post_processed_output.csv
8. At the end of this process, your concept_annotation folder should look like the following:
````
concept_annotation
├── QuickUMLS
│   ├── cui-semtypes.db
│   │   ├── cui.leveldb  [124 entries exceeds filelimit, not opening dir]
│   │   └── semtypes.leveldb  [80 entries exceeds filelimit, not opening dir]
│   ├── database_backend.flag
│   ├── language.flag
│   └── umls-simstring.db  [1429 entries exceeds filelimit, not opening dir]
├── data
│   ├── chunkssmall  [103 entries exceeds filelimit, not opening dir]
│   ├── concatenated_output.csv
│   ├── output.csv
│   └── outputchunkssmall  [80 entries exceeds filelimit, not opening dir]
├── post_processed_output.csv
├── quickUMLS_getCUI.py
├── quickumls_processing.py
└── useful_commands.txt
````
5. Create datasets, train, and test feed-forward neural network
   1. Create a data folder inside of the PyTorch_scripts/mortality_prediction directory
   2. Run python 01_data_prep_mortality.py, which will prepare the datasets for NN training/testing. This script will output two files, prepared_data.npz and prepared_data_deathTime.npz to the data folder
   3. Run python 02_FFN_mortality.py, which will train and test the FNN, and will output AUC-ROC along with loss for the train and test datasets to the terminal