##import statements
import os
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
import sys
from os.path import dirname, abspath

# avoid having to copy files around manually
parent_path = dirname(dirname(abspath(__file__)))
samples_path = os.path.join(parent_path, 'generate_samples\\data\\output')
sys.path.insert(0, samples_path)

##Step 0 - Load files
note_events_path = os.path.join(samples_path, "NOTEEVENTS_SAMPLE.csv")
df_note_events = pd.read_csv(note_events_path)

##Step 1 - Data Cleaning
def remove_anonymization(text):
  cleaned_text = text.strip()
  cleaned_text = re.sub(r'\[\*\*.*?\*\*\]', '', cleaned_text, flags=re.DOTALL | re.MULTILINE)
  return cleaned_text

def remove_doctor_quotes(text):
  cleaned_text = text.strip()
  cleaned_text = re.sub('""', '', cleaned_text, flags=re.DOTALL | re.MULTILINE)
  return cleaned_text

def remove_extra_commas(text):
  cleaned_text = text.strip()
  cleaned_text = cleaned_text.replace(',', '')
  return cleaned_text

def lower_all_text(text):
  cleaned_row = text.lower()
  return cleaned_row

def clean_useless_words(text):
  useless_words = [r'admission date:.*',
                   r'sex *?: *[mf]?[ \n]',
                   r'date of birth *?:',
                   r'service *?:.*',
                   r'addendum *?:?',
                   r'medquist36',
                   r'm\.d\.',
                   r'\Wmd\W|^md\W',
                   r'dictated *?by *?: *',
                   r'completed *?by *?: *',
                   r'cc *?(by)? *?: *',
                   r'd *?: *(\d\d:\d\d)?',
                   r't *?: *(\d\d:\d\d)?',
                   r'phone *?: *',
                   r'provider *?: *',
                   r'date/time *?: *',
                   r'job# *?: *',
                   r'monday',
                   r'tuesday',
                   r'wednesday',
                   r'thursday',
                   r'friday',
                   r'saturday',
                   r'sunday']     
  cleaned_line = text          
  for word in useless_words:
    cleaned_line = cleaned_line.strip()
    cleaned_line = re.sub(word, '', cleaned_line)
  
  return cleaned_line

def remove_time(text):
  cleaned_text = text.strip()
  cleaned_text = re.sub(r'\d?\d:\d\d *?((am|pm)\W)?', '', cleaned_text, flags=re.DOTALL | re.MULTILINE)
  cleaned_text = re.sub(r'\d?\d:\d\d:\d\d *?((am|pm)\W)?', '', cleaned_text, flags=re.DOTALL | re.MULTILINE)
  return cleaned_text

def remove_parentheses_num(text):
  cleaned_text = text.strip()
  cleaned_text = re.sub(r'\(\d\) ?', '', cleaned_text, flags=re.DOTALL | re.MULTILINE)
  return cleaned_text

def remove_spaces(text):
  cleaned_text = text.strip()
  # If paragraph starts with spaces, remove all of them
  cleaned_text = re.sub(r'^ +', '', cleaned_text, flags=re.DOTALL | re.MULTILINE)
  # If more than two spaces within paragraph : leave it with two spaces
  cleaned_text = re.sub(r' +', ' ', cleaned_text, flags=re.DOTALL | re.MULTILINE)
  return cleaned_text

def format_paragraphs(text):
  # REGEX : if the string contains a number followed by a dot, it is the start of a new paragraph
  reg_exp_new_line = re.compile(r'([1-9][0-9]?\. +|^#)', flags=re.DOTALL | re.MULTILINE)
  # REGEX : if a line match this, consider this line as an empty line
  bad_line_re = re.compile(r'^[,.]* *\n$', flags=re.DOTALL | re.MULTILINE)

  cleaned_paragraphs = ""
  previous_paragraph = ""
  
  text = text.strip()
  paragraphs = text.splitlines(True)

  for paragraph in paragraphs:
    if paragraph == "\n" and previous_paragraph.endswith("\n"):
      cleaned_paragraphs += "\n"
      continue

    if bad_line_re.search(paragraph.strip()):
      cleaned_paragraphs += "\n"
      continue
    
    if reg_exp_new_line.search(paragraph.strip()):
      paragraph = re.sub(reg_exp_new_line, r"\n\1", paragraph)

    if not cleaned_paragraphs.endswith("\n"):
      cleaned_paragraphs += ' '
 
    cleaned_paragraphs += paragraph.rstrip()
    previous_paragraph = paragraph
  
  return cleaned_paragraphs.strip()

def process_enums(text):
  cleaned_text = text.strip()
  regex_start_enum = re.compile(r'^[1-9][0-9]?\. +|^#', flags=re.DOTALL | re.MULTILINE)
  start_of_enum = False
  output_text = ""
  len_list = []
  paragraphs = cleaned_text.splitlines(True)

  for paragraph in paragraphs:
    last_paragraph = paragraphs[-1]
    if (paragraph == "\n" or paragraph == last_paragraph) and start_of_enum:
      if max(len_list) < 300 and sum(len_list) / len(len_list) < 250:
        paragraph = re.sub(r'\n', ' ', paragraph)
      elif sum(len_list) / len(len_list) < 150:
        paragraph = re.sub(r'\n', ' ', paragraph)
      start_of_enum = False

    if regex_start_enum.match(paragraph) and paragraph.count('"')==0:
      paragraph = re.sub(r'^[1-9][0-9]?\. +|^#+ *', '', paragraph)
      len_list.append(len(paragraph))
      start_of_enum = True

    else:
        if start_of_enum:
          if max(len_list) < 300 and sum(len_list)/len(len_list) < 250:
            paragraph = re.sub(r'\n', ' ', paragraph)
          elif sum(len_list) / len(len_list) < 150:
            paragraph = re.sub(r'\n', ' ', paragraph)
          start_of_enum = False
          len_list = []

    output_text += paragraph
  return output_text

ones = {"1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine"}
afterones = {"10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen", "14": "fourteen", "15": "fifteen",
            "16": "sixteen", "17": "seventeen", "18": "eighteen", "19": "nineteen"}
tens = {"2": "twenty", "3": "thirty", "4": "fourty", "5": "fifty",
        "6": "sixty", "7": "seventy", "8": "eighty", "9": "ninety"}
grand = {0: " billion ", 1: " million ", 2: " thousand ", 3: ""}

# This is copied directly form the original author and needs to be cited
def three_dig_to_words(val):
    """ Function converting number to words of 3 digit
    Code from Barath Kumar
    Link : https://stackoverflow.com/questions/15598083/python-convert-numbers-to-words
    """
    if val != "000":
        ans = ""
        if val[0] in ones:
            ans = ans + ones[val[0]] + " hundred "
        if val[1:] in afterones:
            ans = ans + afterones[val[1:]] + " "
        elif val[1] in tens:
            ans = ans + tens[val[1]] + " "
        if val[2] in ones and val[1:] not in afterones:
            ans = ans + ones[val[2]]
        return ans

# This is copied directly form the original author and needs to be cited
def num_to_words(value):
    """ This function takes an integer as an input, and outputs its text version
    Works with integer from 0 to 999 999 999 999.
    """
    # Padding with zeros
    pad = 12 - len(str(value))
    padded = "0" * pad + str(value)

    # Exception case
    if padded == "000000000000":
        return "zero"

    # Preparing the values before computation
    result = ""
    number_groups = [padded[0:3], padded[3:6], padded[6:9], padded[9:12]]

    for key, val in enumerate(number_groups):
        if val != "000":
            result = result + three_dig_to_words(val) + grand[key]

    result = re.sub(r'(^ *| *$)', ' ', result)
    return result

# This is copied directly form the original author and needs to be cited
def num_to_text(text):
  text = text.strip()
  cleaned_text = ""

  paragraphs = text.splitlines(True)
  for paragraph in paragraphs:
    if paragraph == "\n":
      continue

    # Avoid transforming IDs to letters
    count_comma = paragraph.count(',')
    count_quote = paragraph.count('"')
    if count_comma >= 10 and count_quote >= 1:
      cleaned_text += paragraph
      continue

    # if \d+\.\d+ is found, transform . to [space]point[space]
    # (and if there are zeros after the dot, replace them by "zero ")
    cleaned_line = re.sub(r'((\d|)*)\.00(\d+)', r' \1 point zero zero \3 ', paragraph, flags=re.DOTALL | re.MULTILINE)
    cleaned_line = re.sub(r'((\d|)*)\.0(\d+)', r' \1 point zero \3 ', cleaned_line, flags=re.DOTALL | re.MULTILINE)
    cleaned_line = re.sub(r'((\d|)*)\.(\d+)', r' \1 point \3 ', cleaned_line, flags=re.DOTALL | re.MULTILINE)
    # for all digits found, replace it by the text form (sub)
    cleaned_line = re.sub(r'([1-9]\d*|0)', lambda x: num_to_words(x.group()), cleaned_line, flags=re.DOTALL | re.MULTILINE)
    
    cleaned_text += cleaned_line
  return cleaned_text

def remove_special_chars(text):
  cleaned_text = text.strip()
  count_comma = cleaned_text.count(',')
  count_quote = cleaned_text.count('"')
  if count_comma >= 10 and count_quote >= 1:
      return cleaned_text

  cleaned_text = re.sub(r'[*<>!?#.^;$&~_/\\]', '', cleaned_text, flags=re.DOTALL | re.MULTILINE)
  cleaned_text = re.sub(r'[-+=():,\']', ' ', cleaned_text, flags=re.DOTALL | re.MULTILINE)
  cleaned_text = re.sub(r'\[|\]', ' ', cleaned_text, flags=re.DOTALL | re.MULTILINE)
  cleaned_text = re.sub(r'\w%', ' percent', cleaned_text, flags=re.DOTALL | re.MULTILINE)
  cleaned_text = re.sub(r'%', 'percent', cleaned_text, flags=re.DOTALL | re.MULTILINE)
  return cleaned_text

df_note_events['TEXT'] = df_note_events.apply(lambda row : remove_anonymization(row['TEXT']), axis = 1)
df_note_events['TEXT'] = df_note_events.apply(lambda row : remove_doctor_quotes(row['TEXT']), axis = 1)
df_note_events['TEXT'] = df_note_events.apply(lambda row : remove_extra_commas(row['TEXT']), axis = 1)
df_note_events['TEXT'] = df_note_events.apply(lambda row : lower_all_text(row['TEXT']), axis = 1)
df_note_events['CATEGORY'] = df_note_events.apply(lambda row : lower_all_text(row['CATEGORY']), axis = 1)
df_note_events['DESCRIPTION'] = df_note_events.apply(lambda row : lower_all_text(row['DESCRIPTION']), axis = 1)
df_note_events['TEXT'] = df_note_events.apply(lambda row : clean_useless_words(row['TEXT']), axis = 1)
df_note_events['TEXT'] = df_note_events.apply(lambda row : remove_time(row['TEXT']), axis = 1)
df_note_events['TEXT'] = df_note_events.apply(lambda row : remove_parentheses_num(row['TEXT']), axis = 1)
df_note_events['TEXT'] = df_note_events.apply(lambda row : remove_spaces(row['TEXT']), axis = 1)
df_note_events['TEXT'] = df_note_events.apply(lambda row : format_paragraphs(row['TEXT']), axis = 1)
df_note_events['TEXT'] = df_note_events.apply(lambda row : process_enums(row['TEXT']), axis = 1)
df_note_events['TEXT'] = df_note_events.apply(lambda row : num_to_text(row['TEXT']), axis = 1)
df_note_events['TEXT'] = df_note_events.apply(lambda row : remove_special_chars(row['TEXT']), axis = 1)
df_note_events['TEXT'] = df_note_events.apply(lambda row : remove_spaces(row['TEXT']), axis = 1)

# Do dataframe-wide level removal of rare words
def get_vocab(df):
  word_dict = dict()
    
  for index, row in df.iterrows():
    clinical_note = row[-1]
    word_list = word_tokenize(clinical_note)
    for w in word_list:
      if w in word_dict.keys():
          word_dict[w] += 1
      else:
          word_dict[w] = 1
  return word_dict

def remove_rare_words(text, rare_words):
  paragraphs = text.splitlines(True)
  cleaned_paragraphs = ""
  for paragraph in paragraphs:
    if paragraph == "\n":
      continue
    word_list = word_tokenize(paragraph)
    for w in word_list :
        if w in rare_words:
            paragraph = re.sub('(\s+)'+ w +'(\s+)', ' ', paragraph)
            paragraph = re.sub('^'+ w +'(\s+)', '', paragraph)
            paragraph = re.sub('$(\s+)'+ w, '', paragraph)
    cleaned_paragraphs += paragraph
  return cleaned_paragraphs.strip()

# Get vocabulary
vocab = get_vocab(df_note_events)
rare_words = {key: value for key, value in vocab.items() if value < 5}

# Remove rare words
df_note_events['TEXT'] = df_note_events.apply(lambda row : remove_rare_words(row['TEXT'], rare_words.keys()), axis = 1)
# Remove spaces (again?)
df_note_events['TEXT'] = df_note_events.apply(lambda row : remove_spaces(row['TEXT']), axis = 1)
# Prepend a newline character so this plays nicely with the annotation processor
df_note_events['TEXT'] = df_note_events.apply(lambda row : "\n" + row['TEXT'], axis = 1)

# avoid having to copy files around manually
concept_path = os.path.join(parent_path, 'concept_annotation\\data')
sys.path.insert(0, concept_path)

output_path = os.path.join(concept_path, 'output.csv')
df_note_events.to_csv(output_path, index=False)

