import os
import pandas as pd

PROJECT_ROOT = "WRITE_YOUR_OWN_FILE_PATH_IN_ENGLISH"
PREDICT_DATA_DIR = os.path.join(PROJECT_ROOT, "Data", "predict_testdata")
PSE_TOOL_PATH = os.path.join(PROJECT_ROOT, "Pse-in-One-2.0", "Change the file name and file path to what you need.py")

def ensure_predict_data_dir():
    os.makedirs(PREDICT_DATA_DIR, exist_ok=True)

def remove_x_from_file(filename):
    ensure_predict_data_dir()
    with open(filename, 'r') as file:
        content = file.read()
        content_without_x = content.replace('X', '')
    with open(os.path.join(PREDICT_DATA_DIR, 'Change the file name and file path to what you need.txt'), 'w') as cleaned_file:
        cleaned_file.write(content_without_x)

def process_file(filename):
    ensure_predict_data_dir()
    copy_txt_file(filename, os.path.join(PREDICT_DATA_DIR, "Change the file name and file path to what you need.txt"))
    remove_x_from_file(filename)
    import sys
    python_executable = sys.executable
    cleaned_path = os.path.join(PREDICT_DATA_DIR, "Change the file name and file path to what you need.txt")
    temp_path = os.path.join(PREDICT_DATA_DIR, "Change the file name and file path to what you need.txt")
    os.system('{} "{}" "{}" "{}" Protein DR -max_dis 1 -f tab'.format(python_executable, PSE_TOOL_PATH, cleaned_path, temp_path))
    df = pd.read_csv(temp_path, sep='\t')
    df.to_csv(os.path.join(PREDICT_DATA_DIR, 'Change the file name and file path to what you need.csv'), index=False)
    print("Feature extraction successful!")

def copy_txt_file(source_file, destination_file):
    try:
        with open(source_file, 'r') as source:
            with open(destination_file, 'w') as destination:
                for line in source:
                    destination.write(line)
        print("File copied successfully")
    except IOError:
        print("File copy failed")
        
def print_even_lines(filename):
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            for i in range(1, len(lines), 2):
                print(lines[i].strip())
    except IOError:
        print("File not found or unable to read")
