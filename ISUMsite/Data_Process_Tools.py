# -*- coding: utf-8 -*-
import os
import pandas as pd

# Remove bogus code X
def remove_x_from_file(filename):
    # Read the original file and remove all 'X' characters
    with open(filename, 'r') as file:
        content = file.read()
        content_without_x = content.replace('X', '')
    # Write the processed data into a temporary file
    with open('./Data/Processed_Data/temp_cleaned.txt', 'w') as cleaned_file:
        cleaned_file.write(content_without_x)
# Feature extraction
def process_file(filename):
    # Copy the original file
    copy_txt_file(filename, "./Data/Processed_Data/copy.txt")
    # Remove all 'X' characters from the file
    remove_x_from_file(filename)
    # Run Python script
    os.system('python Pse-in-One-2.0/nac.py ./Data/Processed_Data/temp_cleaned.txt Protein DR -max_dis 1 -f tab -labels 0 -out ./Data/Processed_Data/temp.txt')
    # Read the txt file and convert it to a csv file
    df = pd.read_csv('./Data/Processed_Data/temp.txt', sep='\t')
    df.to_csv('./Data/Processed_Data/featured.csv', index=False)
    print("Feature extraction successful!")
# -*- coding: utf-8 -*-
# Copy a copy of the original dataset
def copy_txt_file(source_file, destination_file):
    try:
        with open(source_file, 'r') as source:
            with open(destination_file, 'w') as destination:
                for line in source:
                    destination.write(line)
        print("File copied successfully")
    except IOError:
        print("File copy failed")


# Output content of even lines
def print_even_lines(filename):
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            for i in range(1, len(lines), 2):  # Start from index 1, step 2, i.e., even lines
                print(lines[i].strip())
    except IOError:
        print("File not found or unable to read")