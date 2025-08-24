import csv
import os
import re
import pandas as pd

def process_csv(input_path: str, output_path: str, columns_to_keep: list):
    """
    Reads a CSV file, keeps specified columns, and saves it to a new CSV file.

    Args:
        input_path (str): Path to the input CSV file.
        output_path (str): Path to save the new CSV file.
        columns_to_keep (list): A list of column names to keep.
    """
    try:
        # Read the input CSV
        df = pd.read_csv(input_path)

        # Check if required columns exist
        missing_cols = [col for col in columns_to_keep if col not in df.columns]
        if missing_cols:
            print(f"Error: The following columns were not found in the input file: {', '.join(missing_cols)}")
            return

        # Keep only the specified columns
        new_df = df[columns_to_keep]

        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save to the new CSV file without the index
        new_df.to_csv(output_path, index=False)
        print(f"Successfully created '{output_path}' with columns: {', '.join(columns_to_keep)}")

    except FileNotFoundError:
        print(f"Error: The file '{input_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def convert_txt_to_csv(input_txt_path, output_csv_path):
    """
    Converts a txt file to a csv file.

    The input txt file should have one number and one text per line,
    separated by a space. The script will create a two-column csv file,
    with the text first and the number second.

    Args:
        input_txt_path (str): The path to the input .txt file.
        output_csv_path (str): The path where the output .csv file will be saved.
    """
    try:
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_csv_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(input_txt_path, 'r', encoding='utf-8') as infile, \
             open(output_csv_path, 'w', newline='', encoding='utf-8') as outfile:
            
            csv_writer = csv.writer(outfile)
            
            # Optional: Write a header row if you need one
            # csv_writer.writerow(['text', 'number'])

            for line in infile:
                line = line.strip()
                if not line:
                    continue
                
                # Use regex to robustly split the number and the text, handling various whitespaces
                match = re.match(r'^(\d+)\s+(.*)$', line)
                if match:
                    number = match.group(1)
                    text = match.group(2)
                    # Write the row with text first, then number
                    csv_writer.writerow([text, number])
                else:
                    print(f"Skipping malformed line: {line}")

        print(f"Successfully converted '{input_txt_path}' to '{output_csv_path}'")

    except FileNotFoundError:
        print(f"Error: The file '{input_txt_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    # --- Task 1: Convert TXT to CSV ---
    # print("Running TXT to CSV conversion...")
    # input_txt = 'data/VSQ/test.txt'
    # output_csv_from_txt = 'data/VSQ/test.csv'
    # convert_txt_to_csv(input_txt, output_csv_from_txt)

    # --- Task 2: Process CSV to keep specific columns ---
    # --- PLEASE MODIFY THE FILE PATHS AND COLUMNS BELOW ---
    print("\nRunning CSV processing...")
    input_csv = 'output/banking77_clustering_20250824_225137.csv' # <-- MODIFY THIS
    output_csv_processed = 'output/banking77_result.csv' # <-- MODIFY THIS
    columns = ['query_content', 'category_id'] # <-- MODIFY THIS if needed
    # --- END OF MODIFICATION ---
    
    process_csv(input_csv, output_csv_processed, columns)
