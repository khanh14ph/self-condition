import zipfile
import os
from tqdm import tqdm
def create_zip_archive(files, output_zip_name):
    """
    Create a zip archive containing the specified files.
    
    Args:
        files (list): List of file paths to be zipped
        output_zip_name (str): Name of the output zip file
    
    Returns:
        bool: True if successful, False if there was an error
    """
    try:
        # Create a ZipFile object in write mode
        with zipfile.ZipFile(output_zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Iterate through the list of files
            for file in tqdm(files):
                # Check if file exists
                if os.path.exists(file):
                    # Add file to zip archive
                    zipf.write(file, os.path.basename(file))
                else:
                    print(f"Warning: File not found - {file}")
        
        print(f"Successfully created zip archive: {output_zip_name}")
        return True
    
    except Exception as e:
        print(f"Error creating zip archive: {str(e)}")
        return False

# Example usage
if __name__ == "__main__":
    import pandas as pd
    df=pd.read_csv("/home4/khanhnd/self-condition/data/multi_lingual.tsv",sep="\t")
    # List of files to zip
    files_to_zip = list(df["audio_filepath"])
    
    # Create zip archive
    create_zip_archive(files_to_zip, 'multi_lingual.zip')