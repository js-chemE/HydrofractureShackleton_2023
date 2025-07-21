import zipfile
import os
import shutil

def unpack_zips_in_folder(folder_path: str, maintain_structure: bool =False) -> None:
    """
    Unpacks all ZIP files in the given folder.

    :param folder_path: The path where .zip files are located.
    :param maintain_structure: If False, all files are flattened into folder_path.
                               If True, internal folder structure is preserved.
    """
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.zip'):
            zip_file_path = os.path.join(folder_path, file_name)

            try:
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    if maintain_structure:
                        zip_ref.extractall(folder_path)
                        print(f"Extracted {file_name} with structure.")
                    else:
                        # Create a temporary extraction directory
                        temp_extract_dir = os.path.join(folder_path, '__temp_unpack')
                        os.makedirs(temp_extract_dir, exist_ok=True)
                        zip_ref.extractall(temp_extract_dir)

                        # Move all files to folder_path, flattening hierarchy
                        for root, _, files in os.walk(temp_extract_dir):
                            for f in files:
                                src = os.path.join(root, f)
                                dst = os.path.join(folder_path, f)
                                if os.path.exists(dst):
                                    print(f"Overwriting existing file: {dst}")
                                shutil.move(src, dst)
                                print(f"Moved {f} to {folder_path}")

                        # Clean up temp folder
                        shutil.rmtree(temp_extract_dir)

                print(f"Unpacked {file_name} successfully.")

            except Exception as e:
                print(f"Failed to extract {file_name}: {e}")