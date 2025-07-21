import app

app.setup_logger(use_console_handler=True, use_file_handler=False)

folder = r"D:\PhD\21_Experiments\TidesDamageDriver\01_raw\L8S2-mosaics\2023"
app.files.unpack_zips_in_folder(folder)