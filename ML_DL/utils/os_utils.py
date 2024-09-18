import os


def get_folder_number(directory: str = "logs") -> int:
    return len(os.listdir(directory)) + 1


def get_identifier(args: dict) -> str:
    result: str = ''
    for key, value in args.items():
        if key != "locations_to_train" and key != "locations_to_predict":
            result = result + str(value) + '_'

    result = result[:-1]

    return result


def remove_files_in_directory(directory_path) -> None:
    # Get the list of files in the directory
    file_list = os.listdir(directory_path)

    # Iterate through the files and remove only files, not directories
    for file_name in file_list:
        file_path = os.path.join(directory_path, file_name)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error: {e}")


def logs_folder_structure(identifier: str) -> str:

    if not os.path.exists("../logs"):
        os.makedirs("../logs")

    if not os.path.exists("../logs/"):
        os.makedirs("../logs/")

    if not os.path.exists("../logs/" + identifier):
        os.makedirs("../logs/" + identifier)

    number_folder: int = get_folder_number("../logs/" + identifier)
    log_dir: str = "../logs/" + identifier + "/version_" + str(number_folder)
    os.makedirs(log_dir + "/checkpoints/")

    return log_dir


def remove_files_in_directory(directory_path) -> None:
    # Get the list of files in the directory
    file_list = os.listdir(directory_path)

    # Iterate through the files and remove only files, not directories
    for file_name in file_list:
        file_path = os.path.join(directory_path, file_name)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error: {e}")

