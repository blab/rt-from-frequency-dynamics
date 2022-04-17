import os


def make_path_if_absent(path):
    dirname = os.getcwd()
    file_path = os.path.join(dirname, path)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        print(f"{path} created.")
    return None


def make_model_directories(path):
    make_path_if_absent(path)
    make_path_if_absent(path + "/models")
