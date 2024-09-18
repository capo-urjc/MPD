import os
import pickle


def save_model(folder_path, model_name_2_save, trained_model):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(folder_path + '/' + model_name_2_save, "wb") as file:
        pickle.dump(trained_model, file)







