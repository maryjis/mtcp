from src.logger import logger
import os
import pandas as pd
from typing import Optional, Union

def get_patients_from_BraTS(data_path, modalities, with_mask=True, df_with_test:Union[pd.DataFrame, str, None]=None):
    patients = os.listdir(data_path)
    logger.info("Total patients", len(patients))

    #get patients only with target modalities
    patients_with_needed_modalities = []
    needed_modalities = set(modalities)
    if with_mask: needed_modalities.add("seg") #segmentation mask used to compute center of tumor
    for patient in patients:
        available_modalities = set([x.split("-")[-1].split(".")[0] for x in os.listdir(os.path.join(data_path, patient))])
        if needed_modalities.intersection(available_modalities) == needed_modalities:
            patients_with_needed_modalities.append(patient)
    patients = patients_with_needed_modalities
    logger.info(f"Patients with all needed modalities: {len(patients)}")

    if df_with_test is not None:
        if isinstance(df_with_test, str):
            df_with_test = pd.read_csv(df_with_test)

        dataframe_test = df_with_test[df_with_test["group"] == "test"]
        # get patient ids where MRI is not NaN
        dataframe_test = dataframe_test[~dataframe_test["MRI"].isna()]
        patients_to_exclude = [
            patient_path.split("/")[-1] for patient_path in dataframe_test.MRI.values
        ]
        patients = [patient for patient in patients if patient not in patients_to_exclude]
        logger.info("Included patients (pre-train)", len(patients))
        logger.info("Excluded patients (further test)", len(patients_to_exclude))

    return patients