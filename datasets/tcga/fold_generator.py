import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

PATIENT_CSV_PATH = 'single/patients.csv'
FOLD_SAVE_PATH = './folds/'

patients_df = pd.read_csv(PATIENT_CSV_PATH)
kf = KFold(n_splits=4, random_state=42, shuffle=True)

folds_patients = []
unique_patients = np.unique(patients_df['patient'].values)
get_patients = lambda x: unique_patients[x]
for i, (train_index, test_index) in enumerate(kf.split(unique_patients)):
    train_index, validation_index = train_test_split(train_index, test_size=0.2, random_state=42)
    folds_patients.append({
        'train': get_patients(train_index),
        'validation': get_patients(validation_index),
        'test': get_patients(test_index)}
    )

os.makedirs(FOLD_SAVE_PATH, exist_ok=True)
for i in range(4):
    train = patients_df.slide[patients_df.patient.isin(folds_patients[i]['train'])].values
    validation = patients_df.slide[patients_df.patient.isin(folds_patients[i]['validation'])].values
    test = patients_df.slide[patients_df.patient.isin(folds_patients[i]['test'])].values
    print(f'fold {i}')
    print(f'\ttrain {len(train) / len(patients_df)}')
    print(f'\tvalidation {len(validation) / len(patients_df)}')
    print(f'\ttest {len(test) / len(patients_df)}')
    print(
        f'\ttotal {(len(train) + len(validation) + len(test)) / len(patients_df)}, '
        f'{len(train) + len(validation) + len(test)}'
    )
    train_df = pd.DataFrame({'train': train}).reset_index()
    validation_df = pd.DataFrame({'validation': validation}).reset_index()
    test_df = pd.DataFrame({'test': test}).reset_index()
    fold_df = pd.concat([train_df, validation_df, test_df], axis=1)
    fold_df.to_csv(os.path.join(FOLD_SAVE_PATH, f'fold_{i}.csv'), index=False)
