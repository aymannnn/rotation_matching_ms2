import pandas as pd
import random

rotation_names = [
    'VA General Surgery',
    'VA VSU',
    'DRH General',
    'DRH VSU',
    'DRAH ACS',
    'DRAH Surgical Oncology',
    'DUH Surgical Oncology',
    'DUH Transplant',
    'DUH Colorectal',
    'DUH Pediatrics',
    'DUH Trauma/ACS',
    'DUH VSU',
    'DUH Breast/Endocrine',
    'DUH Cardiac',
    'DUH Thoracic']

def generate_test_data(n_students = 25):
    number_of_rotations = len(rotation_names)
    n_students = n_students
    first_row = ['Name']
    for rotation in rotation_names:
        first_row.append(rotation)
    # will addend first row to dataframe later
    test_data = []     
    for i in range(n_students):
        row = [i]
        used_rank = []
        for index_rotation in range(number_of_rotations):
            random_number = random.randint(1, number_of_rotations)
            while random_number in used_rank:
                random_number = random.randint(1, 
                                               number_of_rotations)
            row.append(random_number)
            used_rank.append(random_number)
        test_data.append(row)
    df = pd.DataFrame(test_data,columns=(first_row))
    df.to_csv('test_data.csv',
              index=False)
                
generate_test_data()
