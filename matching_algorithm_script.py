import csv
import random
import pandas as pd
import numpy as np
import time
import sys
from copy import deepcopy

## match students to their rotations
## have to define a few things first

INDEX_BLOCK_ONE = 0
INDEX_BLOCK_TWO = 1
FULL_BLOCK = 2
NO_OPEN_BLOCK = 2
MAX_ITERATIONS = 1

TEST = False
DATA_PATH = 'student_data/block_one/data.csv'
EXPORT_PATH = 'results/block_one/'

class Rotation:
    
    def __init__(self,
                 name,
                 location,
                 maximum_students,
                 subspeciality):
        self.name = name
        self.location = location
        self.maximum_students = maximum_students
        self.subspeciality = subspeciality
        self.block_one_count = 0
        self.block_two_count = 0
        self.full_block_one = False
        self.full_block_two = False
        self.full_rotation = False
    
    def add_specific_block(self, block):
        if block is None:
            print('ERROR, block is None, empty blocks should not be here')
            sys.exit(1)
        # returns True if OK 
        # returns False if not OK
        can_add = None
        if block == INDEX_BLOCK_ONE and self.full_block_one is False:
            self.block_one_count += 1
            self.update_block()
            can_add = True
        elif block == INDEX_BLOCK_TWO and self.full_block_two is False:
            self.block_two_count += 1
            self.update_block()
            can_add = True
        else:
            can_add = False
        return can_add

    def update_block(self):
        
        if self.block_one_count >= self.maximum_students:
            self.full_block_one = True  
        if self.block_two_count >= self.maximum_students:
            self.full_block_two = True 
        if self.full_block_one and self.full_block_two:   
            self.full_rotation = True
            
        return


# defines all rotations possible
# the true false flag is subspecialty

rotation_information = [
    Rotation("VA General Surgery", "VA", 1, False),
    Rotation("VA VSU", "VA", 1, True),
    Rotation("DRH General", "DRH", 1, False),
    Rotation("DRH VSU", "DRH", 1, True),
    Rotation("DRAH ACS", "DRAH", 2, False),
    Rotation("DRAH Surgical Oncology", "DRAH", 1, False),
    Rotation("DUH Surgical Oncology", "DUH", 2, False),
    Rotation("DUH Transplant", "DUH", 1, True),
    Rotation("DUH Colorectal", "DUH", 2, False),
    Rotation("DUH Pediatrics", "DUH", 1, False),
    Rotation("DUH Trauma/ACS", "DUH", 2, False),
    Rotation("DUH VSU", "DUH", 3, True),
    Rotation("DUH Breast/Endocrine", "DUH", 2, True),
    Rotation("DUH Cardiac", "DUH", 2, True),
    Rotation("DUH Thoracic", "DUH", 2, True)
]

def calculate_row_cost(row, df_student_preferences):
    cost1 = df_student_preferences.loc[
        df_student_preferences['Name'] == row['Name']][row['Block One']].values[0]
    cost2 = df_student_preferences.loc[
        df_student_preferences['Name'] == row['Name']][row['Block Two']].values[0]
    return cost1+cost2

def calculate_solution_cost(student_assignments,
                            df_student_preferences):
    student_assignments['Cost'] = student_assignments.apply(
        calculate_row_cost, args=(df_student_preferences,), axis = 1
    )
    total_cost = student_assignments['Cost'].sum()
    return total_cost

def get_general_block(rotations):
    
    # these are open spots 
    
    block_one_general = 0
    block_two_general = 0
    block_one_subspecialty = 0
    block_two_subspecialty = 0
    
    for rotation in rotations:
        if rotation.subspeciality is False:
            block_one_general += (
                rotation.maximum_students - rotation.block_one_count)
            block_two_general += (
                rotation.maximum_students - rotation.block_two_count)
        else:
            block_one_subspecialty += (
                rotation.maximum_students - rotation.block_one_count)
            block_two_subspecialty += (
                rotation.maximum_students - rotation.block_two_count)
    
    general_block = None
    
    if block_one_general == 0 and block_two_general == 0:
        # no space in general
        print('No space in general')
        sys.exit(1)
        return None
    if block_one_general > 0 and block_two_general > 0:
        # space in both general blocks
        # so we then decide based on subspecialty block size
        
        # first if both subspecialty blocks are open, add to either
        if block_one_subspecialty and block_two_subspecialty >0:
            general_block = np.random.choice(
                [INDEX_BLOCK_ONE, INDEX_BLOCK_TWO])
        # otherwise, add general rotation block to open up subspecialty
        elif block_one_subspecialty == 0:
            general_block = INDEX_BLOCK_TWO
        elif block_two_subspecialty == 0:    
            general_block = INDEX_BLOCK_ONE 
    
    elif block_one_general > 0 and block_two_general == 0:
        general_block = INDEX_BLOCK_ONE
    elif block_one_general == 0 and block_two_general > 0:
        general_block = INDEX_BLOCK_TWO
    else:
        print('ERROR, something went wrong with general block selection')
        sys.exit(1)
        return None
        
    return general_block

def generate_solution(df_student_preferences,
                      name_list_randomized):

    # iterate by students
    # Rules

    # 1. one rotation must be at main campus, although this is not necessary
    # 2. one subspeciality, one general (main restriction)
    # 3. regional can take an extra, DRAH 3 total (tentative 2 acs 1 surg onc),
    #       vascular and breast can take extra

    # will do general and subspeciality separately
    # plan is to remove rotation from list if it is full
    
    student_preference_copy = df_student_preferences.copy()
    rotations = deepcopy(rotation_information)  
    student_assignments = pd.DataFrame(
        {
            'Name': name_list_randomized,
            'Block One': [None] * len(name_list_randomized),
            'Block Two': [None] * len(name_list_randomized)
        }
    )

    for name in name_list_randomized:
        # we re-define in each iteration so that we can remove rotations
        # from df student preferences
        # general first, so in this case both of their assignments and blocks
        # are open
        
        # have to re-do the names each time you do a new student
        # because we may remove full rotations 
        
        GENERAL_ROTATIONS_NAME = [
            r.name for r in rotations if not r.subspeciality]
        SUBSPECIALITY_ROTATIONS_NAME = [
            r.name for r in rotations if r.subspeciality]
        
        general_preferences = student_preference_copy[
            student_preference_copy['Name'] == name][GENERAL_ROTATIONS_NAME]
        if isinstance(general_preferences, pd.DataFrame):
            general_pref_sorted = general_preferences.iloc[0].sort_values()
        else:
            general_pref_sorted = general_preferences.sort_values()
        
        preferred_choice = 0
        
        # this will also randomly choose a block if needed
        general_block = get_general_block(rotations)
        subspec_block = None
            
        if general_block is None:
            # nowhere left to add patients
            print('Nowhere left to add patients?')
            sys.exit(1)
            return None
        if general_block == INDEX_BLOCK_TWO:
            gen_col = 'Block Two'
            subspec_col = 'Block One'
            subspec_block = INDEX_BLOCK_ONE
        else:
            gen_col = 'Block One'
            subspec_col = 'Block Two'
            subspec_block = INDEX_BLOCK_TWO
    
        preferred_choice = 0
        while True:
            # we need a while loop here because there's a chance that
            # the rotation is not full (so it's still in the list) BUT
            # that the BLOCK is full so we cannot use it, or the subspec
            # blocks are all full (hence the logic for get_general_block)
            
            rotation_name = general_pref_sorted.index[preferred_choice]
            rotation = [r for r in rotations if r.name == rotation_name][0]
            rotation_index = rotations.index(rotation)
            added = rotation.add_specific_block(general_block)
            if added is True:
                student_assignments.loc[
                    student_assignments['Name'] == name, gen_col] = rotation_name
                if rotation.full_rotation:
                    del rotations[rotation_index]
                    student_preference_copy.drop(
                        rotation_name, axis=1, inplace=True)
                break
            else:
                preferred_choice += 1
                if preferred_choice >= len(general_pref_sorted):
                    print('ERROR, no general rotations left')
                    student_assignments = None
                    return student_assignments
        
        # next subspeciality
        # we do this again because student preferences have changed
        # with the drop of the column
        subspecialty_preferences = student_preference_copy[
            student_preference_copy['Name'] == name][SUBSPECIALITY_ROTATIONS_NAME]
        # Convert to Series and sort values
        if isinstance(subspecialty_preferences, pd.DataFrame):
            subspec_pref_sorted = subspecialty_preferences.iloc[0].sort_values()
        else:
            subspec_pref_sorted = subspecialty_preferences.sort_values()
        
        preferred_choice = 0
        while True:
            # we need a while loop here because there's a chance that
            # the rotation is not full (so it's still in the list) BUT
            # that the BLOCK is full so we cannot use it
            rotation_name = subspec_pref_sorted.index[preferred_choice]
            rotation = [r for r in rotations if r.name == rotation_name][0]
            rotation_index = rotations.index(rotation)
            if subspec_block is None:
                print('ERROR, subspecialty block is None')
                sys.exit(1)
            added = rotation.add_specific_block(subspec_block)
            if added is True:
                student_assignments.loc[
                    student_assignments['Name'] == name, subspec_col] = rotation_name
                if rotation.full_rotation:
                    del rotations[rotation_index]
                    student_preference_copy.drop(
                        rotation_name, axis=1, inplace=True)
                break
            else:
                preferred_choice += 1
                # TODO: 12/11/24 updated this to len(subspec_pref_sorted) + 0
                # instead of -1 because preferred choice is updated above
                if preferred_choice >= len(subspec_pref_sorted):
                    student_assignments = None
                    return student_assignments
          
    return student_assignments

def simple_matching_algorithm(df_student_preferences):
    
    name_list = df_student_preferences['Name'].unique()
    if len(name_list) != len(df_student_preferences):
        print('Error, duplicated names?')
        sys.exit(1)
            
    # start solution with a random order of names
    # this is NOT a very good solution, but too complicated to figure out
    # how to Munkres with these constraints
    
    
    iteration = 1
    MAX_ITERATIONS = 10000
    while iteration < MAX_ITERATIONS:
        random_name_list = name_list.copy()
        np.random.shuffle(random_name_list)
        student_assignments = generate_solution(
            df_student_preferences,
            random_name_list)
        if student_assignments is not None:
            break
        else:
            iteration += 1
            if iteration >= MAX_ITERATIONS:
                print('ERROR, max iterations reached')
                sys.exit(1)
    # can come back to this later
    solution_cost = calculate_solution_cost(student_assignments,
                                            df_student_preferences)
      
    return [solution_cost, student_assignments]
    
    
def print_final_rotations(student_assignments):
    student_assignments.to_csv(
        EXPORT_PATH + 'final_assignments.csv', index=False)
     

def calculate_capacity(df_students):
    maximum_DUH_students = 0
    for rotation in rotation_information:
        if rotation.location =='DUH':
            maximum_DUH_students += rotation.maximum_students
    students = len(df_students)
    
    if students > maximum_DUH_students *2:
        print(f'Maximum DUH capacity is {maximum_DUH_students}, and there are {students} students')
        sys.exit(1)
    else:
        return True

def run_algorithm(student_preferences_path = DATA_PATH,
                  test = TEST):

    if test:
        student_preferences_path = 'test_data/test_data.csv'
    
    # import dataframe with everybody's name and their preferred number
    df_student_preferences = pd.read_csv(student_preferences_path)
    
    # quick check that we have enough spots for DUH to accomidate the students
    # will abort if not
    calculate_capacity(df_student_preferences)
    
    attempts_to_run = 10000
    best_assignment = None
    best_cost = np.inf
    
    for attempt in range(attempts_to_run):
        [solution_cost, student_assignments] = simple_matching_algorithm(
            df_student_preferences)
        if solution_cost < best_cost:
            print(f'Have found better solution at attempt {attempt}, with a cost of {solution_cost}')
            best_cost = solution_cost
            best_assignment = student_assignments

    print_final_rotations(best_assignment)
    
run_algorithm()