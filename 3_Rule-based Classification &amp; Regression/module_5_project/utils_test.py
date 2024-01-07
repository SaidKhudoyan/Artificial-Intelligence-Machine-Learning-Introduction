import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets # for the buttons you see in the notebook
import os
import random
import plotly.express as px
import unittest


from tqdm import tqdm
from IPython.display import display, HTML, clear_output
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import linear_model, naive_bayes, ensemble, neighbors
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from unittest.mock import patch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score




# ==========================================
# UNIT TESTS
# ==========================================

def compare_nan_values(nan_values):
    """
    Task 2: Check if right amount of NaN values calculated
    

    """
    expected_nan_values = pd.Series({
    'Disease': 0,
    'Symptom_1': 0,
    'Symptom_2': 0,
    'Symptom_3': 0,
    'Symptom_4': 348,
    'Symptom_5': 1206,
    'Symptom_6': 1986,
    'Symptom_7': 2652,
    'Symptom_8': 2976,
    'Symptom_9': 3228,
    'Symptom_10': 3408,
    'Symptom_11': 3726,
    'Symptom_12': 4176,
    'Symptom_13': 4416,
    'Symptom_14': 4614,
    'Symptom_15': 4680,
    'Symptom_16': 4728,
    'Symptom_17': 4848 
    })
    
    assert nan_values.equals(expected_nan_values), "Test failed (0/1): NaN counts do not match expected values"
    print("=" * 30)
    print("Test passed (1/1): NaN counts match expected values")
    
def compare_distribution(disease_distribution):
    """
    Task 3: Check wether calculated disease distribution is correct.
    

    """
    expected_dist = pd.Series({
    'Fungal infection': 120,
    'Hepatitis C': 120,
    'Hepatitis E': 120,
    'Alcoholic hepatitis': 120,
    'Tuberculosis': 120,
    'Common Cold': 120,
    'Pneumonia': 120,
    'Dimorphic hemmorhoids(piles)': 120,
    'Heart attack': 120,
    'Varicose veins': 120,
    'Hypothyroidism': 120,
    'Hyperthyroidism': 120,
    'Hypoglycemia': 120,
    'Osteoarthristis': 120,
    'Arthritis': 120,
    '(vertigo) Paroymsal  Positional Vertigo': 120,
    'Acne': 120,
    'Urinary tract infection': 120,
    'Psoriasis': 120,
    'Hepatitis D': 120,
    'Hepatitis B': 120,
    'Allergy': 120,
    'hepatitis A': 120,
    'GERD': 120,
    'Chronic cholestasis': 120,
    'Drug Reaction': 120,
    'Peptic ulcer diseae': 120,
    'AIDS': 120,
    'Diabetes': 120,
    'Gastroenteritis': 120,
    'Bronchial Asthma': 120,
    'Hypertension': 120,
    'Migraine': 120,
    'Cervical spondylosis': 120,
    'Paralysis (brain hemorrhage)': 120,
    'Jaundice': 120,
    'Malaria': 120,
    'Chicken pox': 120,
    'Dengue': 120,
    'Typhoid': 120,
    'Impetigo': 120
    })

    expected_values = expected_dist.values
    calculated_values = disease_distribution.values

    expected_indices = expected_dist.index
    calculated_indices = disease_distribution.index

    calculated_indices_cleaned = calculated_indices.str.strip()

    assert (expected_values == calculated_values).all(), "Test failed (0/1): Disease distribution values do not match expected result"
    assert (expected_indices == calculated_indices_cleaned).all(), "Test failed (0/1): Disease distribution indices do not match expected result. The ordering might be wrong."

    print("=" * 30)
    print("Test passed (1/1): Disease distribution matches expected result")


def test_symptom_frequencies(symp_freq):
    """
    Task 4: Check wether symptom frequency is correct
    
    """
    expected_symptom_freq = {' fatigue': 1932, ' vomiting': 1914, ' high_fever': 1362, ' loss_of_appetite': 1152, 
                             ' nausea': 1146, ' headache': 1134, ' abdominal_pain': 1032, ' yellowish_skin': 912, 
                             ' yellowing_of_eyes': 816, ' chills': 798, ' skin_rash': 786, ' malaise': 702, 
                             ' chest_pain': 696, ' joint_pain': 684, ' sweating': 678, 'itching': 678, 
                             ' dark_urine': 570, ' diarrhoea': 564, ' cough': 564, ' irritability': 474, 
                             ' muscle_pain': 474, ' excessive_hunger': 462, ' lethargy': 456, ' weight_loss': 456, 
                             ' breathlessness': 450, ' phlegm': 354, ' mild_fever': 354, ' swelled_lymph_nodes': 348,
                             ' blurred_and_distorted_vision': 342, ' loss_of_balance': 342, ' dizziness': 336, 
                             ' abnormal_menstruation': 240, ' muscle_weakness': 234, ' depression': 234, 
                             ' red_spots_over_body': 234, ' fast_heart_rate': 234, ' back_pain': 228, 
                             ' stiff_neck': 228, ' neck_pain': 228, ' constipation': 228, ' family_history': 228, 
                             ' obesity': 228, ' painful_walking': 228, ' swelling_joints': 228, ' restlessness': 228,
                             ' mood_swings': 228, ' continuous_sneezing': 222, ' stomach_pain': 222, ' acidity': 222,
                             ' indigestion': 222, ' burning_micturition': 216, ' sinus_pressure': 120, 
                             ' brittle_nails': 120, ' palpitations': 120, ' slurred_speech': 120, 
                             ' swollen_extremeties': 120, ' stomach_bleeding': 120, ' coma': 120, 
                             ' redness_of_eyes': 120, ' enlarged_thyroid': 120, ' blood_in_sputum': 120, 
                             ' rusty_sputum': 120, ' receiving_blood_transfusion': 120, ' throat_irritation': 120, 
                             ' loss_of_smell': 120, ' congestion': 120, ' runny_nose': 120, ' receiving_unsterile_injections': 120, 
                             ' pain_behind_the_eyes': 120, ' increased_appetite': 120, ' polyuria': 120, ' yellow_urine': 114, 
                             ' knee_pain': 114, ' weight_gain': 114, ' cold_hands_and_feets': 114, ' belly_pain': 114, 
                             ' puffy_face_and_eyes': 114, ' internal_itching': 114, ' passage_of_gases': 114, ' anxiety': 114, 
                             ' drying_and_tingling_lips': 114, ' hip_joint_pain': 114, ' swollen_legs': 114, ' movement_stiffness': 114, 
                             ' unsteadiness': 114, ' bladder_discomfort': 114, ' continuous_feel_of_urine': 114, ' skin_peeling': 114, 
                             ' silver_like_dusting': 114, ' small_dents_in_nails': 114, ' inflammatory_nails': 114, ' blister': 114, 
                             ' red_sore_around_nose': 114, ' prominent_veins_on_calf': 114, ' yellow_crust_ooze': 114, ' bruising': 114, 
                             ' irregular_sugar_level': 114, ' visual_disturbances': 114, ' lack_of_concentration': 114, ' mucoid_sputum': 114, 
                             ' fluid_overload': 114, ' history_of_alcohol_consumption': 114, ' distention_of_abdomen': 114, 
                             ' swelling_of_stomach': 114, ' acute_liver_failure': 114, ' altered_sensorium': 114, ' toxic_look_(typhos)': 114, 
                             ' pain_during_bowel_movements': 114, ' pain_in_anal_region': 114, ' bloody_stool': 114, ' irritation_in_anus': 114,
                             ' cramps': 114, ' weakness_of_one_body_side': 108, ' weakness_in_limbs': 108, ' scurring': 108, ' blackheads': 108,
                             ' pus_filled_pimples': 108, ' dischromic _patches': 108, ' spinning_movements': 108, ' nodal_skin_eruptions': 108,
                             ' swollen_blood_vessels': 108, ' dehydration': 108, ' shivering': 108, ' watering_from_eyes': 108, 
                             ' sunken_eyes': 108, ' ulcers_on_tongue': 108, ' spotting_ urination': 108, ' extra_marital_contacts': 108, 
                             ' muscle_wasting': 108, ' patches_in_throat': 108, ' foul_smell_of urine': 102}

    symptom_frequencies = pd.Series(expected_symptom_freq)

    assert symp_freq.equals(symptom_frequencies), "Test failed (0/1): Symptom frequencies do not match expected result"
    print("=" * 30)
    print("Test passed (1/1): Symptom frequencies match expected result")


def test_data_preprocessing(result_df):
    """
    Task 6: Check for removal of Whitespaces and NaN values
    
    """
    count = 0
    # Check for white_space removal
    whitespace_present = any(any(str(val).isspace() for val in row) for _, row in result_df.iterrows())

    if whitespace_present:
        print("="*30)
        print(f"Test failed ({count}/2): DataFrame contains whitespace values")
    else:
        count +=1
        print("="*30)
        print(f"Test passed ({count}/2) DataFrame does not contain whitespace values")

    # Check for NaN values removal
    no_nan_values = not result_df.isna().any().any()

    if no_nan_values:
        count+=1
        print("="*30)
        print(f"Test passed ({count}/2): DataFrame does not contain any NaN values")
    else:
        print("="*30)
        print(f"Test failed ({count}/2)DataFrame contains NaN values")


def check_dataframe_int_values(df):
    """
    Task 7: Check wether all symptoms are integers
    
    """
    non_int_values = df[df.columns[1:]].applymap(lambda x: not isinstance(x, int))
    
    if non_int_values.any().any():
        non_int_count = non_int_values.sum().sum()
        print("="*30)
        print(f"Test failed (0/1): There are {non_int_count} non-integer values in the DataFrame (except the first column).")
    else:
        print("="*30)
        print("Test passed (1/1): All values in the DataFrame (except the first column) are integers.")


def split_encoding_test(encoded_y, x_train, y_train, x_test, y_test):
    """
    Task 8: Check train_test_split and label_encoding
    
    """
    count = 0
    # Check for encoding
    if all(isinstance(element, np.int64) for element in encoded_y):
        count += 1
        print("="*30)
        print(f"Test passed ({count}/2): Labels are properly encoded")
    else:
        print("="*30)
        print(f"Test failed ({count}/2): Labels are not properly encoded")

        
    # Check for train-test split sizes
    if x_train.shape != (4182, 17) or x_test.shape != (738,17) or y_train.shape != (4182,) or y_test.shape != (738, ):
        print("="*30)
        print(f"Test failed ({count}/2): Train or test set sizes are not appropriate")
    else:
        count+=1
        print("="*30)
        print(f"Test passed ({count}/2): Train and test set sizes are appropriate")


def test_model_implementation(svc_model, dt_model, ensemble):
    """
    Task 9: Check correct model implementation
    
    """
    assert svc_model.C == 1, "SVC model has incorrect hyperparameter C."
    assert dt_model.criterion == "gini", "DecisionTree model has incorrect splitting criterion."
    assert dt_model.max_depth == 7, "DecisionTree model has incorrect max depth."
    assert dt_model.min_samples_split == 2, "DecisionTree model has incorrect min_samples_split."
    assert dt_model.min_samples_leaf == 2, "DecisionTree model has incorrect min_samples_leaf."
    assert ensemble.voting == "hard", "Ensemble VotingClassifier has incorrect voting parameter."
    
    print("="*30)
    print("Test passed (1/1): All models have been implemented correctly.")


def compare_models(model1, model2, x_test, y_test):
    """
    Task 13: Check model performance
    
    """
    preds_model1 = model1.predict(x_test)
    preds_model2 = model2.predict(x_test)
    
    accuracy_model1 = accuracy_score(y_test, preds_model1)
    precision_model1 = precision_score(y_test, preds_model1, average='weighted')
    recall_model1 = recall_score(y_test, preds_model1, average='weighted')
    f1_score_model1 = f1_score(y_test, preds_model1, average='weighted')
    
    accuracy_model2 = accuracy_score(y_test, preds_model2)
    precision_model2 = precision_score(y_test, preds_model2, average='weighted')
    recall_model2 = recall_score(y_test, preds_model2, average='weighted')
    f1_score_model2 = f1_score(y_test, preds_model2, average='weighted')
    
    print("="*30)
    print("Evaluation Metrics of your model:")
    print(f"Accuracy: {accuracy_model1}")
    print(f"Precision: {precision_model1}")
    print(f"Recall: {recall_model1}")
    print(f"F1-Score: {f1_score_model1}\n")
    
    print("="*30)
    print("Evaluation Metrics of SVC:")
    print(f"Accuracy: {accuracy_model2}")
    print(f"Precision: {precision_model2}")
    print(f"Recall: {recall_model2}")
    print(f"F1-Score: {f1_score_model2}\n")
    
    better_metrics_count = 0

    if accuracy_model1 > accuracy_model2:
        print("Your model has better accuracy.")
        better_metrics_count += 1
    elif accuracy_model1 < accuracy_model2:
        print("SVC has better accuracy.")
    else:
        print("Both models have the same accuracy.")

    if precision_model1 > precision_model2:
        print("Your model has better precision.")
        better_metrics_count += 1
    elif precision_model1 < precision_model2:
        print("SVC has better precision.")
    else:
        print("Both models have the same precision.")

    if recall_model1 > recall_model2:
        print("Your model has better recall.")
        better_metrics_count += 1
    elif recall_model1 < recall_model2:
        print("SVC has better recall.")
    else:
        print("Both models have the same recall.")

    if f1_score_model1 > f1_score_model2:
        print("Your model has a better F1-Score.")
        better_metrics_count += 1
    elif f1_score_model1 < f1_score_model2:
        print("SVC has a better F1-Score.")
    else:
        print("Both models have the same F1-Score.")
    
    # Check if Model 1 passes the test
    if better_metrics_count >= 3:
        print("="*30)
        print("\nTest passed (1/1): Your model defeated the SVC model. You're ready to continue.")
    else:
        print("="*30)
        print("\nTest failed (0/1): Try to further enhance your model performance.")


# ==========================================
# Helper functions
# ==========================================

def get_bar_color(nan_count):
    """
    Task 2: Plot nan_values
    
    """
    if nan_count < 1500:
        return 'green'
    elif nan_count < 3500:
        return 'orange'
    else:
        return 'red'
    
    