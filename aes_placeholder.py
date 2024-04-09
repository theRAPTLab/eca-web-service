import flask
import json
import os
import pickle
import random
import pprint as pp

import numpy as np

from aes_rl import utils
from aes_rl.aes import RL_AES

all_rl_aes = RL_AES()

def get_mutated_status():
    if "mutated" in flask.current_app.config:
        return 'mutate'
    else:
        return 'dont_mutate'

def set_mutated_status(flag):    
    if flag:
        flask.current_app.config['mutated'] = True    

def get_aes_action(aestype, tracelog) -> str:
    
    mutated = get_mutated_status()
    state = get_aes_state(tracelog)
    if aestype == 'AESHint':
        return all_rl_aes.get_next_hint(state)
    elif aestype == 'AESSetting':
        return get_game_setting(tracelog)
    elif aestype == 'AESDiseaseType':
        return get_disease_type(tracelog, "disease")
    elif aestype == 'AESSourceType':
        return get_disease_type(tracelog, "source")
    elif aestype == 'AESTreatmentType':
        return get_disease_type(tracelog, "treatment")
    elif aestype == 'AESTransmissionSource':
        return get_transmission_source(tracelog)
    elif aestype == 'AESIntroduceCharacter':
        return all_rl_aes.get_next_introduce_character(state)
    elif aestype == 'AESChooseBookContent':
        return get_book_content(tracelog)
    elif aestype == 'AESPretestScore':
        if get_pretest_scores(tracelog)['pretest_score']>5:
            return 'high'
        else:
            return 'low'
    elif aestype == 'AESDiseaseMutation':
        if mutated == "dont_mutate":
            mutated = all_rl_aes.get_next_disease_mutation(state)
            set_mutated_status(mutated == 'mutate')
        return mutated
    else:
        return ""


def get_book_content(tracelog):
    # TODO update logic of getting book content
    action = random.randint(0, 7)
    if action == 0:
        return "microbes_short"
    elif action == 1:
        return "microbes_long"
    elif action == 2:
        return "covid"
    elif action == 3:
        return "botulism"
    elif action == 4:
        return "salmonellosis_short"
    elif action == 5:
        return "salmonellosis_long"
    elif action == 6:
        return "influenza_long"
    elif action == 7:
        return "influenza_short"


def talked_to_kim_sick_test(tracelog: dict) -> (int, int):
    f_kim = 0
    f_sick = 0
    f_test = 0
    for h in tracelog["UtteranceHistory"]:
        if h["SpeakerName"] == "Kim":
            f_kim = 1
        elif h["SpeakerName"] in ["Teresa", "Greg", "Sam"]:
            f_sick = 1
        elif h["SpeakerName"] == "Elise" and "POSITIVE" in h["Utterance"]:
            f_test = 1
        if f_kim == 1 and f_sick == 1 and f_test == 1:
            # no need to search further
            break
    return f_kim, f_sick, f_test


def has_diagnosis_submitted(tracelog: dict) -> int:
    for h in tracelog["LogHistory"]:
        if h.get("DiagnosisError", False):
            return 1
    return 0


def get_aes_state(tracelog: dict) -> np.ndarray:
    mutated = get_mutated_status()
    survey = get_pretest_scores(tracelog)
    if survey.get("pretest_score", 0) > 5:
        f_pretest = 1
    else:
        f_pretest = 0

    f_kim, f_sick, f_test = talked_to_kim_sick_test(tracelog)

    f_worksheet = has_diagnosis_submitted(tracelog)
    # need to implement when using AES-DiseaseMutation
    if mutated=="dont_mutate":
        f_mutated = 0
    else:
        f_mutated = 1

    state = np.array([f_pretest, f_kim, f_sick, f_test, f_worksheet, f_mutated])
    return state


def get_pretest_scores(tracelog):
    workbook_data = None

    for lg in tracelog['LogHistory']:
        if (("WorkbookId" in lg) and 
            ("Target" in lg and lg["Target"] == "Workbook") and
            ("ActivityName" in lg and lg["ActivityName"] == "AffectPreSurvey") and 
            lg['IsSubmitted']):
                workbook_data = lg

    pretest_scores = {'location_pref':'Island'}            

    VGAME_EXP_SCORES = {"None":0,
                            "1 - 5 hours":2,
                            "5 - 10 hours":4,
                            "10 - 20 hours":6,
                            "Over 20 hours":8}
    if workbook_data is not None:
        game_exp_key = list(filter(lambda x: 'How many hours per week' in x, workbook_data.keys()))[0]

        pretest_scores['videogame_experience'] = VGAME_EXP_SCORES[workbook_data[game_exp_key]]
        pretest_scores['pretest_score'] = round(10*workbook_data['Score']/workbook_data['MaxPossibleScore']) 
    else:
        pretest_scores['videogame_experience'] = 8
        pretest_scores['pretest_score'] = 2 

    return pretest_scores


def get_pretest_scores_old(tracelog):
    workbook_data = None
    for h in tracelog["LogHistory"]:
        if (("WorkbookId" in h) and
                (h["ActivityName"] == "EngageNLE_pretest") and
                (h['IsSubmitted'])
           ):
            workbook_data = h
    # pp.pprint(workbook_data)
    pretest_scores = {}
    for k in workbook_data.keys():
        if k.startswith("Please select a student"):
            student_profile = workbook_data[k]
    print(student_profile)
    if student_profile == 'Eugene':
        pretest_scores['location_pref'] = 'Medical Clinic'  
        pretest_scores['videogame_experience'] = 8
        pretest_scores['pretest_score'] = 2              
    elif student_profile == 'Vincent':
        pretest_scores['location_pref'] = 'Island'
        pretest_scores['videogame_experience'] = 2
        pretest_scores['pretest_score'] = 8
    else:
        pretest_scores['location_pref'] = 'Island'
        pretest_scores['videogame_experience'] = 8
        pretest_scores['pretest_score'] = 2      

    #pp.pprint(pretest_scores)
    return pretest_scores


def get_next_hint(tracelog):
    pscore = get_pretest_scores(tracelog)
    if pscore['pretest_score'] < 3:
        return 'hint'
    else:
        return 'no_hint'


def get_game_setting(tracelog):
    pscore = get_pretest_scores(tracelog)
    if (pscore['location_pref'] == 'Island'):
        if (pscore['videogame_experience'] > 5):
            return "ci_sunny"
        else:
            return "ci_sunny"
    else:
        return "hospital"


def get_disease_type(tracelog, field="disease"):
    pscore = get_pretest_scores(tracelog)
    result = {}
    if (pscore['pretest_score'] > 5):
        result = {
            "disease":'Influenza',
            "source": "Direct - Droplet",
            "treatment":"Vaccination"
        }
    else:
        result = {
            "disease":'Salmonellosis',
            "source": "Indirect - Food/Water",
            "treatment":"Rest"
        }
    return result[field]

def get_transmission_source(tracelog):
    r =random.random() 
    if  r > 0.67:
        return 'eggs'
    elif 0.67 > r > 0.33:
        return 'lettuce'
    else:
        return 'chicken'

