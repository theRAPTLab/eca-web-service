# from flask import Flask
# from flask import request
# import pprint as pp
# from utterance import UTTERANCES, ACTIONS
#from utterance_GPT_gen import get_next_utterance
#from aes_placeholder import get_aes_action
import random

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util 

import dlg_agent_a1 as da1
import dlg_agent_a2 as da2
import dlg_agent_a3 as da3 
import dlg_agent_a4 as da4

#app = Flask(__name__)
dlg_approach = 4

knowledge_db = pd.read_csv('data/ci_qna_textbite.csv', header=0)
# S-BERT for similarity measure     
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Approach 1: Question matching 
if dlg_approach == 1:      
    questions = knowledge_db.sentence.tolist()
    questions_em = sbert_model.encode(questions, convert_to_tensor=True)    

# Approach 2: Response matching    
elif dlg_approach == 2: 
    # load the pretrained T5-small model 
    checkpoint = "model/model_ci_textbite"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    device = "cuda:0" if torch.cuda.is_available() else "cpu" 
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint,local_files_only=True).to(device)
    
    # S-BERT for similarity measure 
    answers = knowledge_db.labels.tolist()
    answers_em = sbert_model.encode(answers, convert_to_tensor=True)
    
# Approach 3: Response index prediction
elif dlg_approach == 3:       
    # load the pretrained T5-small model 
    checkpoint = "model/model_ci_textbite"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    device = "cuda:0" if torch.cuda.is_available() else "cpu" 
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint,local_files_only=True).to(device)
    
    # S-BERT for similarity measure 
    questions = knowledge_db.sentence.tolist()
    questions_em = sbert_model.encode(questions, convert_to_tensor=True)    

# Approach 4: Question + Response matching    
elif dlg_approach == 4: 
    # load the pretrained T5-small model 
    checkpoint = "model/model_ci_textbite"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    device = "cuda:0" if torch.cuda.is_available() else "cpu" 
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint,local_files_only=True).to(device)
    
    # S-BERT for similarity measure 
    questions = knowledge_db.sentence.tolist()    
    questions_em = sbert_model.encode(questions, convert_to_tensor=True)        
    answers = knowledge_db.labels.tolist()
    answers_em = sbert_model.encode(answers, convert_to_tensor=True)

    
#@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

#@app.route("/GetUtterance", methods =['POST'])
def get_utterance():
    #json = request.json
    context= [] 
    # for u in json['UtteranceHistory'][-5:]:
    #     pp.pprint(f"{u['SpeakerName']}: {u['Utterance']}")
    #     context.append(u['Utterance'])
    input_utt = input("User : ")  # u['Utterance']  
    chosen_utterance = random.choice(
        # approach 1
        #da1.dialog_agent(sbert_model, knowledge_db, questions_em, input_utt, simThreshold = 0.6)      
        # approach 2
        #da2.dialog_agent(device, tokenizer, model, sbert_model, knowledge_db, answers_em, input_utt, simThreshold = 0.6)
        # approach 3
        #da3.dialog_agent(device, tokenizer, model, sbert_model, knowledge_db, questions_em, input_utt, simThreshold = 0.6)
        # approach 4
        da4.dialog_agent(device, tokenizer, model, sbert_model, knowledge_db, questions_em, answers_em, input_utt, simThreshold = 0.6)
    )
    print("CHOSEN : ", chosen_utterance)
    return chosen_utterance

get_utterance()

#@app.route("/GetChoice", methods =['POST'])
def get_choice():
    json = request.json
    # pp.pprint(json)
    chosen_choice = "|".join(random.sample(UTTERANCES,4))
    print(chosen_choice)
    return chosen_choice

#@app.route("/GetNextAction/<aestype>", methods =['POST'])
def get_next_action(aestype):
    json = request.json
    # pp.pprint(json)
    pp.pprint(f"AES TYPE: {aestype}")
    chosen_act = get_aes_action(aestype, json)
    if chosen_act == "":
        chosen_act = random.choice(ACTIONS)
    print(chosen_act)
    return chosen_act
