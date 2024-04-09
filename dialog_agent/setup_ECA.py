import random
import flask
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util 
import dialog_agent.dlg_agent_a4 as da4
import dialog_agent.dlg_agent_a3 as da3
import dialog_agent.dlg_agent_a2 as da2
import dialog_agent.dlg_agent_a1 as da1

def setup_eca_model(dlg_approach):
    BASE_PATH = "./dialog_agent"
    cfg = flask.current_app.config
    eca_data_pd  = pd.read_csv(f'{BASE_PATH}/data/eca_data.csv', header=0)
    eca_data_pd['type'] = eca_data_pd['sentence'].str.extract(r'Type: (\w+),')
    
    checkpoint = f"{BASE_PATH}/model/model_ci_textbite"
    cfg['sbert_model'] = SentenceTransformer('all-MiniLM-L6-v2')
    cfg['tokenizer'] = AutoTokenizer.from_pretrained(checkpoint)
    cfg['device'] = "cuda:0" if torch.cuda.is_available() else "cpu" 
    cfg['model'] = AutoModelForSeq2SeqLM.from_pretrained(checkpoint,local_files_only=True).to(cfg['device'])

    for typ in eca_data_pd.type.unique():
        cfg[f"{typ}_cfg"] = setup_eca_model_given_config(dlg_approach, eca_data_pd[eca_data_pd.type == typ],
                                                         cfg['sbert_model'])

def setup_eca_model_given_config(dlg_approach, knowledge_pd, sbert_model):
    cfg = dict()
    cfg['knowledge_db'] = knowledge_pd.copy().reset_index(drop=True)
    cfg['dlg_approach'] = dlg_approach

    # S-BERT for similarity measure     
    # cfg['sbert_model'] = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Approach 2: Response matching    
    if dlg_approach == 2: 
        # load the pretrained T5-small model 
        # checkpoint = f"{BASE_PATH}/model/model_ci_textbite"
        # cfg['tokenizer'] = AutoTokenizer.from_pretrained(checkpoint)
        # cfg['device'] = "cuda:0" if torch.cuda.is_available() else "cpu" 
        # cfg['model'] = AutoModelForSeq2SeqLM.from_pretrained(checkpoint,local_files_only=True).to(cfg['device'])
        
        # S-BERT similarity for responses
        answers = cfg['knowledge_db'].labels.tolist()
        cfg['answer_em'] = sbert_model.encode(answers, convert_to_tensor=True)        
        
    # Approach 3: Response index prediction
    elif dlg_approach == 3: 
        # load the pretrained T5-small model 
        # checkpoint = f"{BASE_PATH}/model/model_ci_textbite"
        # cfg['tokenizer'] = AutoTokenizer.from_pretrained(checkpoint)
        # cfg['device'] = "cuda:0" if torch.cuda.is_available() else "cpu" 
        # cfg['model'] = AutoModelForSeq2SeqLM.from_pretrained(checkpoint,local_files_only=True).to(cfg['device'])
        
        # S-BERT similarity for questions
        questions = cfg['knowledge_db'].sentence.tolist()
        cfg['question_em'] = sbert_model.encode(questions, convert_to_tensor=True)            
        
    # Approach 4: Question + Response matching    
    elif dlg_approach == 4: 
        # load the pretrained T5-small model 
        # checkpoint = f"{BASE_PATH}/model/model_ci_textbite"
        # cfg['tokenizer'] = AutoTokenizer.from_pretrained(checkpoint)
        # cfg['device'] = "cuda:0" if torch.cuda.is_available() else "cpu" 
        # cfg['model'] = AutoModelForSeq2SeqLM.from_pretrained(checkpoint,local_files_only=True).to(cfg['device'])

        # S-BERT for similarity measure 
        questions = cfg['knowledge_db'].sentence.tolist()    
        cfg['question_em'] = sbert_model.encode(questions, convert_to_tensor=True)        
        answers = cfg['knowledge_db'].labels.tolist()
        cfg['answer_em'] = sbert_model.encode(answers, convert_to_tensor=True)

    else:  
        # S-BERT similarity for qeustions     
        questions = cfg['knowledge_db'].sentence.tolist()
        cfg['question_em'] = sbert_model.encode(questions, convert_to_tensor=True)    

    return cfg


def get_next_utterance(input_utt, eca_type = 'CI', threshold = 0.6):
    gcfg = flask.current_app.config 
    cfg = gcfg[f'{eca_type}_cfg']
    if  ("dlg_approach") in cfg:
        if cfg["dlg_approach"] == 4:
            utt_set = da4.dialog_agent(gcfg['device'],
                            gcfg['tokenizer'],
                            gcfg['model'],
                            gcfg['sbert_model'],
                            cfg['knowledge_db'],
                            cfg['question_em'],
                            cfg['answer_em'],
                            input_utt, 
                            eca_type,
                            simThreshold = threshold)            
        # elif cfg["dlg_approach"] == 1:
        #     utt_set = da1.dialog_agent(cfg['sbert_model'],
        #                          cfg['knowledge_db'],
        #                          cfg['question_em'],
        #                         input_utt,
        #                         simThreshold = 0.6
        #         )
        # elif cfg["dlg_approach"] == 2:
        #     utt_set = da2.dialog_agent(cfg['device'],
        #                                cfg['tokenizer'],
        #                                cfg['model'],
        #                                cfg['sbert_model'],
        #                                cfg['knowledge_db'],
        #                                cfg['answer_em'],
        #                                input_utt, simThreshold = 0.6)
        # elif cfg["dlg_approach"] == 3:                                    
        #     utt_set = da3.dialog_agent(cfg['device'],
        #                                cfg['tokenizer'],
        #                                cfg['model'],
        #                                cfg['sbert_model'],
        #                                cfg['knowledge_db'],
        #                                cfg['question_em'],
        #                                input_utt, simThreshold = 0.6)
        else:
            utt_set = ["Sorry I am busy, please come back later."]
    else:
        utt_set = ["Sorry I am busy, please come back later."]
    
    return utt_set