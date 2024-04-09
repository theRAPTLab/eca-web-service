"""
    ECA - Dialog Agent
    Date: 11/17/2022
    Author: Yeo Jin Kim (NC State University)
"""
import torch
import time
#import numpy as np
from sentence_transformers import util 

def dialog_agent(sbert_model, knowledge_db, question_em, input_utt, simThreshold = 0.6):         
    startTime = time.process_time()
    
    # Find the most similar question from the knowledge DB, using S-BERT & cosine similarity            
    input_em = sbert_model.encode([input_utt], convert_to_tensor=True)    
    cosine_scores = util.cos_sim(question_em, input_em)
    idx =  torch.argmax(cosine_scores).item()
    score = cosine_scores[idx].item()
    candidate_question = knowledge_db.loc[idx, 'sentence']
    candidate_response = knowledge_db.loc[idx, 'labels']
    
    if score >= simThreshold:
        response = candidate_response
    elif score >= simThreshold - 0.2:
        response = "I'm not sure, but this might be helpful. "+ candidate_response
    else:
        response = "Sorry, I don\'t know."

    if True: # args.explain:
        responseTime = time.process_time() - startTime
        print("   (Found-{:.3f}: {} - reponseTime: {:.3f})".format(score, candidate_question, responseTime))

    return [response]
        

