"""
    ECA - Dialog Agent
    Date: 12/15/2022
    Author: Yeo Jin Kim (NC State University)
"""
import torch
import time
#import numpy as np
from sentence_transformers import util 
import flask

#device, tokenizer, model, sbert_model, knowledge_db, question_em, input_utt, eca_type, simThreshold = 0.6
def dialog_agent(device, tokenizer, model, sbert_model, knowledge_db, question_em, answers_em,
                  input_utt, eca_type='CI', simThreshold = 0.6): 
    startTime = time.process_time()
    
    # Add the prompt
    #input_utt_prompt = "Generate: " + input_utt
    input_utt_prompt = f"Type: {eca_type}, Response: {input_utt}"
    
    # Generate response from the model
    input_utt_ids =  tokenizer(input_utt_prompt, return_tensors="pt").to(device).input_ids
    response_ids = model.generate(input_utt_ids,  max_length = 256) #num_beams = 2, num_return_sequences = num_sents
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)

    # Approach 1: Find the most similar question from the knowledge DB, using S-BERT & cosine similarity            
    input_em = sbert_model.encode([input_utt], convert_to_tensor=True)    
    question_scores = util.cos_sim(question_em, input_em)

    # Approach 2: Find the most similar response from the knowledge DB, using S-BERT & cosine similarity
    response_em = sbert_model.encode([response], convert_to_tensor=True)    
    response_scores = util.cos_sim(answers_em, response_em)

    # Combined
    combined_scores = (question_scores + response_scores)/2
    idx =  torch.argmax(combined_scores).item()
    final_score = combined_scores[idx].item()    
    
    # Get the most similar response
    candidate_response = knowledge_db.loc[idx, 'labels']            
    
    if final_score >= simThreshold:
        response = candidate_response
    elif final_score >= simThreshold - 0.2:
            response = "I'm not sure, but this might be helpful. "+ candidate_response                            
    else:
        response = "Sorry, I don\'t know." 

    if True: # args.explain:
        responseTime = time.process_time() - startTime
        flask.current_app.logger.info("ECA:A:  (Found: {} -> {} - similarity score: {:.3f} (Q: {:.3f}, R: {:.3f}), responseTime: {:.3f})".format(
            candidate_response, response, final_score, question_scores[idx].item(), response_scores[idx].item(), responseTime))

    return [response]
