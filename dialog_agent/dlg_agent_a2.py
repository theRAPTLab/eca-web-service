"""
    ECA - Dialog Agent
    Date: 12/15/2022
    Author: Yeo Jin Kim (NC State University)
"""
import torch
import time
from sentence_transformers import util 
    
def dialog_agent(device, tokenizer, model, sbert_model, knowledge_db, answers_em, input_utt, simThreshold = 0.6): 
    startTime = time.process_time()

    # Add the prompt
    input_utt_prompt = "Generate: " + input_utt
    
    # Generate response from the model
    input_utt_ids =  tokenizer(input_utt_prompt, return_tensors="pt").to(device).input_ids
    response_ids = model.generate(input_utt_ids,  max_length = 256) #num_beams = 2, num_return_sequences = num_sents
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)

    # Find the most similar response from the knowledge DB, using S-BERT & cosine similarity
    response_em = sbert_model.encode([response], convert_to_tensor=True)    
    cosine_scores = util.cos_sim(answers_em, response_em)
    idx =  torch.argmax(cosine_scores).item()
    score = cosine_scores[idx].item()
    candidate_response = knowledge_db.loc[idx, 'labels']            

    if score >= simThreshold:
        response = candidate_response
    elif score >= simThreshold - 0.2:
        response = "I'm not sure, but this might be helpful. "+ candidate_response                    
    else:
        response = "Sorry, I don\'t know." 

    if True: # args.explain:
        responseTime = time.process_time() - startTime
        print("   (Found-{:.3f}: {} - responseTime: {:.3f})".format(score, candidate_response, responseTime))

    return [response]
