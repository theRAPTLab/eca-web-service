"""
    ECA - Dialog Agent - Approach 3 (LLM-based Response Index Prediction)
    Date: 12/15/2022
    Author: Yeo Jin Kim (NC State University)
    
    Function: 
     - Predicts the index of response in the give knoweldge DB. 
     - Check the SBERT-cosine similarity between the matched question corresponding to the predicted index.
    
"""

import time
from sentence_transformers import util 


def dialog_agent(device, tokenizer, model, sbert_model, knowledge_db, question_em, input_utt, simThreshold = 0.6):     
    
    startTime = time.process_time()

    # Add the prompt
    input_utt_prompt = "Index: " + input_utt
    
    # Predict the index of response in knoweldge_db
    input_response_ids =  tokenizer(input_utt_prompt, return_tensors="pt").to(device).input_ids
    output_response = model.generate(input_response_ids, max_length = 256)#, num_beams = 2, num_return_sequences = 2)
    response = tokenizer.decode(output_response[0], skip_special_tokens=True)
    #print(response)
    
    if response[0] == 'i': # when an index is predicted
        idx =  int(response[1:]) # Extract the predicted response index (only number)
        print("predicted idx: {}".format(idx))
        candidate_response = knowledge_db.loc[idx, 'labels']

        # Get the S-BERT embedding cosine simiarlity between the predected index's question and the input question. 
        input_em = sbert_model.encode([input_utt], convert_to_tensor=True)    
        cosine_score = util.cos_sim(question_em[idx], input_em).item()
        predicted_question = knowledge_db.loc[idx, 'sentence']
        
        if (idx in range(len(knowledge_db))) and (cosine_score >= simThreshold):
            response = candidate_response
        elif (idx in range(len(knowledge_db))) and cosine_score >= simThreshold - 0.2:
            response = "I'm not sure, but this might be helpful. "+ candidate_response
        else:
            response = "Sorry, I don\'t know."

        if True: #args.explain:
            responseTime = time.process_time() - startTime
            print("   (Question-{} -> {} - similarity score: {:.3f})".format(input_utt, predicted_question, cosine_score))
            print("   (Response-{} -> {} - response time: {:.3f} sec)".format(candidate_response, response, responseTime))
            
    else: # when an index is NOT predicted
        response = "Sorry, I have no idea."
        
    return [response]

