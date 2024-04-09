from flask import Flask
from flask import request, g
import pprint as pp
from utterance import UTTERANCES, ACTIONS
from dialog_agent.setup_ECA import setup_eca_model, get_next_utterance
from aes_placeholder import get_aes_action
import random
import os
import json
from flask_cors import CORS
import logging
from logging.handlers import RotatingFileHandler



app = Flask(__name__)
CORS(app)

# Create a StreamHandler for logging to stdout
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

# Create a FileHandler for logging to a file
file_handler = RotatingFileHandler('log/flask.log', maxBytes=1024*1024, backupCount=5)
file_handler.setLevel(logging.INFO)

# Create a Formatter for the timestamp and message formatting
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Set the Formatter for the handlers
stream_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add the handlers to the Flask logger
app.logger.addHandler(stream_handler)
app.logger.addHandler(file_handler)


@app.before_first_request
def before_first_request():
    app.logger.info("before_first_request")
    setup_eca_model(4)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/GetUtterance", methods =['OPTIONS'])
def utterance_option_test():
    app.logger.info("In GetUtterance Test options call")
    return 'success'


@app.route("/GetUtterance", methods =['POST', 'PUT'])
def get_utterance():
    json_obj = request.json
    context= [] 
    for u in json_obj['UtteranceHistory'][-5:]:
        # pp.pprint(f"{u['SpeakerName']}: {u['Utterance']}")
        context.append(u['Utterance'])
    app.logger.info(f"ECA:Q: {context[-1]}")    
    chosen_utterance = random.choice(
        get_next_utterance(context[-1]))
    app.logger.info(f"ECA:A: {chosen_utterance}")
    return chosen_utterance



@app.route("/GetUtteranceByType/<ecatype>", methods =['POST', 'PUT'])
def get_utterance_by_type(ecatype):
    json_obj = request.json
    context= [] 
    for u in json_obj['UtteranceHistory'][-5:]:
        # pp.pprint(f"{u['SpeakerName']}: {u['Utterance']}")
        context.append(u['Utterance'])

    json_obj['ECAType'] = ecatype
    app.logger.info(f"ECA:Q: {json_obj}")
    threshold = 0.6
    chosen_utterance = random.choice(
        get_next_utterance(context[-1],
                           eca_type=ecatype,
                           threshold=threshold)
                           )
    app.logger.info(f"ECA:A: {chosen_utterance}")
    return chosen_utterance

@app.route("/GetECAResponse", methods =['POST', 'PUT'])
def get_eca_response():
    json_obj = request.json
    app.logger.info(f"ECA:Q: {json_obj}")
    if 'ConfidenceThreshold' in json_obj:
        threshold = json_obj['ConfidenceThreshold']
    else:
        threshold = 0.6
    chosen_utterance = random.choice(
        get_next_utterance(json_obj['Utterance'],
                           eca_type=json_obj['ECAType'],
                           threshold=threshold)
                           )
    app.logger.info(f"ECA:A: {chosen_utterance}")
    return chosen_utterance



@app.route("/GetChoice", methods =['POST'])
def get_choice():
    json_obj = request.json
    # pp.pprint(json)
    chosen_choice = "|".join(random.sample(UTTERANCES,4))
    app.logger.info(chosen_choice)
    return chosen_choice

@app.route("/GetNextAction/<aestype>", methods =['OPTIONS'])
def option_test(aestype):
    app.logger.info("In GetNextAction Test options call")
    return 'success'

@app.route("/GetNextAction/<aestype>", methods =['POST', 'PUT'])
def get_next_action(aestype):
    json_obj = request.get_json(force=True)
    app.logger.info(f"NP:Q: {aestype}")    
    # with open("trace_log_data/sample.json", "w") as outfile:
    #     outfile.write(json.dumps(json_obj, indent=4))

    chosen_act = get_aes_action(aestype, json_obj)
    if chosen_act == "":
        chosen_act = random.choice(ACTIONS)
    app.logger.info(f"NP:A: {chosen_act}")    
    return chosen_act

