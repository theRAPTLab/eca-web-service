import requests
import json
import pandas as pd
import re
from tqdm import tqdm

from requests.packages.urllib3.exceptions import InsecureRequestWarning


def eca_response(utterance, ecatype, confidence_threshold):
    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
    payload = {
        "Utterance": utterance,
        "ECAType": ecatype,
        "ConfidenceThreshold": confidence_threshold
    }
    headers = {
        "Content-Type": "application/json"
    }

    # To ignore SSL certificate warnings, you can use verify=False (not recommended for production)
    # url = "https://127.0.0.1:5000/GetECAResponse"
    # response = requests.post(url, data=json.dumps(payload), headers=headers, verify=False)


    url = "https://tracedata-01.csc.ncsu.edu:5000/GetECAResponse"
    response = requests.post(url, data=json.dumps(payload), headers=headers)

    if response.status_code == 200:
        return response.text
    else:
        return f"An error occurred: {response.text}"

# write a test using the above function to get a response from the ECA
# loop through the data/eca_data.csv csv file using pandas and get a response for each sentence
# from the sentence column which has the format "Type: <eca_type>, response: <utterance>"
# get the eca_type and utterance and pass them to the eca_response function
# check that the response contains the string in the labels column
def test_eca_response():
    # Read the CSV file into a DataFrame
    edf = pd.read_csv("../dialog_agent/data/eca_data.csv")
    edf['type'] = edf['sentence'].str.extract(r'Type: (\w+),')
    for typ in edf.type.unique():
        cnt = min(500, len(edf[edf.type == typ]))
        print(f"Type: {typ} ({cnt})")        
        df = edf[edf.type == typ].copy().sample(cnt).reset_index()
        success = []
        for index, row in tqdm(df.iterrows()):
            sentence = row['sentence']
            label = row['labels']

            # Use regular expressions to parse the sentence
            match = re.search(r'Type: (\w+), response: (.+)', sentence)
            if not match:
                #print(f"Skipping row {index}: Invalid format")
                continue

                
            eca_type = match.group(1)
            utterance = match.group(2)

            # Call your function
            response = eca_response(utterance, eca_type, 0.4)

            # Check that the response contains the string in the labels column
            success.append(int(label in response))

        print(f"Type: {typ}\tSuccess rate: {sum(success) / len(success)} out of {len(success)}")

test_eca_response()    
