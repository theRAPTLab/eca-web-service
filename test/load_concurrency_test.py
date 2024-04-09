import asyncio
import json
import aiohttp
import time
from tqdm import tqdm

async def post_request(session, json_data):
    url = "https://tracedata-01.csc.ncsu.edu:5000/GetUtterance"
    headers = {'Content-Type': 'application/json'}
    async with session.post(url, json=json_data, headers=headers) as response:
        status_code = response.status
        response_data = await response.text()
        if status_code != 200:
            print(f"Status Code: {status_code}, Response Data: {response_data}")

async def load_test(concurrency, num_requests, json_data):
    for _ in tqdm(range(num_requests)):
        tasks = []
        async with aiohttp.ClientSession() as session:
            for _ in range(concurrency):
                task = asyncio.ensure_future(post_request(session, json_data))
                tasks.append(task)
            
            await asyncio.gather(*tasks)        

if __name__ == "__main__":
    json_data = {
        "LogHistory": [],
        "UtteranceHistory": [
            {
                "SpeakerName": "Player",
                "Utterance": "What causes stomach ache"
            }
        ]
    }

    concurrency = 500  # This controls how many requests will actually be concurrently executed
    num_requests = 10  # Total number of requests
    
    start_time = time.time()
    asyncio.run(load_test(concurrency, num_requests, json_data))
    end_time = time.time()
    
    print(f"Total time taken: {end_time - start_time} seconds")


