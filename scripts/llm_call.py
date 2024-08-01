import json
import os
from openai import OpenAI
import requests
import urllib3
import os

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def get_response_from_chatgpt_openai(prompt: str, model_name: str, temperature: str):
    # Set up the OpenAI API client
    client = OpenAI(
        api_key = ## replace with your own api key ##,
    )

    temperature_float = float(temperature)
    # Generate a response
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature_float
    )

    response = completion.choices[0].message.content.strip()

    return response


def get_response_from_qwen(prompt: str):
    url = ## replace with your own qwen service url ##
    headers = {'Content-Type': 'application/json'}
    data = {"prompt": prompt}

    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        return response.json()['response']
    else:
        return {"error": "Failed to generate response", "status_code": response.status_code}


def get_response_from_ernie(prompt: str, temperature: str):
    access_token = ## replace with your own access token ##
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-3.5-8k-0329?access_token=" + access_token

    if temperature == '0':
        temperature = float(0.01)
    else:
        temperature = float(temperature)
    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": temperature
    })
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    if response.status_code == 200:
        return response.json().get("result")
    else:
        return {"error": "Failed to generate response", "status_code": response.status_code}


def get_response_from_llm(type: str, prompt: str, model_name: str, temperature: str):
    if type in ['gpt3.5', 'gpt4']:
        response = get_response_from_chatgpt_openai(prompt, model_name, temperature)
    elif type == 'qwen':
        response = get_response_from_qwen(prompt)
    elif type == 'ernie':
        response = get_response_from_ernie(prompt, temperature)
    else:
        raise RuntimeError('LLM name error: ' + type)

    return response
