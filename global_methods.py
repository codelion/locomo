import openai
import numpy as np
import json
import time
import sys
import os

from google import genai
from anthropic import Anthropic


def get_openai_embedding(texts, model="text-embedding-ada-002"):
   texts = [text.replace("\n", " ") for text in texts]
   return np.array([openai.Embedding.create(input = texts, model=model)['data'][i]['embedding'] for i in range(len(texts))])

def set_anthropic_key():
    pass

def set_gemini_key():
    # This is no longer needed with the new SDK
    # The client will automatically use the GEMINI_API_KEY or GOOGLE_API_KEY environment variable
    pass

def get_gemini_client():
    # Get API key from environment variables
    api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("Please set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")
    
    # Create and return the client with timeout configuration
    try:
        # Try to set timeout if supported
        return genai.Client(api_key=api_key, timeout=60)
    except TypeError:
        # Fallback if timeout parameter is not supported
        return genai.Client(api_key=api_key)

def set_openai_key():
    openai.api_key = os.environ['OPENAI_API_KEY']


def run_json_trials(query, num_gen=1, num_tokens_request=1000, 
                model='davinci', use_16k=False, temperature=1.0, wait_time=1, examples=None, input=None):

    run_loop = True
    counter = 0
    while run_loop:
        try:
            if examples is not None and input is not None:
                output = run_chatgpt_with_examples(query, examples, input, num_gen=num_gen, wait_time=wait_time,
                                                   num_tokens_request=num_tokens_request, use_16k=use_16k, temperature=temperature).strip()
            else:
                output = run_chatgpt(query, num_gen=num_gen, wait_time=wait_time, model=model,
                                                   num_tokens_request=num_tokens_request, use_16k=use_16k, temperature=temperature)
            output = output.replace('json', '') # this frequently happens
            facts = json.loads(output.strip())
            run_loop = False
        except json.decoder.JSONDecodeError:
            counter += 1
            time.sleep(1)
            print("Retrying to avoid JsonDecodeError, trial %s ..." % counter)
            print(output)
            if counter == 10:
                print("Exiting after 10 trials")
                sys.exit()
            continue
    return facts


def run_claude(query, max_new_tokens, model_name):

    if model_name == 'claude-sonnet':
        model_name = "claude-3-sonnet-20240229"
    elif model_name == 'claude-haiku':
        model_name = "claude-3-haiku-20240307"

    client = Anthropic(
    # This is the default and can be omitted
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )
    # print(query)
    message = client.messages.create(
        max_tokens=max_new_tokens,
        messages=[
            {
                "role": "user",
                "content": query,
            }
        ],
        model=model_name,
    )
    print(message.content)
    return message.content[0].text


def run_gemini(client, model_name: str, content: str, max_tokens: int = 0):
    import time
    import threading
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
    
    max_retries = 3
    retry_delay = 5  # Start with longer initial delay
    request_timeout = 60  # 60 second timeout per request
    
    def make_api_call():
        return client.models.generate_content(
            model=model_name,
            contents=content
        )
    
    for attempt in range(max_retries):
        try:
            # Execute API call with timeout using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(make_api_call)
                try:
                    response = future.result(timeout=request_timeout)
                    return response.text
                except FutureTimeoutError:
                    raise TimeoutError(f"Gemini API request timed out after {request_timeout} seconds")
                    
        except (ConnectionError, ConnectionResetError) as e:
            if attempt < max_retries - 1:
                print(f'Connection error on attempt {attempt + 1}: {e}, retrying in {retry_delay}s...')
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff: 5s, 10s, 20s
                continue
            else:
                print(f'Connection error: Failed after {max_retries} attempts: {e}')
                return None
        except TimeoutError as e:
            if attempt < max_retries - 1:
                print(f'Timeout on attempt {attempt + 1}: {e}, retrying in {retry_delay}s...')
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            else:
                print(f'Timeout: Failed after {max_retries} attempts: {e}')
                return None
        except Exception as e:
            print(f'{type(e).__name__}: {e}')
            return None


def run_chatgpt(query, num_gen=1, num_tokens_request=1000, 
                model='chatgpt', use_16k=False, temperature=1.0, wait_time=1):

    completion = None
    while completion is None:
        wait_time = wait_time * 2
        try:
            # if model == 'davinci':
            #     completion = openai.Completion.create(
            #                     # model = "gpt-3.5-turbo",
            #                     model = "text-davinci-003",
            #                     temperature = temperature,
            #                     max_tokens = num_tokens_request,
            #                     n=num_gen,
            #                     prompt=query
            #                 )
            if model == 'chatgpt':
                messages = [
                        {"role": "system", "content": query}
                    ]
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    temperature = temperature,
                    max_tokens = num_tokens_request,
                    n=num_gen,
                    messages = messages
                )
            elif 'gpt-4' in model:
                completion = openai.ChatCompletion.create(
                    model=model,
                    temperature = temperature,
                    max_tokens = num_tokens_request,
                    n=num_gen,
                    messages = [
                        {"role": "user", "content": query}
                    ]
                )
            else:
                print("Did not find model %s" % model)
                raise ValueError
        except openai.error.APIError as e:
            #Handle API error here, e.g. retry or log
            print(f"OpenAI API returned an API Error: {e}; waiting for {wait_time} seconds")
            time.sleep(wait_time)
            pass
        except openai.error.APIConnectionError as e:
            #Handle connection error here
            print(f"Failed to connect to OpenAI API: {e}; waiting for {wait_time} seconds")
            time.sleep(wait_time)
            pass
        except openai.error.RateLimitError as e:
            #Handle rate limit error (we recommend using exponential backoff)
            print(f"OpenAI API request exceeded rate limit: {e}")
            pass
        except openai.error.ServiceUnavailableError as e:
            #Handle rate limit error (we recommend using exponential backoff)
            print(f"OpenAI API request exceeded rate limit: {e}; waiting for {wait_time} seconds")
            time.sleep(wait_time)
            pass
        # except Exception as e:
        #     if e:
        #         print(e)
        #         print(f"Timeout error, retrying after waiting for {wait_time} seconds")
        #         time.sleep(wait_time)
    

    if model == 'davinci':
        outputs = [choice.get('text').strip() for choice in completion.get('choices')]
        if num_gen > 1:
            return outputs
        else:
            # print(outputs[0])
            return outputs[0]
    else:
        # print(completion.choices[0].message.content)
        return completion.choices[0].message.content
    

def run_chatgpt_with_examples(query, examples, input, num_gen=1, num_tokens_request=1000, use_16k=False, wait_time = 1, temperature=1.0):

    completion = None
    
    messages = [
        {"role": "system", "content": query}
    ]
    for inp, out in examples:
        messages.append(
            {"role": "user", "content": inp}
        )
        messages.append(
            {"role": "system", "content": out}
        )
    messages.append(
        {"role": "user", "content": input}
    )   
    
    while completion is None:
        wait_time = wait_time * 2
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo" if not use_16k else "gpt-3.5-turbo-16k",
                temperature = temperature,
                max_tokens = num_tokens_request,
                n=num_gen,
                messages = messages
            )
        except openai.error.APIError as e:
            #Handle API error here, e.g. retry or log
            print(f"OpenAI API returned an API Error: {e}; waiting for {wait_time} seconds")
            time.sleep(wait_time)
            pass
        except openai.error.APIConnectionError as e:
            #Handle connection error here
            print(f"Failed to connect to OpenAI API: {e}; waiting for {wait_time} seconds")
            time.sleep(wait_time)
            pass
        except openai.error.RateLimitError as e:
            #Handle rate limit error (we recommend using exponential backoff)
            print(f"OpenAI API request exceeded rate limit: {e}")
            pass
        except openai.error.ServiceUnavailableError as e:
            #Handle rate limit error (we recommend using exponential backoff)
            print(f"OpenAI API request exceeded rate limit: {e}; waiting for {wait_time} seconds")
            time.sleep(wait_time)
            pass
    
    return completion.choices[0].message.content
