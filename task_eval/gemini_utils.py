import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import random
import os, json
from tqdm import tqdm
import time
from global_methods import run_gemini, get_gemini_client


MAX_LENGTH={'gemini-2.0-flash': 1000000, 'gemini-2.5-flash': 1000000, 'gemini-2.5-pro': 1000000}
PER_QA_TOKEN_BUDGET = 50

QA_PROMPT = """
Based on the above context, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {} Short answer:
"""

QA_PROMPT_CAT_5 = """
Based on the above context, answer the following question.

Question: {} Short answer:
"""

# QA_PROMPT_BATCH = """
# Based on the above conversations, answer the following questions in a few words. Write the answers as a list of strings in the json format. Start and end with a square bracket.

# """

QA_PROMPT_BATCH = """
Based on the above conversations, write short answers for each of the following questions in a few words. Write the answers in the form of a json dictionary where each entry contains the question number as 'key' and the short answer as value. Use single-quote characters for named entities. Answer with exact words from the conversations whenever possible.

"""

# If no information is available to answer the question, write 'No information available'.

CONV_START_PROMPT = "Below is a conversation between two people: {} and {}. The conversation takes place over multiple days and the date of each conversation is wriiten at the beginning of the conversation.\n\n"


def process_ouput(text):

    single_quote_count = text.count("'")
    double_quote_count = text.count('"')
    if single_quote_count > double_quote_count:
        text = text.replace('"', "")
        text = text.replace("'", '"')
    
    answers = json.loads(text)
    if type(answers) == dict:
        for k, v in answers.items():
            if v is None:
                answers[k] = ""
                continue
            # Convert to string if it's not already a string
            if not isinstance(v, str):
                v = str(v)
                answers[k] = v
            if v.startswith('{') and v.endswith('}'):
                try:
                    answers[k] = json.loads(v)['answer']
                except:
                    continue

        return answers
    elif type(answers) == list:
        for k, v in enumerate(answers):
            if v is None:
                answers[k] = ""
                continue
            # Convert to string if it's not already a string
            if not isinstance(v, str):
                v = str(v)
                answers[k] = v
            if v.startswith('{') and v.endswith('}'):
                try:
                    answers[k] = json.loads(v)['answer']
                except:
                    continue
    return answers

def get_cat_5_answer(model_prediction, answer_key):

    model_prediction = model_prediction.strip().lower()
    if len(model_prediction) == 1:
        if 'a' in model_prediction:
            return answer_key['a']
        else:
            return answer_key['b']
    elif len(model_prediction) == 3:
        if '(a)' in model_prediction:
            return answer_key['a']
        else:
            return answer_key['b']
    else:
        return model_prediction

def get_input_context(data, num_question_tokens, client, args):

    query_conv = ''
    min_session = -1
    stop = False
    session_nums = [int(k.split('_')[-1]) for k in data.keys() if 'session' in k and 'date_time' not in k]
    for i in range(min(session_nums), max(session_nums) + 1):
        if 'session_%s' % i in data:
            query_conv += "\n\n"
            for dialog in data['session_%s' % i][::-1]:
                turn = ''
                turn = dialog['speaker'] + ' said, \"' + dialog['text'] + '\"' + '\n'
                if "blip_caption" in dialog:
                    turn += ' and shared %s.' % dialog["blip_caption"]
                turn += '\n'
        
                # num_tokens = client.models.count_tokens(model=args.model, contents='DATE: ' + data['session_%s_date_time' % i] + '\n' + 'CONVERSATION:\n' + turn).total_tokens

                # if (num_tokens + client.models.count_tokens(model=args.model, contents=query_conv).total_tokens + num_question_tokens) < (MAX_LENGTH[args.model]-(PER_QA_TOKEN_BUDGET*(args.batch_size))): # 20 tokens assigned for answers
                #     query_conv = turn + query_conv
                # else:
                #     min_session = i
                #     stop = True
                #     break

                query_conv = turn + query_conv

            query_conv = '\nDATE: ' + data['session_%s_date_time' % i] + '\n' + 'CONVERSATION:\n' + query_conv
        if stop:
            break
        
        # if min_session == -1:
        #     print("Saved %s tokens in query conversation from full conversation" % client.models.count_tokens(model=args.model, contents=query_conv).total_tokens)
        # else:
        #     print("Saved %s conv. tokens + %s question tokens in query from %s out of %s sessions" % (client.models.count_tokens(model=args.model, contents=query_conv).total_tokens, num_question_tokens, max_session-min_session, max_session))

    return query_conv


def get_gemini_answers(client, in_data, out_data, prediction_key, args, memory_system=None):


    assert len(in_data['qa']) == len(out_data['qa']), (len(in_data['qa']), len(out_data['qa']))

    # start instruction prompt
    speakers_names = list(set([d['speaker'] for d in in_data['conversation']['session_1']]))
    start_prompt = CONV_START_PROMPT.format(speakers_names[0], speakers_names[1])
    # start_tokens = client.models.count_tokens(model=args.model, contents=start_prompt).total_tokens
    start_tokens = 100

    if args.rag_mode:
        raise NotImplementedError
    else:
        context_database, query_vectors = None, None

    for batch_start_idx in tqdm(range(0, len(in_data['qa']), args.batch_size), desc='Generating answers'):

        questions = []
        include_idxs = []
        cat_5_idxs = []
        cat_5_answers = []
        for i in range(batch_start_idx, batch_start_idx + args.batch_size):

            if i>=len(in_data['qa']):
                break

            qa = in_data['qa'][i]
            
            if prediction_key not in out_data['qa'][i] or args.overwrite:
                include_idxs.append(i)
            else:
                print("Skipping -->", qa['question'])
                continue

            if qa['category'] == 2:
                questions.append(qa['question'] + ' Use DATE of CONVERSATION to answer with an approximate date.')
            elif qa['category'] == 5:
                # Check for both 'answer' and 'adversarial_answer' keys
                answer_text = None
                if 'answer' in qa:
                    answer_text = qa['answer']
                elif 'adversarial_answer' in qa:
                    answer_text = qa['adversarial_answer']
                else:
                    print(f"Warning: Missing 'answer' or 'adversarial_answer' key in QA item: {qa}")
                    print(f"Available keys: {list(qa.keys())}")
                    # Skip this question if no answer is available
                    continue
                
                question = qa['question'] + " Select the correct answer: (a) {} (b) {}. "
                if random.random() < 0.5:
                    question = question.format('Not mentioned in the conversation', answer_text)
                    answer = {'a': 'Not mentioned in the conversation', 'b': answer_text}
                else:
                    question = question.format(answer_text, 'Not mentioned in the conversation')
                    answer = {'b': 'Not mentioned in the conversation', 'a': answer_text}

                cat_5_idxs.append(len(questions))
                questions.append(question)
                cat_5_answers.append(answer)
                # questions.append(qa['question'] + "Write NOT ANSWERABLE if the question cannot be answered")
            else:
                questions.append(qa['question'])


        if questions == []:
            continue


        context_ids = None
        if args.use_rag:
            
            raise NotImplementedError
        else:
            question_prompt =  QA_PROMPT_BATCH + "\n".join(["%s: %s" % (k, q) for k, q in enumerate(questions)])
            # Try to count tokens, but use default if it fails
            try:
                num_question_tokens = client.models.count_tokens(model=args.model, contents=question_prompt).total_tokens
            except Exception as e:
                print(f"Warning: Failed to count tokens: {e}, using default value")
                num_question_tokens = 200
            query_conv = get_input_context(in_data['conversation'], num_question_tokens + start_tokens, client, args)
            query_conv = start_prompt + query_conv
        

        # print("%s tokens in query" % client.models.count_tokens(model=args.model, contents=query_conv).total_tokens)

        if 'pro' in args.model:
            time.sleep(30)
        else:
            # Add a longer delay for all Gemini calls to reduce connection pressure
            time.sleep(5)

        if args.batch_size == 1:
            # Add memory context if memory system is available
            memory_context = ""
            if memory_system and questions:
                try:
                    memory_context = memory_system.get_relevant_context(questions[0], in_data.get('sample_id', 'unknown'))
                except Exception as e:
                    print(f"Warning: Failed to get memory context: {e}")

            query = query_conv + '\n\n' + memory_context + QA_PROMPT.format(questions[0]) if len(cat_5_idxs) == 0 else query_conv + '\n\n' + memory_context + QA_PROMPT_CAT_5.format(questions[0])

            answer = run_gemini(client, args.model, query)
            
            if answer is None:
                print("Warning: Gemini API returned None response, skipping question")
                continue
            
            if len(cat_5_idxs) > 0:
                answer = get_cat_5_answer(answer, cat_5_answers[0])

            out_data['qa'][include_idxs[0]][prediction_key] = answer.strip()
            if args.use_rag:
                out_data['qa'][include_idxs[0]][prediction_key + '_context'] = context_ids

        else:
            # Add memory context for batch processing if memory system is available
            memory_context = ""
            if memory_system and questions:
                try:
                    # For batch processing, use the first question to get relevant memory context
                    memory_context = memory_system.get_relevant_context(questions[0], in_data.get('sample_id', 'unknown'))
                except Exception as e:
                    print(f"Warning: Failed to get memory context: {e}")
            
            # query = query_conv + '\n' + QA_PROMPT_BATCH + "\n".join(["QUESTION: %s" % q for q in questions])
            query = query_conv + '\n' + memory_context + question_prompt
            # print(query)
            
            trials = 0
            while trials < 5:
                try:
                    trials += 1
                    # print("Trial %s" % trials)
                    # print("Sending query of %s tokens" % client.models.count_tokens(model=args.model, contents=query).total_tokens)
                    # print("Trying with answer token budget = %s per question" % PER_QA_TOKEN_BUDGET)
                    answer = run_gemini(client, args.model, query)
                    
                    if answer is None:
                        print(f"Warning: Gemini API returned None response on trial {trials}, retrying...")
                        continue
                    
                    answer = answer.replace('\\"', "'").replace('json','').replace('`','').strip()

                    # try:
                    #     answers = json.loads(answer.strip())
                    # except:
                    answers = process_ouput(answer.strip())
                    break
                except json.decoder.JSONDecodeError:
                    pass
            
            # Check if all trials failed
            if answer is None:
                print("Error: All Gemini API trials failed, skipping batch")
                continue
            
            for k, idx in enumerate(include_idxs):
                try:
                    answers = process_ouput(answer.strip())
                    # answers = json.loads(answer.strip())
                    # data['qa'][idx]['%s_prediction' % args.model] = answers[k]['answer'].strip()
                    if k in cat_5_idxs:
                        predicted_answer = get_cat_5_answer(answers[str(k)], cat_5_answers[cat_5_idxs.index(k)])
                        out_data['qa'][idx][prediction_key] = predicted_answer
                    else:
                        try:
                            out_data['qa'][idx][prediction_key] = str(answers[str(k)]).replace('(a)', '').replace('(b)', '').strip()
                        except:
                            out_data['qa'][idx][prediction_key] = ', '.join([str(n) for n in list(answers[str(k)].values())])
                except:
                    try:
                        answers = json.loads(answer.strip())
                        if k in cat_5_idxs:
                            predicted_answer = get_cat_5_answer(answers[k], cat_5_answers[cat_5_idxs.index(k)])
                            out_data['qa'][idx][prediction_key] = predicted_answer
                        else:
                            out_data['qa'][idx][prediction_key] = answers[k].replace('(a)', '').replace('(b)', '').strip()
                    except:
                        if k in cat_5_idxs:
                            predicted_answer = get_cat_5_answer(answer.strip(), cat_5_answers[cat_5_idxs.index(k)])
                            out_data['qa'][idx][prediction_key] = predicted_answer
                        else:
                            out_data['qa'][idx][prediction_key] = json.loads(answer.strip().replace('(a)', '').replace('(b)', '').split('\n')[k])[0]

    return out_data

