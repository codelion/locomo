import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os, json
from tqdm import tqdm
import argparse
from global_methods import set_openai_key, set_anthropic_key, set_gemini_key, get_gemini_client
from task_eval.evaluation import eval_question_answering
from task_eval.evaluation_stats import analyze_aggr_acc
from task_eval.gpt_utils import get_gpt_answers
from task_eval.claude_utils import get_claude_answers
from task_eval.gemini_utils import get_gemini_answers
from task_eval.hf_llm_utils import init_hf_model, get_hf_answers
from task_eval.memory_module import BaselineMemory

import numpy as np

# Category mapping for QA evaluation
CATEGORY_MAPPING = {
    1: "Multi-hop",
    2: "Temporal", 
    3: "Open-domain",
    4: "Single-hop",
    5: "Adversarial"
}

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--out-file', required=True, type=str)
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--data-file', type=str, required=True)
    parser.add_argument('--use-rag', action="store_true")
    parser.add_argument('--use-4bit', action="store_true")
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--rag-mode', type=str, default="")
    parser.add_argument('--emb-dir', type=str, default="")
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--retriever', type=str, default="contriever")
    parser.add_argument('--overwrite', action="store_true")
    parser.add_argument('--memory', action="store_true", help="Use memory-enhanced evaluation")
    parser.add_argument('--debug-memory', action="store_true", help="Enable verbose memory debugging")
    parser.add_argument('--max-samples', type=int, help="Maximum samples to evaluate (takes max from each category)")
    args = parser.parse_args()
    return args


def main():

    # get arguments
    args = parse_args()

    print("******************  Evaluating Model %s ***************" % args.model)

    if 'gpt' in args.model:
        # set openai API key
        set_openai_key()

    elif 'claude' in args.model:
        # set openai API key
        set_anthropic_key()

    elif 'gemini' in args.model:
        # Get Gemini client
        gemini_client = get_gemini_client()
        # Map old model names to new ones if needed
        if args.model == "gemini-pro-1.0":
            print("Warning: gemini-pro-1.0 is deprecated. Using gemini-2.5-pro instead.")
            args.model = "gemini-2.5-pro"
    
    elif any([model_name in args.model for model_name in ['gemma', 'llama', 'mistral']]):
        hf_pipeline, hf_model_name = init_hf_model(args)

    else:
        raise NotImplementedError

    # Initialize memory system if requested
    memory_system = None
    if args.memory:
        print("*** Initializing memory-enhanced evaluation ***")
        try:
            # Use the same model and client as the evaluation
            if 'gpt' in args.model:
                import openai
                gpt_client = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
                memory_system = BaselineMemory(gpt_client, args.model, debug=args.debug_memory)
                print(f"Memory system initialized with {args.model}")
            elif 'claude' in args.model:
                import anthropic
                anthropic_key = os.environ.get('ANTHROPIC_API_KEY')
                if not anthropic_key:
                    print("Warning: ANTHROPIC_API_KEY not found. Memory functionality disabled.")
                    args.memory = False
                else:
                    claude_client = anthropic.Client(api_key=anthropic_key)
                    memory_system = BaselineMemory(claude_client, args.model, debug=args.debug_memory)
                    print(f"Memory system initialized with {args.model}")
            elif 'gemini' in args.model:
                memory_system = BaselineMemory(gemini_client, args.model, debug=args.debug_memory)
                print(f"Memory system initialized with {args.model}")
            elif any([model_name in args.model for model_name in ['gemma', 'llama', 'mistral']]):
                # For HuggingFace models, we'll use the pipeline but this is more complex
                # For now, fallback to a simpler approach or skip memory for HF models
                print("Warning: Memory functionality not yet supported for HuggingFace models")
                args.memory = False
            else:
                print(f"Warning: Memory functionality not supported for model: {args.model}")
                args.memory = False
                
        except Exception as e:
            print(f"Warning: Failed to initialize memory system: {e}")
            print("Continuing without memory...")
            args.memory = False

    # load conversations
    samples = json.load(open(args.data_file))
    
    # Apply max-samples filtering if specified
    if args.max_samples:
        print(f"Filtering to max {args.max_samples} samples per category...")
        
        # Group samples by category
        category_samples = {}
        for sample in samples:
            # Look for category in QA items
            for qa_item in sample.get('qa', []):
                if 'category' in qa_item:
                    category = qa_item['category']
                    if category not in category_samples:
                        category_samples[category] = []
                    category_samples[category].append(sample)
                    break  # Only need to categorize sample once
        
        # Take max samples per category
        max_per_category = args.max_samples // len(category_samples) if category_samples else args.max_samples
        filtered_samples = []
        
        for category, cat_samples in category_samples.items():
            category_name = CATEGORY_MAPPING.get(category, f"Category-{category}")
            selected = cat_samples[:max_per_category]
            filtered_samples.extend(selected)
            print(f"  {category_name}: {len(selected)}/{len(cat_samples)} samples")
        
        samples = filtered_samples
        print(f"Total samples after filtering: {len(samples)}")
    
    prediction_key = "%s_prediction" % args.model if not args.use_rag else "%s_%s_top_%s_prediction" % (args.model, args.rag_mode, args.top_k)
    model_key = "%s" % args.model if not args.use_rag else "%s_%s_top_%s" % (args.model, args.rag_mode, args.top_k)
    # load the output file if it exists to check for overwriting
    if os.path.exists(args.out_file):
        out_samples = {d['sample_id']: d for d in json.load(open(args.out_file))}
    else:
        out_samples = {}

    for data in samples:

        out_data = {'sample_id': data['sample_id']}
        if data['sample_id'] in out_samples:
            out_data['qa'] = out_samples[data['sample_id']]['qa'].copy()
        else:
            out_data['qa'] = data['qa'].copy()

        # Add conversation to memory if memory system is enabled
        if memory_system and 'conversation' in data:
            try:
                memory_system.add_conversation(data['conversation'], data['sample_id'])
                print(f"Added conversation {data['sample_id']} to memory")
            except Exception as e:
                print(f"Warning: Failed to add conversation to memory: {e}")

        if 'gpt' in args.model:
            # get answers for each sample
            answers = get_gpt_answers(data, out_data, prediction_key, args, memory_system)
        elif 'claude' in args.model:
            answers = get_claude_answers(data, out_data, prediction_key, args, memory_system)
        elif 'gemini' in args.model:
            answers = get_gemini_answers(gemini_client, data, out_data, prediction_key, args, memory_system)
        elif any([model_name in args.model for model_name in ['gemma', 'llama', 'mistral']]):
            answers = get_hf_answers(data, out_data, args, hf_pipeline, hf_model_name, memory_system)
        else:
            raise NotImplementedError

        # Ensure all questions have prediction keys (add default for skipped questions)
        for i, qa_item in enumerate(answers['qa']):
            if prediction_key not in qa_item:
                print(f"Warning: Question {i} missing prediction key, adding default empty prediction")
                qa_item[prediction_key] = ""  # Empty string for failed predictions
        
        # evaluate individual QA samples and save the score
        exact_matches, lengths, recall = eval_question_answering(answers['qa'], prediction_key)
        for i in range(0, len(answers['qa'])):
            answers['qa'][i][model_key + '_f1'] = round(exact_matches[i], 3)
            if args.use_rag and len(recall) > 0:
                answers['qa'][i][model_key + '_recall'] = round(recall[i], 3)
            
            # Add category name to output
            category_num = answers['qa'][i].get('category', 0)
            answers['qa'][i]['category_name'] = CATEGORY_MAPPING.get(category_num, f"Unknown-{category_num}")

        out_samples[data['sample_id']] = answers

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.out_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    with open(args.out_file, 'w') as f:
        json.dump(list(out_samples.values()), f, indent=2)

    
    analyze_aggr_acc(args.data_file, args.out_file, args.out_file.replace('.json', '_stats.json'),
                model_key, model_key + '_f1', rag=args.use_rag)
    # encoder=tiktoken.encoding_for_model(args.model))


main()

