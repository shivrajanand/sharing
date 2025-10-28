import json
import os
import torch
import pandas as pd
from datasets import Dataset
import csv
from transformers import BitsAndBytesConfig
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import datetime
import itertools
import sys
import re
from Eval_USS_LSS import unlabeled_metric as calc_uss, metric as calc_lss
from eval_F1 import metric as eval_f1_metric
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    PeftConfig
)




# Configuration
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"  # Use appropriate size based on your hardware
OUTPUT_DIR = "./llama-3.2-3B-finetuned-sa-hi"
TRAIN_PATH = "./Datasets/No Context CSV files/train.csv"
DEV_PATH = "./Datasets/No Context CSV files/dev.csv"
TEST_PATH = "./Datasets/No Context CSV files/test.csv"  # Path to test file for evaluation
MAX_LENGTH = 512
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-5
NUM_EPOCHS = 10
WARMUP_STEPS = 100
SAVE_STEPS = 100
SHOT = 5

# LoRA Configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]




# Check if CUDA is available and determine precision
device = "cuda" if torch.cuda.is_available() else "cpu"
# Choose one precision mode only - prioritize BF16 if available, then FP16, then FP32
use_bf16 = False
use_fp16 = False
if torch.cuda.is_available():
    if torch.cuda.is_bf16_supported():
        use_bf16 = True
        precision = torch.bfloat16
        print(f"Using device: {device} with BF16 precision")
    elif torch.cuda.get_device_capability()[0] >= 7:
        use_fp16 = True
        precision = torch.float16
        print(f"Using device: {device} with FP16 precision")
    else:
        precision = torch.float32
        print(f"Using device: {device} with FP32 precision")
else:
    precision = torch.float32
    print(f"Using device: {device} with FP32 precision")




# Import functions from evaluation files
def lines_to_relations(true_lines):
    relations_list = []
    relations_oneline = []
    for line in true_lines:
        if line == '\n':
            relations_oneline = [relation for relation in relations_oneline if (relation[2]!= 'No_rel' and relation[2]!= 'root')]
            relations_list.append(relations_oneline)
            relations_oneline = []
        else:
            lst = line.strip().split('\t')
            relation = [lst[0],lst[4],lst[5]]
            relations_oneline.append(relation)
    
    return relations_list

def comps_from_relations(relations):
    lst = []
    nested_comp = []
    for rel in relations:
        if 'Comp_root' in rel:
            lst.append(rel)
            nested_comp.append(lst)
            lst = []
        else:
            lst.append(rel)
    return nested_comp

def unlabeled_metric(true_lines, pred_lines):
    correct = 0
    true_count = 0
    pred_count = 0
    for i in range(len(true_lines)):
        if true_lines[i]!= '\n': 
            true_lst = true_lines[i].split('\t')
            pred_lst = pred_lines[i].split('\t')
            if true_lst[5]!='No_rel':
                true_span = [true_lst[0],true_lst[4]]
                pred_span = [pred_lst[0],pred_lst[4]]
                if true_span==pred_span:
                    correct += 1
                true_count += 1
                pred_count += 1
    p = correct / pred_count if pred_count != 0 else 0
    r = correct / true_count if true_count != 0 else 0
    f1 = 2 * p * r / (p + r) if p != 0 and r != 0 else 0
    
    return round(100*f1, 2)

def metric(true_lines, pred_lines):
    true_relations = lines_to_relations(true_lines)
    pred_relations = lines_to_relations(pred_lines)
    correct = 0
    predict_count = 0
    true_count = 0
    match = []
    true_labels = []
    em = 0
    tot_comps = 0
    for i in range(len(pred_relations)):
        
        true_relation_oneline = true_relations[i]
        pred_relation_oneline = pred_relations[i]
        tr_copy = [','.join(lst) for lst in true_relation_oneline]
        pr_copy = [','.join(lst) for lst in pred_relation_oneline]
        
        tr_comps = comps_from_relations(tr_copy)
        pr_comps = comps_from_relations(pr_copy)
        for comp in pr_comps:
            if comp in tr_comps:
                em += 1
        tot_comps += len(tr_comps)
        
        for rel in pred_relation_oneline:
            if rel in true_relation_oneline:
                correct += 1
        predict_count += len(pred_relation_oneline)
        true_count += len(true_relation_oneline)

    if correct == 0:
        p = 0
        r = 0
    else:
        p = correct / predict_count
        r = correct / true_count
    if p == 0 or r == 0:
        f1 = 0
    else:
        f1 = 2 * p * r / (p + r)
    a = 1.0*correct/(predict_count+true_count-correct)
    em_per = em/tot_comps if tot_comps > 0 else 0
    metrics_list = [100*p, 100*r, 100*f1, 100*a, 100*em_per]
    metrics_list = [round(i, 2) for i in metrics_list]

    return metrics_list

def raw_to_clean(line):
    line = re.sub('( {1,})',' ',line)
    line = re.sub('-$','',line) #to remove the - in the end
    line = re.sub('<','',line)
    line = re.sub('-',' ',line)
    line = re.sub('(>\w+)','',line)
    line = re.sub(' $','',line) #to remove the space in the end
    line = re.sub('^ ','',line) #to remove the space in the beginning
    line = re.sub('( {1,})',' ',line)
    return line

def Conversion(infile, outfile):
    import re
    df = pd.read_csv(infile)
    with open(infile) as f:
        raw_lines = f.readlines()
        lines = [raw_to_clean(line) for line in raw_lines]
    with open(outfile, 'w') as w:
        for k in range(df.shape[0]):
            raw_line = raw_lines[k].strip()
            line = lines[k]
            comps_and_words = raw_line.strip().split() #space separated raw line gives list of compounds and individual words
            clean_tokens = line.strip().split() #space separated clean line gives clean tokens
            outmost_comp_list = [token for token in comps_and_words if '<' in token] #identifying compounds from comps_and_words
            c = 0
            d = 0
            i = 0
            while i < len(clean_tokens):   ###traversing through sentence tokens
                if c < len(outmost_comp_list):
                    outcomp = outmost_comp_list[c]
                clean_outcomp = raw_to_clean(outcomp) ### clean tokens of the outermost/parent compound
                token = comps_and_words[d] ### choosing token i.e., compound and word from comps_and_words list
                word = clean_tokens[i]
                if word == token: ###for individual words other than compounds
                    w.write(f'{i+1}\t{word}\tCompNo\t_\t{len(clean_tokens)+1}\tNo_rel'+'\n')
                    i += 1
                    d += 1
                else:
                    rem_string = outcomp
                    comp_len = len(clean_outcomp.split())
                    for p in range(comp_len):
                        subword = clean_outcomp.split()[p] ### getting p-th subword of the compound
                        if p==comp_len-1:   ###for last subword of an out-most compound
                            w.write(f'{i+1}\t{subword}\tComp{comp_len}\t_\t{len(clean_tokens)+1}\tComp_root'+'\n')
                            i += 1
                        else:    ### for remaining subwords in compound
                            subword_end = re.search('[^>]'+subword,rem_string).end() #getting end of the subword
                            rem_string = rem_string[subword_end:] ### remaining string after the subword/token
                            n = 0
                            ind = 0
                            p = 0
                            if rem_string[0] == '>': ###if remaining string starts with tag >\w+
                                flag = 1
                            else:                   
                                flag = 0
                            for ind in range(len(rem_string)):
                                if rem_string[ind] == '<': ###adding 1 to n if encountered an open bracket
                                    n -= 1
                                elif rem_string[ind] == '>': ###subtracting 1 from n if encountered an open bracket
                                    n += 1
                                elif rem_string[ind] == '-': ###adding 1 if encountered hyphen
                                    p += 1
                                if n-flag == 0: ### setting p to 0 if a compound is completed i.e., n-flag==0
                                    p = 0
                                if rem_string[0] == '>': #case1: remaining string starts with tag => >\w+
                                    #subcase1: remaining string has no new compound/ has an immediate relation word i.e., -\w+
                                    if rem_string.count('<')==0 or len(re.findall('^>\w+-\w+',rem_string))>0:
                                        st = ind+re.search('-\w+>',rem_string).end()-1
                                        temp = rem_string[st:]
                                        tag = re.findall('>(\w+)',temp)[0]
                                        break
                                    else:
                                        if len(re.findall('^>\w+>\w+',rem_string))==0: #subcase2 counterpart
                                            if n-flag == 1 and p>=0:
                                                st = ind
                                                temp = rem_string[st:]
                                                tag = re.findall('>(\w+)',temp)[0]
                                                break
                                        else: #subcase2: multiple tags closing
                                            #subsubcase1: multiple tags closing after the relation word
                                            if len(re.findall('-\w+(?:>\w+)+>(\w+)',rem_string))>0:  
                                                st = re.search('-\w+(?:>\w+)+>(\w+)',rem_string).end()-1
                                                tag = re.findall('-\w+(?:>\w+)+>(\w+)',rem_string)[0]
                                            else: #subsubcase2: normal tags after relation word
                                                g = re.findall('^(?:>\w+)*>\w+',rem_string)[0].count('>')
                                                if n-flag-g==1:
                                                    st = ind
                                                    temp = rem_string[st:]
                                                    tag = re.findall('>(\w+)',temp)[0]
                                            break
                                #remaining cases: remaining string starting with related word => -\w+ (or) compound => -<\w+-\w+>\w+
                                else: 
                                    if n-flag == 1:
                                        st = ind
                                        temp = rem_string[st:]
                                        tag = re.findall('>(\w+)',temp)[0]
                                        break
                            pre_tag_string = rem_string[:st]
                            num = len(raw_to_clean(pre_tag_string).split())
                            w.write(f'{i+1}\t{subword}\tComp{comp_len}\t_\t{i+num+1}\t{tag}'+'\n')
                            i += 1
                    c += 1
                    d += 1
            w.write(f'{i+1}\tDUMMY\tCompNo\t_\t{0}\troot\n')
            w.write('\n')
    return

quantization_config=BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_compute_dtype = precision,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load and prepare the dataset
def prepare_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        file = pd.read_csv(file_path)
        sources = file["Clean"]
        targets = file["Coarse_Span_Tagged"]
        # Create prompt format for instruction tuning
        formatted_data = []
        for src, tgt in zip(sources, targets):
            # Format as instruction prompt
            instruction = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful assistant Who helps me to find compounded words in sanskrit sentence.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n Given compound type label set [Avyayibhava, Tatpurusha, Bahuvrihi, Dvandva], identify the compounds in the sentence '{src}'. Answer only in the format <start_word_index1><end_word_index1>  <Compound Type>. \n\nDefinitions:\nTatpurusha – an endocentric compound where the first element (the attributive) determines the second.\nAvyayibhava – adverbial compounds composed of an indeclinable element and a noun, expressing an adverb or another indeclinable.\n Dvandva – copulative compounds of two or more noun stems joined by 'and'.\nBahuvrihi – exocentric compounds describing something by reference to its parts.\n\nText will be in segmented transliterated Sanskrit. There may be nested compounds.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            response = tgt
            
            formatted_data.append({
                "instruction": instruction,
                "response": response,
                "input": "",  # No additional input needed
                "source": src,  # Keep original source for evaluation
                "target": tgt   # Keep original target for evaluation
            })
        
        # Convert to DataFrame first, then to HF Dataset
        df = pd.DataFrame(formatted_data)
        dataset = Dataset.from_dict({
            "instruction": df["instruction"].tolist(),
            "response": df["response"].tolist(),
            "input": df["input"].tolist(),
            "source": df["source"].tolist(),
            "target": df["target"].tolist()
        })
        
        return dataset

# Load the tokenizer
def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

# Prepare the model with LoRA
def get_model():
    # Load the base model with 4-bit quantization for efficiency
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=precision,
        device_map="auto",
        use_cache=False,
        quantization_config=quantization_config
    )
    
    # Prepare the model for LoRA training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Print trainable parameters percentage
    
    return model

# Function to generate predictions and evaluate with USS & LSS
def evaluate_compounds(model, tokenizer, test_dataset):
    model.eval()
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    results_file = "results/results.csv"
    pred_file = "results/pred_file.txt"
    gold_input = "results/gold_input.txt"
    gold_input_nectis = "results/gold_input_NeCTIS_format.txt"
    
    # Create and open the results CSV file
    with open(results_file, 'w', newline='', encoding='utf-8') as csvfile, \
         open(gold_input, 'w', encoding='utf-8') as gold_f, \
         open(pred_file, 'w', encoding='utf-8') as pred_f:
        
        fieldnames = ['source_sentence', 'true_compounds', 'predicted_compounds', 'USS', 'LSS', 'exact_match']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        print("Generating compounds for test set...")
        for i, item in enumerate(tqdm(test_dataset)):
            source_text = item["source"]
            target_text = item["target"]
            
            # Write gold data
            gold_f.write(f"{target_text}\n")
            
            # Create prompt for compound detection
            prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful assistant Who helps me to find compounded words in sanskrit sentence.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n Given compound type label set [Avyayibhava, Tatpurusha, Bahuvrihi, Dvandva], identify the compounds in the sentence '{source_text}'. Answer only in the format <start_word_index1><end_word_index1>  <Compound Type>. \n\nDefinitions:\nTatpurusha – an endocentric compound where the first element (the attributive) determines the second.\nAvyayibhava – adverbial compounds composed of an indeclinable element and a noun, expressing an adverb or another indeclinable.\n Dvandva – copulative compounds of two or more noun stems joined by 'and'.\nBahuvrihi – exocentric compounds describing something by reference to its parts.\n\nText will be in segmented transliterated Sanskrit. There may be nested compounds.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Generate compound detection output
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=False
                )
            
            # Decode generated tokens
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract the assistant's response (removing system prompt)
            try:
                # Clean any model formatting in the response
                if "<|" in generated_text:
                    # Find the assistant's actual response
                    parts = generated_text.split("<|start_header_id|>assistant<|end_header_id|>")
                    if len(parts) > 1:
                        prediction = parts[1].strip()
                        if "<|" in prediction:  # Remove any trailing tokens
                            prediction = prediction.split("<|")[0].strip()
                    else:
                        prediction = generated_text
                else:
                    prediction = generated_text
            except Exception:
                prediction = generated_text  # Fallback if parsing fails
            
            # Write prediction to file
            pred_f.write(f"{prediction}\n")
            
    # Convert to NeCTIS format for evaluation
    Conversion(gold_input, gold_input_nectis)
    
    # Evaluate using USS and LSS metrics
    with open(gold_input_nectis) as t:
        with open(pred_file) as p:
            true_lines = t.readlines()
            pred_lines = p.readlines()
            
            # Calculate USS
            uss_score = unlabeled_metric(true_lines, pred_lines)
            
            # Calculate LSS and Exact Match
            metrics = metric(true_lines, pred_lines)
            lss_score = metrics[2]  # F1 score
            exact_match = metrics[4]  # Exact match percentage
            
    # Write final results
    with open(os.path.join(OUTPUT_DIR, "compound_metrics.txt"), "w") as f:
        f.write("===== COMPOUND DETECTION METRICS =====\n")
        f.write(f"USS: {uss_score}\n")
        f.write(f"LSS: {lss_score}\n")
        f.write(f"Exact Match: {exact_match}\n")
    
    print("\n===== COMPOUND DETECTION METRICS =====")
    print(f"USS: {uss_score}")
    print(f"LSS: {lss_score}")
    print(f"Exact Match: {exact_match}")
    print("===============================")
    
    # Return metrics as dictionary
    return {
        "uss": uss_score,
        "lss": lss_score,
        "exact_match": exact_match
    }

def tokenize_map_fn(examples):
    prompt_template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful assistant Who helps me to find compounded words in sanskrit sentence.<|eot_id|><|start_header_id|>user<|end_header_id|>{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    formatted_examples = []
    for instruction, response in zip(examples[1]["Clean"], examples[1]["Coarse_Span_Tagged"]):
        formatted_text = prompt_template.format(
            instruction=instruction,
        )
        formatted_examples.append(formatted_text)
    
    tokenizer = get_tokenizer() 
    
    tokenized_inputs = tokenizer(
        formatted_examples,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"  # Return Python lists
    )
    input_ids = tokenized_inputs["input_ids"].squeeze().tolist()
    attention_mask = tokenized_inputs["attention_mask"].squeeze().tolist()
    
    if isinstance(input_ids, list) and all(isinstance(item, list) for item in input_ids):
        input_ids = input_ids[0]
    if isinstance(attention_mask, list) and all(isinstance(item, list) for item in attention_mask):
        attention_mask = attention_mask[0]
    
    # Create labels (same as input_ids for causal LM)
    labels = input_ids.copy()   
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def main():
    print("Loading training dataset...")
    train_dataset = pd.read_csv(TRAIN_PATH)
    test_dataset = pd.read_csv(TEST_PATH)
    dev_dataset = pd.read_csv(DEV_PATH)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(dev_dataset)}")
    
    print("Loading tokenizer...")
    tokenizer = get_tokenizer()
    
    print("Tokenizing dataset...")
    # Use tokenizer inside the map function directly to avoid closure issues
    tokenized_train_dataset = [tokenize_map_fn(x) for x in train_dataset[["Clean","Coarse_Span_Tagged"]].iterrows()]
    tokenized_dev_dataset = [tokenize_map_fn(x) for x in dev_dataset[["Clean","Coarse_Span_Tagged"]].iterrows()]
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=WARMUP_STEPS,
        logging_dir="./logs",
        logging_steps=50,
        report_to="tensorboard",
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=True,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    )
    
    print("Loading model with LoRA configuration...")
    model = get_model()
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're using causal language modeling, not masked
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_dev_dataset,
        data_collator=data_collator,
    )
    
    # Start training
    print("Starting training with LoRA...")
    trainer.train()
    
    # Save the final model (only LoRA adapters)
    print("Saving LoRA adapters...")
    model.save_pretrained(os.path.join(OUTPUT_DIR, "lora_adapters"))
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"Training complete! LoRA adapters saved to {os.path.join(OUTPUT_DIR, 'lora_adapters')}")
    
    # Load test data for USS/LSS evaluation
    if os.path.exists(TEST_PATH):
        print(f"Loading test dataset from {TEST_PATH}...")
        test_dataset = prepare_dataset(TEST_PATH)
        
        print(f"Test dataset size: {len(test_dataset)}")
        print("Evaluating model on test set...")
        
        # Run evaluation to calculate USS and LSS scores
        metrics = evaluate_compounds(model, tokenizer, test_dataset)
        
        print("\n===== COMPOUND DETECTION METRICS =====")
        print(f"USS Score: {metrics['uss']:.2f}")
        print(f"LSS Score: {metrics['lss']:.2f}")
        print(f"Exact Match: {metrics['exact_match']:.2f}")
        print("===============================")
        
        # Save metrics to file
        with open(os.path.join(OUTPUT_DIR, "compound_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)
            
    else:
        print(f"Test file {TEST_PATH} not found, skipping metrics calculation.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()  # Print the full traceback for detailed debugging
        
        # Log error to compounding_results.txt
        with open("compounding_results.txt", "a", encoding="utf-8") as f:
            f.write("\n" + "="*50 + "\n")
            f.write(f"ERROR at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Error message: {str(e)}\n")