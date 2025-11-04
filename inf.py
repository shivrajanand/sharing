import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from unsloth import FastLanguageModel
from transformers import EarlyStoppingCallback

# Load the necessary model and tokenizer (using the same setup as your script)
# Assuming you've already fine-tuned and saved the model at this point
OUT_DIR = "/home/shivraj-pg/DEPNECT/OUT_gemma4B"  # replace with your model output path

# Set up the tokenizer and model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/home/shivraj-pg/DEPNECT/OUT_gemma4B/checkpoint-1500",  # or your own fine-tuned model
    max_seq_length=1024,
    load_in_4bit=False,
    load_in_8bit=False,
    load_in_16bit=True,
    full_finetuning=False,
)

# Function to generate compound output
def get_compound(sentence):
    prompt = TEMPLATE.format(
        system=SYSTEM, sentence=sentence.strip(), sandhi="", compound_types=""
    )
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=None,
            temperature=0.25,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.05,
        )
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

# Set your system prompt
SYSTEM = """
You are an expert in Sanskrit grammar, who identifies and classifies compounds in the given Sanskrit sentence. You will be given the original sentence and the sentence with each compounded word is broken down. 
Follow these rules strictly:
 1. Output only a single line in the format: `1 विकसित Comp6_Start 4 Tatpurusha`
 example: `'1 स Comp2_Start 2 Bahuvrihi\n2 सर्षपं Comp2_End 11 Comp_root...'`
 2. Only use the following **4 compound types**. Do **not** invent or include other types:
- **Tatpurusha**: An endocentric compound where the first element (the attributive) determines the second.
- **Avyayibhava**: An adverbial compound made of an indeclinable element and a noun, expressing an adverbial meaning.
 - **Dvandva**: A copulative compound where two or more noun stems are joined by 'and'.
- **Bahuvrihi**: An exocentric compound that describes something by referring to its parts.
3. The sentence may contain **nested compounds** or **non-compounded words** — handle appropriately.
4. Maintain strict formatting and provide **only the answer line**. Do not include explanations.
5. The start or end indexes must not exceed the number of words in the sentence.
6. Answer in the latin script only, there shouldn't be any devnagari in the answer
"""

TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system}<|eot_id|>\n<|start_header_id|>user<|end_header_id|>Sanskrit text:{sentence}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>Sandhi-Text: {sandhi}\nCompoundTypes:{compound_types}\n<|eot_id|>"""

# Load the test CSV
TEST_CSV = "/home/shivraj-pg/DEPNECT/DATASETS/Test_withoutContext_coarse.csv"

# Read the test CSV and clean up the columns
test_df = pd.read_csv(TEST_CSV)
test_df = test_df.drop(columns=["Unnamed: 0"], errors="ignore").reset_index(drop=True)

# Check the columns to ensure 'sentence' exists
print("Columns in TEST_CSV:", test_df.columns)

# Ensure model output column exists
if "model_out" not in test_df.columns:
    test_df["model_out"] = ""

# Iterate through the test dataset and generate compound information
print("Generating compound info...")
for idx in tqdm(test_df.index, desc="compound"):
    src = str(test_df.at[idx, "sentence"]).strip()
    if pd.isna(src) or src == "":
        test_df.at[idx, "model_out"] = "NO_SOURCE"
        continue
    test_df.at[idx, "model_out"] = get_compound(src)

# Save the predictions
out_csv = TEST_CSV.replace(".csv", "_gemma3-4b-modelOut.csv")
test_df.to_csv(out_csv, index=False)
print("Predictions saved to:", out_csv)

# Clean up
del model
torch.cuda.empty_cache()
