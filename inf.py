import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

# ---------------------
# Config
# ---------------------
OUT_DIR = "/home/shivraj-pg/DEPNECT/OUT_gemma4B"
TEST_CSV = "/home/shivraj-pg/DEPNECT/DATASETS/Test_withoutContext_coarse.csv"
BATCH_SIZE = 16  # adjust based on GPU memory

# ---------------------
# Load model and tokenizer
# ---------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/home/shivraj-pg/DEPNECT/OUT_gemma4B/checkpoint-1500",
    max_seq_length=1024,
    load_in_4bit=False,
    load_in_8bit=False,
    load_in_16bit=True,
    full_finetuning=False,
)

SYSTEM = """
You are an expert in Sanskrit grammar, who identifies and classifies compounds in the given Sanskrit sentence. You will be given the original sentence and the sentence with each compounded word is broken down. 
Follow these rules strictly:
 1. Output only a single line in the format: `1 विकसित Comp6_Start 4 Tatpurusha`
 example: `'1 स Comp2_Start 2 Bahuvrihi\n2 सर्षपं Comp2_End 11 Comp_root...'`
 2. Only use the following **4 compound types**. Do **not** invent or include other types:
- **Tatpurusha**
- **Avyayibhava**
- **Dvandva**
- **Bahuvrihi**
3. The sentence may contain nested compounds or non-compounded words.
4. Maintain strict formatting and provide **only the answer line**.
5. Answer in latin script only, no Devanagari.
"""

TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system}<|eot_id|>\n<|start_header_id|>user<|end_header_id|>Sanskrit text:{sentence}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>Sandhi-Text: {sandhi}\nCompoundTypes:{compound_types}\n<|eot_id|>"""

# ---------------------
# Batch inference function
# ---------------------
def generate_batch(sentences):
    prompts = [
        TEMPLATE.format(system=SYSTEM, sentence=s.strip(), sandhi="", compound_types="")
        for s in sentences
    ]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.25,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.05,
        )

    decoded = tokenizer.batch_decode(
        outputs[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )

    return [d.strip() for d in decoded]

# ---------------------
# Load data
# ---------------------
test_df = pd.read_csv(TEST_CSV)
test_df = test_df.drop(columns=["Unnamed: 0"], errors="ignore").reset_index(drop=True)
if "model_out" not in test_df.columns:
    test_df["model_out"] = ""

# ---------------------
# Run batch inference
# ---------------------
sentences = test_df["sentence"].astype(str).fillna("").tolist()

for start in tqdm(range(0, len(sentences), BATCH_SIZE), desc="Generating"):
    end = start + BATCH_SIZE
    batch_sents = sentences[start:end]
    # handle empty inputs
    if all(s.strip() == "" for s in batch_sents):
        test_df.loc[start:end-1, "model_out"] = "NO_SOURCE"
        continue
    outputs = generate_batch(batch_sents)
    test_df.loc[start:end-1, "model_out"] = outputs

# ---------------------
# Save results
# ---------------------
out_csv = TEST_CSV.replace(".csv", "_gemma3-4b-batchOut.csv")
test_df.to_csv(out_csv, index=False)
print("✅ Predictions saved to:", out_csv)

# Cleanup
del model
torch.cuda.empty_cache()
