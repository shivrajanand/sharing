import unsloth
import torch
import pandas as pd
import gc
from tqdm import tqdm
from datasets import Dataset
from transformers import TrainingArguments
from unsloth import FastLanguageModel, FastModel
from trl import SFTTrainer
from transformers import EarlyStoppingCallback


early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)

TRAIN_CSV = "/home/shivraj-pg/DEPNECT/DATASETS/Train_withoutContext_coarse.csv"
TEST_CSV = "/home/shivraj-pg/DEPNECT/DATASETS/Test_withoutContext_coarse.csv"
OUT_DIR = "/home/shivraj-pg/DEPNECT/OUT_gemma27B"

MAX_SEQ = 1024
R, ALPHA = 16, 32
DROPOUT = 0.05
LR = 2e-4
BATCH = 4
GRAD_ACC = 16
EPOCHS = 10
SAVE_STEPS = 250
LOG_STEPS = 50

SYSTEM = """
You are an expert in Sanskrit grammar, who identifies and classifies compounds in the given Sanskrit sentence. You will be given the original sentence and the sentence with each compounded word is broken down 
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
# ---------- dataset sanitiser ----------


def csv_to_ds(path):
    df = pd.read_csv(
        path)[["sentence", "sandhied-sent", "compound-info"]].dropna()
    texts = []
    for _, r in df.iterrows():
        txt = TEMPLATE.format(system=SYSTEM,
                              sentence=str(r["sentence"]).strip(),
                              sandhi=str(r["sandhied-sent"]).strip(),
                              compound_types=str(r["compound-info"].strip()))
        texts.append(txt)
    return Dataset.from_dict({"text": texts})


# ---------- train/val split ----------
raw_ds = csv_to_ds(TRAIN_CSV)
split = raw_ds.train_test_split(test_size=0.1, seed=42)
train_ds = split["train"]
eval_ds = split["test"]
print("Train:", len(train_ds), "Eval:", len(eval_ds))

# -------- model --------------

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-3-27b-it",
    device_map="auto",
    # model_name="google/gemma-3-27b-it",
    max_seq_length=MAX_SEQ,
    load_in_4bit=False,
    load_in_8bit=False,
    load_in_16bit=True,
    full_finetuning=False,

)

model = model.to_empty(device="cuda")
model.load_weights()

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,  # Turn off for just text!
    finetune_language_layers=True,  # Should leave on!
    finetune_attention_modules=True,  # Attention good for GRPO
    finetune_mlp_modules=True,  # SHould leave on always!

    r=R,           # Larger = higher accuracy, but might overfit
    lora_alpha=ALPHA,  # Recommended alpha == r at least
    lora_dropout=DROPOUT,
    bias="none",
    random_state=3407,
)

# ---------- training ----------
training_args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACC,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    fp16=False,
    bf16=True,                   # BF16 activations
    logging_steps=LOG_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    eval_strategy="steps",
    eval_steps=SAVE_STEPS,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    optim="adamw_torch",
    seed=42,
    report_to="none",
    dataloader_num_workers=4
)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ,
    args=training_args,
    callbacks=[early_stopping_callback],
)
trainer.train(resume_from_checkpoint=False)
trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

FastLanguageModel.for_inference(model)


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

# ---------- inference (no length cap) ----------
test_df = pd.read_csv(TEST_CSV)
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

# ---------- clean-up ----------
del model, trainer
gc.collect()
torch.cuda.empty_cache()
