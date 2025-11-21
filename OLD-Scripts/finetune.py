import os
import json
import torch
import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType,
)
from datasets import Dataset

# ---------------- CONFIG ----------------

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
OUTPUT_DIR = "./llama-3.2-3B-finetuned-sa-hi"

TRAIN_PATH = "./Datasets/No Context CSV files/train.csv"
DEV_PATH = "./Datasets/No Context CSV files/dev.csv"

MAX_LENGTH = 512
BATCH_SIZE = 4
EPOCHS = 10
LR = 1e-5
WARMUP_STEPS = 100
SAVE_STEPS = 100

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj","k_proj","v_proj","o_proj",
    "gate_proj","up_proj","down_proj"
]

device = "cuda" if torch.cuda.is_available() else "cpu"
precision = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=precision
)

# ---------------- DATASET PREP ----------------

def prepare_dataset(csv_path):
    df = pd.read_csv(csv_path)

    formatted_data = []
    for src, tgt in zip(df["Clean"], df["Coarse_Span_Tagged"]):
        instruction = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
            "You are a helpful assistant Who helps me to find compounded words in sanskrit sentence."
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"Given compound type label set [Avyayibhava, Tatpurusha, Bahuvrihi, Dvandva], "
            f"identify the compounds in the sentence '{src}'. Answer only in the format "
            "<start_word_index1><end_word_index1>  <Compound Type>. \n\n"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        formatted_data.append({
            "instruction": instruction,
            "response": tgt,
            "Clean": src,
            "Coarse_Span_Tagged": tgt
        })

    return Dataset.from_pandas(pd.DataFrame(formatted_data))

def tokenize(examples, tokenizer):
    text = [
        ex["instruction"] + ex["response"]
        for ex in examples
    ]
    tok = tokenizer(
        text,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length"
    )
    tok["labels"] = tok["input_ids"].copy()
    return tok

# ---------------- TOKENIZER ----------------

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

# ---------------- MODEL ----------------

def get_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=precision,
        quantization_config=quantization_config
    )
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM
    )
    return get_peft_model(model, lora_cfg)

# ---------------- MAIN ----------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tokenizer = get_tokenizer()

    train_dataset = prepare_dataset(TRAIN_PATH)
    dev_dataset = prepare_dataset(DEV_PATH)

    train_dataset = train_dataset.map(lambda x: tokenize([x], tokenizer)[0])
    dev_dataset = dev_dataset.map(lambda x: tokenize([x], tokenizer)[0])

    model = get_model()

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        warmup_steps=WARMUP_STEPS,
        save_steps=SAVE_STEPS,
        logging_steps=50,
        evaluation_strategy="steps",
        fp16=True,
        gradient_accumulation_steps=4,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        ),
    )

    trainer.train()

    model.save_pretrained(os.path.join(OUTPUT_DIR, "lora"))
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
