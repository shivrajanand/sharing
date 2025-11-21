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
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType,
)
from datasets import Dataset
from transformers import EarlyStoppingCallback

# ---------------- CONFIG ----------------

early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)
MODEL_NAME = "google/gemma-3-4b-it"
OUTPUT_DIR = "/home/shivraj-pg/DEPNECT/OUT_Gemma4B_newscript"

TRAIN_PATH = "/home/shivraj-pg/DEPNECT/conllu-style-csv/without-context-coarse-train.csv"
DEV_PATH = "/home/shivraj-pg/DEPNECT/conllu-style-csv/without-context-coarse-dev.csv"

MAX_LENGTH = 3500
BATCH_SIZE = 1
EPOCHS = 100
LR = 1e-5
WARMUP_STEPS = 100
SAVE_STEPS = 100
LOG_STEPS = 10

LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.1

precision = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# ---------------- SYSTEM + TEMPLATE ----------------

SYSTEM = """
You are an expert in Sanskrit grammar, who identifies and classifies compounds in the given Sanskrit sentence. You will be given the original sentence. First break the sentence in compounds.
Follow these rules strictly:
1. Only use the following 4 compound types. Do not invent or include other types:
    - Tatpurusha: An endocentric compound where the first element (the attributive) determines the second.
    - Avyayibhava: An adverbial compound made of an indeclinable element and a noun, expressing an adverbial meaning.
    - Dvandva: A copulative compound where two or more noun stems are joined by 'and'.
    - Bahuvrihi: An exocentric compound that describes something by referring to its parts.
2. The sentence may contain nested compounds or non-compounded words — handle appropriately.
3. Maintain strict formatting and provide only the answer line. Do not include explanations.
4. The start or end indexes must not exceed the number of words in the sentence.
5. Answer in the devnagri script only, there shouldn't be any latin in the answer

Text:
{INPUT}

Return strictly in JSON with keys:
{
  "tokens": [...] ,
  "compounds":""
}

Rules:
- Tokenize by meaningful Sanskrit units.
- span = inclusive of start index, exclusive of end index.
- If multiple nested samāsa exist, include all.
- If none, return empty lists for compounds.
- Do not output anything outside JSON.
- The compound data is in tab separated conllu format as given in example.


Example 1:
Input: रागआदिरोगान् सततअनुषक्तान् अशेषकायप्रसृतान् अशेषान् औत्सुक्यमोहअरतिदान् जघान (यः) .
Output:
 {'tokens': ['राग', 'आदि', 'रोगान्', 'सतत', 'अनुषक्तान्', 'अ', 'शेष', 'काय', 'प्रसृतान्', 'अ', 'शेषान्', 'औत्सुक्य', 'मोह', 'अ', 'रति', 'दान्', 'जघान', '(यः)', '.'], 'compounds': '1\\tराग-\\tराग-आदि-रोगान्\\t2\\tComp3\\t_\\tबहुव्रीहिः\\n2\\tआदि-\\t--\\t3\\tComp3\\t_\\tकर्मधारयः\\n3\\tरोगान्\\t--\\t20\\tComp3\\t_\\tComp_root\\n4\\tसतत-\\tसतत-अनुषक्तान्\\t5\\tComp2\\t_\\tकर्मधारयः\\n5\\tअनुषक्तान्\\t--\\t20\\tComp2\\t_\\tComp_root\\n6\\tअ-\\tअ-शेष-काय-प्रसृतान्\\t7\\tComp4\\t_\\tनञ्-तत्पुरुषः\\n7\\tशेष-\\t--\\t8\\tComp4\\t_\\tकर्मधारयः\\n8\\tकाय-\\t--\\t9\\tComp4\\t_\\tसप्तमी-तत्पुरुषः\\n9\\tप्रसृतान्\\t--\\t20\\tComp4\\t_\\tComp_root\\n10\\tअ-\\tअ-शेषान्\\t11\\tComp2\\t_\\tअस्त्यर्थ-मध्यमपदलोपी(नञ्)-बहुव्रीहिः\\n11\\tशेषान्\\t--\\t20\\tComp2\\t_\\tComp_root\\n12\\tऔत्सुक्य-\\tऔत्सुक्य-मोह-अरति-दान्\\t13\\tComp4\\t_\\tइतरेतर-द्वन्द्वः\\n13\\tमोह-\\t--\\t14\\tComp4\\t_\\tइतरेतर-द्वन्द्वः\\n14\\tअ-\\t--\\t15\\tComp4\\t_\\tनञ्-तत्पुरुषः\\n15\\tरति-\\t--\\t16\\tComp4\\t_\\tComp_root\\n16\\tदान्\\t--\\t20\\tComp4\\t_\\tविशेषणम्\\n17\\tजघान\\tजघान\\t20\\tCompNo\\t_\\tNo_rel\\n18\\t(यः)\\t(यः)\\t20\\tCompNo\\t_\\tNo_rel\\n19\\t.\\t.\\t20\\tCompNo\\t_\\tNo_rel\\n20\\tDUMMY\\t_\\t0\\tCompNo\\t_\\troot '}
   
Example 2:
Input: 'अपूर्ववैद्याय नमः अस्तु तस्मै .
Output:
 {'tokens': ['अ', 'पूर्व', 'वैद्याय', 'नमः', 'अस्तु', 'तस्मै', '.'], 
 'compounds': '1\\tअ-\\tअ-पूर्व-वैद्याय\\t2\\tComp3\\t_\\tअस्त्यर्थ-मध्यमपदलोपी(नञ्)-बहुव्रीहिः\\n2\\tपूर्व-\\t--\\t3\\tComp3\\t_\\tकर्मधारयः\\n3\\tवैद्याय\\t--\\t8\\tComp3\\t_\\tComp_root\\n4\\tनमः\\tनमः\\t8\\tCompNo\\t_\\tNo_rel\\n5\\tअस्तु\\tअस्तु\\t8\\tCompNo\\t_\\tNo_rel\\n6\\tतस्मै\\tतस्मै\\t8\\tCompNo\\t_\\tNo_rel\\n7\\t.\\t.\\t8\\tCompNo\\t_\\tNo_rel\\n8\\tDUMMY\\t_\\t0\\tCompNo\\t_\\troot'}
    
Input: अथ अतः आयुष्कामीयम् अध्यायम् व्याख्यास्यामः .
Output:
 {'tokens': ['अथ', 'अतः', 'आयुष्कामीयम्', 'अध्यायम्', 'व्याख्यास्यामः', '.'], 
 'compounds': '1\\tअथ\\tअथ\\t7\\tCompNo\\t_\\tNo_rel\\n2\\tअतः\\tअतः\\t7\\tCompNo\\t_\\tNo_rel\\n3\\tआयुष्कामीयम्\\tआयुष्कामीयम्\\t7\\tCompNo\\t_\\tNo_rel\\n4\\tअध्यायम्\\tअध्यायम्\\t7\\tCompNo\\t_\\tNo_rel\\n5\\tव्याख्यास्यामः\\tव्याख्यास्यामः\\t7\\tCompNo\\t_\\tNo_rel\\n6\\t.\\t.\\t7\\tCompNo\\t_\\tNo_rel\\n7\\tDUMMY\\t_\\t0\\tCompNo\\t_\\troot'}
"""

TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system}<|eot_id|>
<|start_header_id|>user<|end_header_id|>Now analyze:{INPUT}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>Answer: {OUTPUT}<|eot_id|>"""

# ---------------- DATASET PREP ----------------


def prepare_dataset(csv_path, system_prompt):
    df = pd.read_csv(csv_path)[["sentence", "gold"]].dropna()

    formatted_data = []
    for src, tgt in zip(df["sentence"], df["gold"]):

        instruction = TEMPLATE.format(
            system=system_prompt,
            INPUT=str(src).strip(),
            OUTPUT=""                       # output stays empty in prompt
        )

        formatted_data.append({
            "instruction": instruction,
            "response": str(tgt).strip()
        })

    return Dataset.from_pandas(pd.DataFrame(formatted_data))


# ---------------- TOKENIZATION ----------------


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token


def tokenize(batch):
    texts = [instr + resp for instr,
             resp in zip(batch["instruction"], batch["response"])]
    tok = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
        return_attention_mask=True
    )
    return tok


# ---------------- MODEL ----------------


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=precision,
    attn_implementation="flash_attention_2"
)

model = prepare_model_for_kbit_training(model)

lora_cfg = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "qkv_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# ---------------- MAIN ----------------


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_dataset = prepare_dataset(TRAIN_PATH, SYSTEM)
    dev_dataset = prepare_dataset(DEV_PATH, SYSTEM)

    train_dataset = train_dataset.map(
        lambda batch: tokenize(batch),
        batched=True
    )
    dev_dataset = dev_dataset.map(
        lambda batch: tokenize(batch),
        batched=True
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        warmup_steps=WARMUP_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        eval_strategy="steps",
        logging_steps=LOG_STEPS,
        fp16=(precision == torch.float16),
        bf16=(precision == torch.bfloat16),
        gradient_accumulation_steps=4,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,

    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        callbacks=[early_stopping_callback],
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        ),
    )

    trainer.train()

    model.save_pretrained(os.path.join(OUTPUT_DIR, "lora"))
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    try:
        main()
        print("TRAINING SUCCESSFULL!")
    except Exception as e:
        print("ERROR! PROGRAM FAILED")
        print(e)
