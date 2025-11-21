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
import json
# ---------- configs ----------
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)

TRAIN_CSV = "/home/shivraj-pg/DEPNECT/conllu-style-csv/without-context-coarse-train.csv"
OUT_DIR = "/home/shivraj-pg/DEPNECT/OUT_gemma4B_conllu"

MAX_SEQ = 3500
R, ALPHA = 64, 128
DROPOUT = 0.1
LR = 1e-5
BATCH = 16
GRAD_ACC = 16
EPOCHS = 50
SAVE_STEPS = 100
LOG_STEPS = 10

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

Return strictly in JSON with keys:
{
  "tokens": [...] ,
  "compounds":""
}

Rules:
- Tokenize by meaningful Sanskrit units.
- If multiple nested samāsa exist, include all.
- If none, return empty lists for compounds.
- Do not output anything outside JSON.
- The compound data is in tab separated conllu format as given in example.
"""

ex1_input = "रागआदिरोगान् सततअनुषक्तान् अशेषकायप्रसृतान् अशेषान् औत्सुक्यमोहअरतिदान् जघान (यः) ."

ex1_output = {
    "tokens": ["राग", "आदि", "रोगान्", "सतत", "अनुषक्तान्", "अ", "शेष", "काय", "प्रसृतान्", "अ", "शेषान्", "औत्सुक्य", "मोह", "अ", "रति", "दान्", "जघान", "(यः)", "."],
    "compounds": "1\tराग-\tराग-आदि-रोगान्\t2\tComp3\t_\tबहुव्रीहिः\n2\tआदि-\t--\t3\tComp3\t_\tकर्मधारयः\n3\tरोगान्\t--\t20\tComp3\t_\tComp_root\n4\tसतत-\tसतत-अनुषक्तान्\t5\tComp2\t_\tकर्मधारयः\n5\tअनुषक्तान्\t--\t20\tComp2\t_\tComp_root\n6\tअ-\tअ-शेष-काय-प्रसृतान्\t7\tComp4\t_\tनञ्-तत्पुरुषः\n7\tशेष-\t--\t8\tComp4\t_\tकर्मधारयः\n8\tकाय-\t--\t9\tComp4\t_\tसप्तमी-तत्पुरुषः\n9\tप्रसृतान्\t--\t20\tComp4\t_\tComp_root\n10\tअ-\tअ-शेषान्\t11\tComp2\t_\tअस्त्यर्थ-मध्यमपदलोपी(नञ्)-बहुव्रीहिः\n11\tशेषान्\t--\t20\tComp2\t_\tComp_root\n12\tऔत्सुक्य-\tऔत्सुक्य-मोह-अरति-दान्\t13\tComp4\t_\tइतरेतर-द्वन्द्वः\n13\tमोह-\t--\t14\tComp4\t_\tइतरेतर-द्वन्द्वः\n14\tअ-\t--\t15\tComp4\t_\tनञ्-तत्पुरुषः\n15\tरति-\t--\t16\tComp4\t_\tComp_root\n16\tदान्\t--\t20\tComp4\t_\tविशेषणम्\n17\tजघान\tजघान\t20\tCompNo\t_\tNo_rel\n18\t(यः)\t(यः)\t20\tCompNo\t_\tNo_rel\n19\t.\t.\t20\tCompNo\t_\tNo_rel\n20\tDUMMY\t_\t0\tCompNo\t_\troot"
}
ex2_input = "अपूर्ववैद्याय नमः अस्तु तस्मै ."

ex2_output = {
    "tokens": ["अ", "पूर्व", "वैद्याय", "नमः", "अस्तु", "तस्मै", "."],
    "compounds": "1\tअ-\tअ-पूर्व-वैद्याय\t2\tComp3\t_\tअस्त्यर्थ-मध्यमपदलोपी(नञ्)-बहुव्रीहिः\n2\tपूर्व-\t--\t3\tComp3\t_\tकर्मधारयः\n3\tवैद्याय\t--\t8\tComp3\t_\tComp_root\n4\tनमः\tनमः\t8\tCompNo\t_\tNo_rel\n5\tअस्तु\tअस्तु\t8\tCompNo\t_\tNo_rel\n6\tतस्मै\tतस्मै\t8\tCompNo\t_\tNo_rel\n7\t.\t.\t8\tCompNo\t_\tNo_rel\n8\tDUMMY\t_\t0\tCompNo\t_\troot"
}
ex3_input = "अथ अतः आयुष्कामीयम् अध्यायम् व्याख्यास्यामः ."

ex3_output = {
    "tokens": ["अथ", "अतः", "आयुष्कामीयम्", "अध्यायम्", "व्याख्यास्यामः", "."],
    "compounds": "1\tअथ\tअथ\t7\tCompNo\t_\tNo_rel\n2\tअतः\tअतः\t7\tCompNo\t_\tNo_rel\n3\tआयुष्कामीयम्\tआयुष्कामीयम्\t7\tCompNo\t_\tNo_rel\n4\tअध्यायम्\tअध्यायम्\t7\tCompNo\t_\tNo_rel\n5\tव्याख्यास्यामः\tव्याख्यास्यामः\t7\tCompNo\t_\tNo_rel\n6\t.\t.\t7\tCompNo\t_\tNo_rel\n7\tDUMMY\t_\t0\tCompNo\t_\troot"
}


DEMO_EXAMPLES = [
    (ex1_input, json.dumps(ex1_output, ensure_ascii=False)),
    (ex2_input, json.dumps(ex2_output, ensure_ascii=False)),
    (ex3_input, json.dumps(ex3_output, ensure_ascii=False)),
]

# ---------- dataset prep ----------


def create_demo_block(demos):
    blocks = []
    for i, (inp, out_json) in enumerate(demos, 1):
        block = (
            "<|start_header_id|>user<|end_header_id|>\n"
            f"Example {i} Input:\n{inp}\n"
            "<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\n"
            f"Example {i} Output:\n{out_json}\n"
            "<|eot_id|>\n"
        )
        blocks.append(block)
    return "".join(blocks)



demo_block = create_demo_block(DEMO_EXAMPLES)


def build_prompt(system_text, demo_block, target_sentence):
    return (
        "<|begin_of_text|>\n"
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{system_text}\n"
        "<|eot_id|>\n"
        f"{demo_block}"
        "<|start_header_id|>user<|end_header_id|>\n"
        "Now analyze this sentence:\n"
        f"{target_sentence}\n"
        "<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )



def csv_to_ds(path):
    df = pd.read_csv(path)[["sentence", "gold"]].dropna()

    texts = []
    for _, r in df.iterrows():
        prompt = build_prompt(SYSTEM, demo_block, str(r["sentence"]).strip())
        # gold must be strict JSON string already in your dataset
        gold = str(r["gold"]).strip()
        # final string = prompt + gold (assistant completion)
        full_text = prompt + gold
        texts.append(full_text)

    return Dataset.from_dict({"text": texts})


# ---------- train/val split ----------
raw_ds = csv_to_ds(TRAIN_CSV)
split = raw_ds.train_test_split(test_size=0.1, seed=42)
train_ds = split["train"]
eval_ds = split["test"]
print("Train:", len(train_ds), "Eval:", len(eval_ds))

# ---------- model setup ----------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-3-4b-it",
    device_map="auto",
    max_seq_length=MAX_SEQ,
    load_in_4bit=False,
    load_in_8bit=False,
    load_in_16bit=True,
    full_finetuning=False,
)

# model = model.to_empty(device="cuda")

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=False,
    r=R,
    lora_alpha=ALPHA,
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
    bf16=False,
    fp16_full_eval=False,      
    bf16_full_eval=False,
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
    dataloader_num_workers=4,
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
    remove_unused_columns=False
)

# ---------- run training ----------
trainer.train(resume_from_checkpoint=True)
trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

FastLanguageModel.for_inference(model)

# ---------- clean-up ----------
del model, trainer
gc.collect()
torch.cuda.empty_cache()
