import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_DIR = "./llama-3.2-3B-finetuned-sa-hi/lora"
BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
TEST_PATH = "./Datasets/No Context CSV files/test.csv"
OUT_PRED = "pred_file.txt"

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, device_map="auto"
    )

    model = PeftModel.from_pretrained(base, MODEL_DIR)
    model.eval()
    return tokenizer, model

def make_prompt(sentence):
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
        "You are a helpful assistant Who helps me to find compounded words in sanskrit sentence."
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"Identify compounds in the sentence '{sentence}'."
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

def main():
    df = pd.read_csv(TEST_PATH)
    sentences = df["Clean"]

    tokenizer, model = load_model()

    with open(OUT_PRED, "w", encoding="utf-8") as f:
        for sent in sentences:
            prompt = make_prompt(sent)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    temperature=0.7
                )

            generated = tokenizer.decode(output[0], skip_special_tokens=True)
            f.write(generated + "\n")

if __name__ == "__main__":
    main()
