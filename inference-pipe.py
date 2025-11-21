import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
import json

# ---------- CONFIG ----------
BASE_MODEL = "google/gemma-3-4b-it"
MODEL_DIR = "/home/shivraj-pg/DEPNECT/OUT_gemma4B_conllu/checkpoint-800"  # merged fine-tuned model
INPUT_CSV = "/home/shivraj-pg/DEPNECT/DATASETS/without_context_coarse_test.csv"
OUTPUT_CSV = INPUT_CSV.replace(".csv", "_gemma3-4B-it_pred_conllu.csv")

BATCH_SIZE = 64
MAX_NEW_TOKENS = 4096
TEMPERATURE = 0.0
TOP_P = 1.0
DO_SAMPLE = False
SAVE_INTERVAL = 200

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
            "<|start_header_id|>assistant<|end_header_id|>\n"
            f"Example {i} Output:\n{out_json}\n"
        )
        blocks.append(block)
    return "\n".join(blocks)


demo_block = create_demo_block(DEMO_EXAMPLES)


def build_prompt(system_text: str, demo_block: str, target_sentence: str) -> str:
    # This returns only the prompt (no target gold).
    return (
        "<|begin_of_text|>\n"
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{system_text}\n\n"
        "<|eot_id|>\n"
        f"{demo_block}\n"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        "Now analyze this sentence:\n"
        f"{target_sentence}\n"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

# ---------- LOAD PIPELINE ----------
print("Loading pipeline with fine-tuned model...")
pipe = pipeline(
    "text-generation",
    model=MODEL_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
print("Pipeline loaded successfully!")

# ---------- LOAD DATA ----------
df = pd.read_csv(INPUT_CSV)
if "model_out" not in df.columns:
    df["model_out"] = ""

n = len(df)
print(f"Loaded {n} rows for inference. Batch size = {BATCH_SIZE}")

# ---------- INFERENCE ----------
prompts, row_indices = [], []
processed = 0

for idx in tqdm(range(n), desc="Batched Inference"):
    sentence = str(df.at[idx, "sentence"]).strip() if "sentence" in df.columns else ""

    if not sentence or pd.isna(sentence):
        df.at[idx, "model_out"] = "NO_SOURCE"
        processed += 1
        if processed % SAVE_INTERVAL == 0:
            df.to_csv(OUTPUT_CSV, index=False)
        continue

    # Build prompt inline
    prompt = build_prompt(SYSTEM, demo_block, str(sentence).strip())
    prompts.append(prompt)
    row_indices.append(idx)

    # Run batch when full or last row
    if len(prompts) == BATCH_SIZE or idx == n - 1:
        try:
            outputs = pipe(
                prompts,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=DO_SAMPLE,
                pad_token_id=pipe.tokenizer.eos_token_id,
                batch_size=BATCH_SIZE,
            )

            # Extract and assign clean model output
            for j, out in enumerate(outputs):
                text = out[0]["generated_text"]
                answer = text.split("Answer:")[-1].strip()
                df.at[row_indices[j], "model_out"] = answer

        except Exception as e:
            print(f"Batch generation error at rows {row_indices}: {e}")
            for j in row_indices:
                df.at[j, "model_out"] = f"ERROR: {str(e)}"

        # Clear caches
        processed += len(prompts)
        prompts, row_indices = [], []

        # Save periodically
        if processed % SAVE_INTERVAL == 0:
            print(f"Saving intermediate results after {processed} rows...")
            df.to_csv(OUTPUT_CSV, index=False)

# ---------- FINAL SAVE ----------
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Inference completed. Results saved to: {OUTPUT_CSV}")

torch.cuda.empty_cache()
