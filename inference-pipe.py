import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import pipeline

# ---------- CONFIG ----------
BASE_MODEL = "google/gemma-3-4b-it"
MODEL_DIR = "/home/shivraj-pg/DEPNECT/OUT_finetuned_gemma3_4b_ft"  # merged fine-tuned model
INPUT_CSV = "/home/shivraj-pg/DEPNECT/DATASETS/without_context_coarse_test.csv"
OUTPUT_CSV = INPUT_CSV.replace(".csv", "_gemma3-4B-it_pred_pipeline.csv")

BATCH_SIZE = 64
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.0
TOP_P = 1.0
DO_SAMPLE = False
SAVE_INTERVAL = 200

# ---------- PROMPT / SYSTEM ----------
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
  "tokens": [...],
  "compounds": [
    {
      "span": [start_token_index, end_token_index],
      "label": "<Samasa_type>"
    }
  ]
}

Rules:
- Tokenize by meaningful Sanskrit units.
- span = inclusive of start index, exclusive of end index.
- If multiple nested samāsa exist, include all.
- If none, return empty lists for compounds.
- Do not output anything outside JSON.


Example 1:
Input: ससर्षपंतुम्बुरुधान्यवन्यंचण्डांचचूर्णानिसमानिकुर्यात्DUMMY
Output:
{'tokens': ['स', 'सर्षपं', 'तुम्बुरु', 'धान्य', 'वन्यं', 'चण्डां', 'च', 'चूर्णानि', 'समानि', 'कुर्यात्', 'DUMMY'], 
'compounds': [
    {'span': ['1', '2'], 'label': 'Bahuvrihi'}, 
    {'span': ['2', '11'], 'label': 'Comp_root'}, 
    {'span': ['3', '5'], 'label': 'Dvandva'}, 
    {'span': ['4', '5'], 'label': 'Dvandva'}, 
    {'span': ['5', '11'], 'label': 'Comp_root'}, 
    {'span': ['6', '11'], 'label': 'No_rel'}, 
    {'span': ['7', '11'], 'label': 'No_rel'}, 
    {'span': ['8', '11'], 'label': 'No_rel'}, 
    {'span': ['9', '11'], 'label': 'No_rel'}, 
    {'span': ['10', '11'], 'label': 'No_rel'}, 
    {'span': ['11', '0'], 'label': 'root'}]}
   
Example 2:
Input: आपाततसामान्याइवप्रतीयमानाएतेयदिसूक्ष्मम्निरीक्ष्येरन्तर्हिएतेषाम्हृत्अन्तस्थसंकटबोधDUMMY
Output:
{'tokens': ['आपातत', 'सामान्या', 'इव', 'प्रतीयमाना', 'एते', 'यदि', 'सूक्ष्मम्', 'निरीक्ष्येरन्', 'तर्हि', 'एतेषाम्', 'हृत्', 'अन्त', 'स्थ', 'संकट', 'बोध', 'DUMMY'], 
'compounds': [
    {'span': ['1', '16'], 'label': 'No_rel'}, 
    {'span': ['2', '16'], 'label': 'No_rel'}, 
    {'span': ['3', '16'], 'label': 'No_rel'}, 
    {'span': ['4', '16'], 'label': 'No_rel'}, 
    {'span': ['5', '16'], 'label': 'No_rel'}, 
    {'span': ['6', '16'], 'label': 'No_rel'}, 
    {'span': ['7', '16'], 'label': 'No_rel'}, 
    {'span': ['8', '16'], 'label': 'No_rel'}, 
    {'span': ['9', '16'], 'label': 'No_rel'}, 
    {'span': ['10', '16'], 'label': 'No_rel'}, 
    {'span': ['11', '12'], 'label': 'Tatpurusha'}, 
    {'span': ['12', '13'], 'label': 'Tatpurusha'}, 
    {'span': ['13', '16'], 'label': 'Comp_root'}, 
    {'span': ['14', '15'], 'label': 'Tatpurusha'}, 
    {'span': ['15', '16'], 'label': 'Comp_root'}, 
    {'span': ['16', '0'], 'label': 'root'}]}
    
Input: सःचनविशिष्टवैशिष्ट्यअवगाहीDUMMY
Output:
{'tokens': ['सः', 'च', 'न', 'विशिष्ट', 'वैशिष्ट्य', 'अवगाही', 'DUMMY'], 
'compounds': [
    {'span': ['1', '7'], 'label': 'No_rel'}, 
    {'span': ['2', '7'], 'label': 'No_rel'}, 
    {'span': ['3', '7'], 'label': 'No_rel'}, 
    {'span': ['4', '5'], 'label': 'Tatpurusha'}, 
    {'span': ['5', '6'], 'label': 'Tatpurusha'}, 
    {'span': ['6', '7'], 'label': 'Comp_root'}, 
    {'span': ['7', '0'], 'label': 'root'}]}
"""

TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system}<|eot_id|>
<|start_header_id|>user<|end_header_id|>Now analyze:{INPUT}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>Answer:"""

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
    prompt = TEMPLATE.format(system=SYSTEM, INPUT=sentence)
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
