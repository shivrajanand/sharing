import re
import pandas as pd

# ================================================================
#                Relation Extraction for LSS/USS
# ================================================================

def lines_to_relations(true_lines):
    relations_list = []
    relations_oneline = []
    for line in true_lines:
        if line == '\n':
            relations_oneline = [relation for relation in relations_oneline if (relation[2] != 'No_rel' and relation[2] != 'root')]
            relations_list.append(relations_oneline)
            relations_oneline = []
        else:
            lst = line.strip().split('\t')
            relation = [lst[0], lst[4], lst[5]]
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


# ================================================================
#                      USS METRIC
# ================================================================

def unlabeled_metric(true_lines, pred_lines):
    correct = 0
    true_count = 0
    pred_count = 0
    for i in range(len(true_lines)):
        if true_lines[i] != '\n':
            true_lst = true_lines[i].split('\t')
            pred_lst = pred_lines[i].split('\t')
            if true_lst[5] != 'No_rel':
                true_span = [true_lst[0], true_lst[4]]
                pred_span = [pred_lst[0], pred_lst[4]]
                if true_span == pred_span:
                    correct += 1
                true_count += 1
                pred_count += 1
    p = correct / pred_count if pred_count != 0 else 0
    r = correct / true_count if true_count != 0 else 0
    f1 = 2 * p * r / (p + r) if p != 0 and r != 0 else 0
    return round(100 * f1, 2)


# ================================================================
#                      LSS + EXACT MATCH METRIC
# ================================================================

def metric(true_lines, pred_lines):
    true_relations = lines_to_relations(true_lines)
    pred_relations = lines_to_relations(pred_lines)

    correct = 0
    predict_count = 0
    true_count = 0
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

    a = 1.0 * correct / (predict_count + true_count - correct)
    em_per = em / tot_comps if tot_comps > 0 else 0

    metrics_list = [100 * p, 100 * r, 100 * f1, 100 * a, 100 * em_per]
    metrics_list = [round(i, 2) for i in metrics_list]

    return metrics_list


# ================================================================
#                  RAW→CLEAN + NeCTIS Conversion
# ================================================================

def raw_to_clean(line):
    line = re.sub('( {1,})', ' ', line)
    line = re.sub('-$', '', line)
    line = re.sub('<', '', line)
    line = re.sub('-', ' ', line)
    line = re.sub('(>\\w+)', '', line)
    line = re.sub(' $', '', line)
    line = re.sub('^ ', '', line)
    line = re.sub('( {1,})', ' ', line)
    return line


def Conversion(infile, outfile):
    df = pd.read_csv(infile)
    with open(infile) as f:
        raw_lines = f.readlines()
        lines = [raw_to_clean(line) for line in raw_lines]

    with open(outfile, 'w') as w:
        for k in range(df.shape[0]):
            raw_line = raw_lines[k].strip()
            line = lines[k]

            comps_and_words = raw_line.strip().split()
            clean_tokens = line.strip().split()

            outmost_comp_list = [token for token in comps_and_words if '<' in token]

            c = 0
            d = 0
            i = 0

            while i < len(clean_tokens):
                if c < len(outmost_comp_list):
                    outcomp = outmost_comp_list[c]
                else:
                    outcomp = ""

                clean_outcomp = raw_to_clean(outcomp)
                token = comps_and_words[d]
                word = clean_tokens[i]

                if word == token:
                    w.write(f"{i+1}\t{word}\tCompNo\t_\t{len(clean_tokens)+1}\tNo_rel\n")
                    i += 1
                    d += 1

                else:
                    rem_string = outcomp
                    comp_len = len(clean_outcomp.split())

                    for p in range(comp_len):
                        subword = clean_outcomp.split()[p]
                        if p == comp_len - 1:
                            w.write(f"{i+1}\t{subword}\tComp{comp_len}\t_\t{len(clean_tokens)+1}\tComp_root\n")
                            i += 1
                        else:
                            subword_end = re.search('[^>]' + subword, rem_string).end()
                            rem_string = rem_string[subword_end:]

                            n = 0
                            ind = 0
                            p_ = 0

                            if rem_string[0] == '>':
                                flag = 1
                            else:
                                flag = 0

                            for ind in range(len(rem_string)):
                                if rem_string[ind] == '<':
                                    n -= 1
                                elif rem_string[ind] == '>':
                                    n += 1
                                elif rem_string[ind] == '-':
                                    p_ += 1

                                if n - flag == 0:
                                    p_ = 0

                                if rem_string[0] == '>':
                                    if rem_string.count('<') == 0 or len(re.findall('^>\\w+-\\w+', rem_string)) > 0:
                                        st = ind + re.search('-\\w+>', rem_string).end() - 1
                                        temp = rem_string[st:]
                                        tag = re.findall('>(\\w+)', temp)[0]
                                        break
                                    else:
                                        if len(re.findall('^>\\w+>\\w+', rem_string)) == 0:
                                            if n - flag == 1 and p_ >= 0:
                                                st = ind
                                                temp = rem_string[st:]
                                                tag = re.findall('>(\\w+)', temp)[0]
                                                break
                                        else:
                                            if len(re.findall('-\\w+(?:>\\w+)+>(\\w+)', rem_string)) > 0:
                                                st = re.search('-\\w+(?:>\\w+)+>(\\w+)', rem_string).end() - 1
                                                tag = re.findall('-\\w+(?:>\\w+)+>(\\w+)', rem_string)[0]
                                            else:
                                                g = re.findall('^(?:>\\w+)*>\\w+', rem_string)[0].count('>')
                                                if n - flag - g == 1:
                                                    st = ind
                                                    temp = rem_string[st:]
                                                    tag = re.findall('>(\\w+)', temp)[0]
                                            break
                                else:
                                    if n - flag == 1:
                                        st = ind
                                        temp = rem_string[st:]
                                        tag = re.findall('>(\\w+)', temp)[0]
                                        break

                            pre_tag_string = rem_string[:st]
                            num = len(raw_to_clean(pre_tag_string).split())
                            w.write(f"{i+1}\t{subword}\tComp{comp_len}\t_\t{i+num+1}\t{tag}\n")
                            i += 1

                    c += 1
                    d += 1

            w.write(f"{i+1}\tDUMMY\tCompNo\t_\t0\troot\n")
            w.write("\n")

    return


# ================================================================
#                        MAIN EVALUATION
# ================================================================

def evaluate(gold_file_raw, pred_file, out_metrics="compound_metrics.txt"):
    gold_nectis = "gold_nectis_format.txt"

    # convert to NeCTIS
    Conversion(gold_file_raw, gold_nectis)

    with open(gold_nectis) as t, open(pred_file) as p:
        true_lines = t.readlines()
        pred_lines = p.readlines()

    # evaluate
    uss_score = unlabeled_metric(true_lines, pred_lines)
    lss_metrics = metric(true_lines, pred_lines)

    lss_score = lss_metrics[2]
    exact_match = lss_metrics[4]

    print("===== COMPOUND METRICS =====")
    print("USS:", uss_score)
    print("LSS:", lss_score)
    print("Exact Match:", exact_match)

    with open(out_metrics, "w") as f:
        f.write("===== COMPOUND METRICS =====\n")
        f.write(f"USS: {uss_score}\n")
        f.write(f"LSS: {lss_score}\n")
        f.write(f"Exact Match: {exact_match}\n")

    return {
        "uss": uss_score,
        "lss": lss_score,
        "exact_match": exact_match
    }


# ================================================================
#                        SCRIPT ENTRY POINT
# ================================================================

if __name__ == "__main__":
    GOLD = "gold_input.txt"
    PRED = "pred_file.txt"
    evaluate(GOLD, PRED)
