"""
Costruisce un bio_lexicon robusto dai file di output dell'analisi
reverse-embedding

model:

    BioBERT       dmis-lab/biobert-v1.1
    BERT-cased    bert-base-cased
    BERT-uncased  bert-base-uncased
    ClinicalBERT  medicalai/ClinicalBERT
    RoBERTa       roberta-base
    RoBERTa-unc.  roberta-base-uncased

dataset:
    AnaTeM: 100 token
    JNLPBA: 80 token
    BC5CDR: 50 token

uso:
    python build_bio_lexicon.py --input file.txt [file2.txt ...]
"""

import re
import json
import argparse
from pathlib import Path
from collections import defaultdict

# AnaTeM: anatomia microscopica e macroscopica
# BC5CDR: malattie, farmaci, sostanze chimiche
# JNLPBA: geni, proteine, RNA, tipi cellulari


SEED_TERMS: set[str] = {
    # AnaTeM: strutture anatomiche
    "cell", "cells", "tissue", "tissues", "membrane", "membranes",
    "organ", "organs", "nucleus", "nuclei", "cytoplasm", "axon", "axons",
    "dendrite", "dendrites", "neuron", "neurons", "cortex", "striatum",
    "hippocampus", "cerebellum", "cerebrum", "thalamus", "hypothalamus",
    "amygdala", "retina", "lens", "cornea", "cochlea", "kidney", "kidneys",
    "liver", "lung", "lungs", "heart", "spleen", "thymus", "pancreas",
    "intestine", "colon", "stomach", "esophagus", "trachea", "artery",
    "arteries", "vein", "veins", "capillary", "vessel", "vessels", "lymph",
    "bone", "bones", "cartilage", "tendon", "ligament", "muscle", "muscles",
    "epithelium", "epithelial", "endothelium", "endothelial", "stroma",
    "parenchyma", "lumen", "alveoli", "alveolus", "nephron", "glomerulus",
    "tubule", "tubules", "hepatocyte", "hepatocytes", "astrocyte",
    "astrocytes", "microglia", "glia", "glial", "fibroblast", "fibroblasts",
    "macrophage", "macrophages", "lymphocyte", "lymphocytes", "erythrocyte",
    "erythrocytes", "platelet", "platelets", "mitochondria", "mitochondrial",
    "ribosome", "ribosomes", "golgi", "lysosome", "lysosomes", "peroxisome",
    "basal", "apical", "dorsal", "ventral", "proximal", "distal",
    "cerebral", "spinal", "renal", "hepatic", "cardiac", "pulmonary",
    "vascular", "muscular", "neural", "cortical", "subcortical",
    "adipose", "connective", "glandular", "synovial", "peritoneal",
    "pleural", "pericardial", "meninges", "dura", "arachnoid", "pia",
    "myelin", "synapse", "synaptic", "dendritic", "neuronal", "glomerular",
    "tubular", "interstitial", "stromal", "luminal", "abluminal",
    "submucosal", "mucosal", "serosal", "adventitial",

    # BC5CDR: malattie
    "disease", "diseases", "disorder", "disorders", "syndrome", "syndromes",
    "cancer", "cancers", "tumor", "tumors", "tumour", "tumours",
    "carcinoma", "adenoma", "sarcoma", "lymphoma", "leukemia", "melanoma",
    "glioma", "mesothelioma", "neoplasm", "neoplasia", "malignancy",
    "infection", "infections", "inflammation", "inflammatory",
    "diabetes", "hypertension", "obesity", "asthma", "epilepsy",
    "alzheimer", "parkinson", "schizophrenia", "depression", "autism",
    "hepatitis", "cirrhosis", "fibrosis", "sclerosis", "dystrophy",
    "neuropathy", "myopathy", "arthritis", "osteoporosis", "anemia",
    "thrombosis", "embolism", "ischemia", "infarction", "stroke", "sepsis",
    "pneumonia", "tuberculosis", "malaria", "hiv", "aids", "covid",
    "lesion", "lesions", "wound", "injury", "injuries", "trauma",
    "hemorrhage", "edema", "necrosis", "apoptosis", "atrophy", "hypertrophy",
    "metastasis", "metastases", "toxicity", "overdose", "adverse",
    "symptom", "symptoms", "diagnosis", "prognosis", "etiology", "pathology",
    "pathological", "pathogenic", "pathogen", "pathogens",
    "acute", "chronic", "benign", "malignant", "idiopathic",
    "congenital", "hereditary", "autoimmune", "inflammatory",

    # BC5CDR: farmaci e sostanze chimiche
    "drug", "drugs", "compound", "compounds", "chemical", "chemicals",
    "molecule", "molecules", "ligand", "substrate", "metabolite",
    "enzyme", "enzymes", "receptor", "receptors", "inhibitor", "inhibitors",
    "agonist", "antagonist", "antibody", "antibodies", "antigen", "antigens",
    "vaccine", "vaccines", "hormone", "hormones", "insulin", "glucagon",
    "adrenaline", "dopamine", "serotonin", "acetylcholine", "cortisol",
    "steroid", "steroids", "lipid", "lipids", "fatty", "acid", "acids",
    "amino", "peptide", "peptides", "nucleotide", "nucleotides",
    "glucose", "lactate", "pyruvate", "calcium", "sodium", "potassium",
    "chloride", "iron", "zinc", "copper", "magnesium",
    "toxin", "toxins", "carcinogen", "mutagen", "allergen",
    "antibiotic", "antibiotics", "antiviral", "antifungal",
    "chemotherapy", "immunotherapy", "radiotherapy",
    "therapy", "therapies", "treatment", "treatments",
    "dose", "dosage", "concentration", "plasma", "serum",
    "morphine", "cisplatin", "doxorubicin", "tamoxifen", "metformin",
    "warfarin", "heparin", "aspirin", "ibuprofen", "paracetamol",

    # JNLPBA: geni, proteine, molecole
    "gene", "genes", "protein", "proteins", "mrna", "rna", "dna",
    "cdna", "rrna", "trna", "mirna", "sirna", "lncrna", "snrna",
    "genome", "genomic", "transcriptome", "proteome", "exome",
    "chromosome", "chromosomal", "locus", "allele", "alleles",
    "mutation", "mutations", "variant", "variants", "snp",
    "expression", "transcription", "translation", "replication",
    "promoter", "enhancer", "codon", "exon", "intron", "splicing",
    "isoform", "homolog", "ortholog", "paralog",
    "kinase", "phosphatase", "protease", "transferase", "synthase",
    "oxidase", "reductase", "ligase", "polymerase", "helicase",
    "signaling", "pathway", "cascade", "binding", "affinity",
    "domain", "motif", "overexpression", "downregulation", "upregulation",
    "knockout", "knockdown", "transfection", "transgenic", "recombinant",
    "sequencing", "pcr", "elisa", "blot", "assay", "assays",

    # JNLPBA: tipi cellulari
    "monocyte", "monocytes", "neutrophil", "neutrophils",
    "eosinophil", "basophil", "dendritic", "thymocyte", "thymocytes",
    "keratinocyte", "osteoblast", "osteoclast", "chondrocyte",
    "adipocyte", "myoblast", "cardiomyocyte", "mesenchymal",
    "stem", "progenitor", "differentiation", "proliferation",
    "migration", "invasion", "adhesion",

    

    "clinical", "medical", "biomarker", "biomarkers",
    "patient", "patients", "cohort", "biopsy", "sample", "samples",
    "activity", "function", "mechanism", "mechanisms",
    "regulation", "inhibition", "activation", "suppression", "induction",
    "fluorescent", "staining", "immunostaining", "histological",
    "molecular", "cellular", "genetic", "epigenetic",
    "phenotype", "genotype", "phenotypic",
    "surgery", "surgical", "invasive", "diagnosis", "diagnostic",
    "viral", "bacterial", "microbial", "fungal",
    "blood", "urine", "serum", "plasma", "tissue",
}

SEED_TERMS = {t.lower() for t in SEED_TERMS}

# SUBWORD PATTERNS per RoBERTa

SUBWORD_BIO_FRAGMENTS: list[str] = [
    "ocyte", "ocytes", "ocyte", "helial", "ascular", "betes", "thritis",
    "ritis", "pathy", "ology", "itis", "emia", "osis", "osis", "tomy",
    "plasty", "scopy", "ectomy", "otomy",
    "rotein", "enome", "ranscr", "eceptor", "kinase", "phosph", "glyco",
    "nucleo", "peptid", "lipid", "steroid", "hormon", "cytok", "antib",
    "immun", "tumor", "carcin", "oncol", "leuk", "lymph", "hepat",
    "cardio", "neuro", "myelo", "fibro", "osteo",
    "abetes", "zyme", "amin", "acil", "mycin", "cillin", "oxacin",
    "statin", "sartan", "pril", "olol",
    "bral", "renal", "patic", "lmonary", "estinal", "ascular",
    "rtical", "ellum", "thalm", "glomer", "tubul",
]


def _is_bio_subword(token: str) -> bool:
    """Verifica se un frammento BPE di RoBERTa è biomedico."""
    t = token.lower().lstrip("Ġ▁")
    if t in SEED_TERMS:
        return True
    return any(frag in t for frag in SUBWORD_BIO_FRAGMENTS)


def _is_bio_wholeword(token: str) -> bool:
    """Verifica se un token whole-word (BERT/BioBERT/ClinicalBERT) è biomedico."""
    t = token.lower()
    if t in SEED_TERMS:
        return True
    if len(t) >= 5:
        for seed in SEED_TERMS:
            if len(seed) < 5:
                continue
            if abs(len(seed) - len(t)) <= 3:
                if t.startswith(seed) or seed.startswith(t):
                    return True
    return False

BACKBONE_FAMILIES = {
    "roberta": "roberta",         # roberta-base, roberta-base-uncased
    "biobert": "bert_bio",        # dmis-lab/biobert-v1.1
    "clinicalbert": "bert_clinical",  # medicalai/ClinicalBERT
    "bert-base": "bert_base",     # bert-base-cased, bert-base-uncased
}

def detect_family(model_name: str) -> str:
    m = model_name.lower()
    for key, family in BACKBONE_FAMILIES.items():
        if key in m:
            return family
    return "bert_base"  # fallback

def parse_output_file(filepath: str) -> list[dict]:
    """
    Legge un file che può contenere più blocchi CONFIGURAZIONE ESPERIMENTO.
    Restituisce lista di esperimenti, ciascuno con:
      - metadata: {model, dataset, prefix, lr, family}
      - virtual_token_data: [{vt_idx, neighbors: [{text, score, was_bio}]}]
    """
    text = Path(filepath).read_text(encoding="utf-8", errors="replace")
    # Splitta sui blocchi di configurazione
    blocks = re.split(r"(?=CONFIGURAZIONE ESPERIMENTO)", text)
    blocks = [b.strip() for b in blocks if b.strip()]

    experiments = []
    vt_pattern = re.compile(r"Virtual Token\s+(\d+)\s+\[Bio:\s*([\d.]+)%\]:\s*(.+)")
    neighbor_pattern = re.compile(r"([^\s(|*]+)(\*)?\(\s*([\d.]+)\s*\)")

    for block in blocks:
        meta = {}
        for key, pat in [
            ("model",   r"Modello:\s+(.+)"),
            ("dataset", r"Dataset:\s+(.+)"),
            ("prefix",  r"Prefix:\s+(\d+)"),
            ("lr",      r"LR:\s+(.+)"),
        ]:
            m = re.search(pat, block)
            if m:
                meta[key] = m.group(1).strip()

        if "model" not in meta:
            continue

        meta["family"] = detect_family(meta.get("model", ""))

        vt_data = []
        for line in block.splitlines():
            m = vt_pattern.match(line.strip())
            if not m:
                continue
            vt_idx = int(m.group(1))
            neighbors = []
            for nm in neighbor_pattern.finditer(m.group(3)):
                token_text = nm.group(1).strip()
                was_bio = nm.group(2) == "*"
                score = float(nm.group(3))
                neighbors.append({
                    "text": token_text,
                    "score": score,
                    "was_bio_heuristic": was_bio,
                })
            if neighbors:
                vt_data.append({"vt_idx": vt_idx, "neighbors": neighbors})

        if vt_data:
            experiments.append({"metadata": meta, "virtual_token_data": vt_data})

    return experiments


def build_lexicon(all_experiments: list[dict]) -> set[str]:
    lexicon: set[str] = set()

    # seed diretto
    for exp in all_experiments:
        family = exp["metadata"]["family"]
        # RoBERTa ha token non interi
        if family == "roberta":
            continue
        # ClinicalBERT ha vocabolario multilingue rumoroso
        strict = (family == "bert_clinical")

        for vt in exp["virtual_token_data"]:
            for nb in vt["neighbors"]:
                t = nb["text"].lower()
                if strict:
                    if t in SEED_TERMS:
                        lexicon.add(t)
                else:
                    if _is_bio_wholeword(t):
                        lexicon.add(t)

    cooccurrence: dict[str, int] = defaultdict(int)
    for exp in all_experiments:
        family = exp["metadata"]["family"]
        min_bio_neighbors = 4 if family == "bert_clinical" else 3

        for vt in exp["virtual_token_data"]:
            nbrs = vt["neighbors"]

            if family == "roberta":
                bio_count = sum(1 for nb in nbrs if _is_bio_subword(nb["text"]))
            else:
                bio_count = sum(
                    1 for nb in nbrs if nb["text"].lower() in lexicon
                )

            if bio_count >= min_bio_neighbors:
                for nb in nbrs:
                    t = nb["text"].lower()
                    if (family != "roberta"
                            and t not in lexicon
                            and len(t) >= 5
                            and t.isalpha()):
                        cooccurrence[t] += 1

    for token, count in cooccurrence.items():
        if count >= 2:
            lexicon.add(token)

    # remove falsi positivi
    BLOCKLIST = {
        "instead", "during", "therefore", "using", "within", "between",
        "after", "before", "however", "although", "because", "which",
        "another", "other", "these", "those", "their", "there", "where",
        "finished", "joined", "opened", "changed", "achieved", "required",
        "mariners", "football", "rugby", "tennis", "hearing",
        "bin", "again", "location", "locations",
    }
    lexicon -= BLOCKLIST

    return lexicon

def is_bio_token(token: str, family: str, lexicon: set[str]) -> bool:
    if family == "roberta":
        return _is_bio_subword(token)
    else:
        return token.lower() in lexicon


def compute_das(exp: dict, lexicon: set[str]) -> tuple[float, float]:
    family = exp["metadata"]["family"]
    total_bio, total = 0, 0
    scores = []
    for vt in exp["virtual_token_data"]:
        for nb in vt["neighbors"]:
            if is_bio_token(nb["text"], family, lexicon):
                total_bio += 1
            scores.append(nb["score"])
            total += 1
    das = (total_bio / total * 100) if total else 0.0
    mean_cos = sum(scores) / len(scores) if scores else 0.0
    return das, mean_cos

DATASET_PREFIXES = {
    "anatem":  100,
    "jnlpba":  80,
    "bc5cdr":  50,
}

def print_report(all_experiments: list[dict], lexicon: set[str]) -> None:
    # Raggruppa per dataset
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for exp in all_experiments:
        ds = exp["metadata"].get("dataset", "?").split("/")[-1].lower()
        by_dataset[ds].append(exp)

    for ds in sorted(by_dataset):
        exps = by_dataset[ds]
        prefix = DATASET_PREFIXES.get(ds, "?")
        print(f"\n")
        print(f"  DATASET: {ds.upper()}   |   Prefix: {prefix} token   |   LR: 1e-4")
        print(f"{'═'*72}")
        print(f"  {'Backbone':<30} {'DAS orig':>9} {'DAS new':>9} {'Δ':>7}  {'Cos sim':>8}  {'Family'}")
        print(f"\n")

        for exp in sorted(exps, key=lambda e: e["metadata"].get("model", "")):
            meta = exp["metadata"]
            model_short = meta.get("model", "?").split("/")[-1]

            # DAS originale (dai marcatori * già nel file)
            k = len(exp["virtual_token_data"][0]["neighbors"]) if exp["virtual_token_data"] else 10
            n_vt = len(exp["virtual_token_data"])
            orig_bio = sum(
                sum(1 for nb in vt["neighbors"] if nb["was_bio_heuristic"])
                for vt in exp["virtual_token_data"]
            )
            orig_das = orig_bio / (n_vt * k) * 100

            new_das, mean_cos = compute_das(exp, lexicon)
            delta = new_das - orig_das
            sign = "↑" if delta > 0 else "↓"

            print(f"  {model_short:<30} {orig_das:8.2f}% {new_das:8.2f}% "
                  f"{sign}{abs(delta):5.2f}%  {mean_cos:.4f}   {meta['family']}")

    print("\n")
    print(f"  Lessico: {len(lexicon)} termini whole-word")
    print(f"  RoBERTa usa pattern subword ({len(SUBWORD_BIO_FRAGMENTS)} frammenti)")
    print("\n")

def save_lexicon(lexicon: set[str], out_dir: str = ".") -> tuple[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sorted_lex = sorted(lexicon)
    txt_path  = out / "bio_lexicon.txt"
    json_path = out / "bio_lexicon.json"
    txt_path.write_text("\n".join(sorted_lex) + "\n", encoding="utf-8")
    json_path.write_text(
        json.dumps(sorted_lex, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return str(txt_path), str(json_path)


def load_bio_lexicon(path: str) -> set[str]:
    return {
        line.strip().lower()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


DEFAULT_FILES = [
    "./prompt_tuning_rev/interpret_soft_token_anatem.txt",
    "./prompt_tuning_rev/interpret_soft_token_bc5cdr.txt",
    "./prompt_tuning_rev/interpret_soft_token_jnlpba.txt",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build bio lexicon from reverse-embedding output files")
    parser.add_argument("--input", nargs="+", default=DEFAULT_FILES, metavar="FILE")
    parser.add_argument("--output-dir", default="./outputs")
    args = parser.parse_args()

    # Carica tutti gli esperimenti
    all_experiments: list[dict] = []
    for fp in args.input:
        if not Path(fp).exists():
            print(f"File non trovato: {fp}")
            continue
        exps = parse_output_file(fp)
        print(f"{Path(fp).name}: {len(exps)} esperimenti trovati")
        all_experiments.extend(exps)

    if not all_experiments:
        print(" Nessun esperimento caricato.")
        raise SystemExit(1)

    # Costruzione lessico
    lexicon = build_lexicon(all_experiments)
    txt_path, json_path = save_lexicon(lexicon, args.output_dir)

    print(f"\nLessico costruito: {len(lexicon)} termini")
    print(f"       {txt_path}")
    print(f"       {json_path}")    

    print_report(all_experiments, lexicon)
