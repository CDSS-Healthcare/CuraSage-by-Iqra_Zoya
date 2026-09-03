Explore CuraSage as an agentic frontier that uses consensus mechanism built upon temporal topic modeling to predict clinical pathways for patients
REPRODUCIBILITY
CuraSage: An Agentic AI Engine using Consensus Mechanism for Clinical Pathways Prediction

================================================================================
ENVIRONMENT
================================================================================
Hardware used for all reported experiments: NVIDIA RTX A6000 
================================================================================
DATA
================================================================================
MIMIC-III reference files: D_ICD_DIAGNOSES.csv, D_ICD_PROCEDURES.csv,
D_ITEMS.csv, D_LABITEMS.csv (code-to-text mapping); ADMISSIONS.csv,
ICUSTAYS.csv (ICU-stay ID recovery)
Test cohort: 2,000 patients, held out at unique patient level. (seed=42)
Dataset link: https://www.kaggle.com/datasets/healthcarecg/mimic-iii-10k
================================================================================

Python package requirements

    numpy>=1.23
    pandas>=1.5
    scipy>=1.10
    torch>=2.0
    sentence-transformers>=2.2
    transformers>4.57
   bitsandbytes>=0.42
    bertopic>=0.16
    umap-learn>=0.5
    hdbscan>=0.8
    scikit-learn>=1.2
    gensim>=4.3
    fastdtw>=0.3.4
    langchain>=0.1
    matplotlib>=3.7

================================================================================
MODELS
================================================================================
Semantic / patient-profiling embedding: emilyalsentzer/Bio_ClinicalBERT
LLM reasoning / validation:             BioMistral/BioMistral-7B
    Precision: float16 (GPU) / float32 (CPU); 8-bit fallback via
    BitsAndBytesConfig(load_in_8bit=True)
Evaluation-only embedding (metrics computation, not used at inference):
    all-MiniLM-L6-v2

================================================================================
SEEDS
================================================================================
RANDOM_SEED = 42   (UMAP fitting, BERTopic hyperparameter grid search)
Train/test split seed: [42]

================================================================================
CONSENSUS WEIGHTS
================================================================================
Markov = 0.50   Semantic = 0.30   LLM = 0.10   Guideline = 0.10
================================================================================
TABLE 4: CURASAGE CONFIGURATION HYPERPARAMETERS
================================================================================
Parameter                                   Value   Description
Min. support threshold         5       Minimum patient transitions to include an edge
Max. pathway steps                20      Maximum steps in a generated pathway
Min. steps before terminal   2       Guard against premature discharge
Max. repetitions per topic     1       Loop prevention: max times any topic may recur
Repetition window                   4       Recency window for repetition check
Semantic repetition threshold  0.85    Jaccard stem-overlap for duplicate detection
Vital boost multiplier              2.5     Score amplification for vital-matched categories
Vital block multiplier              0.0     Score suppression for vital-excluded categories
Domain boost multiplier      1.8     Affinity boost for patient's core domain
Topic coherence threshold  0.35    Minimum keyword coherence to enter candidate pool
LLM reranking top-N                8       Candidates submitted to BioMistral reranking
Max. alternatives                       3       Number of alternative pathways generated
Max. alternative depth            6       Maximum steps in each alternative sub-pathway
Min. acceptable confidence     0.15    Minimum composite score to include a candidate
================================================================================
BERTOPIC / UMAP / HDBSCAN
================================================================================
Selected via grid search (c_v coherence), 15% patient subset, seed = 42.

Chosen parameters for final model:
n_neighbors         : 15
n_components        : 10
min_dist            : 0.1
min_cluster_size    : 15
min_samples         : 10
ngram               : (1, 2)
Fixed: UMAP metric="cosine", random_state=42; HDBSCAN prediction_data=True
Final selected values: see best_params.json  [ADD final values]

================================================================================
LLM PROMPTS (BioMistral-7B)
================================================================================
Shared generation config: do_sample=True, temperature=0.1, top_p=0.9,
repetition_penalty=1.1

<s>[INST] You are a clinical expert. A {age} year old {gender} with chief
complaint '{chief_complaint}' is currently at clinical stage:
"{current_topic_name}".

Rank these next possible clinical steps from most to least appropriate
for this patient:
1. {candidate_1}
2. {candidate_2}
...

Consider: clinical logic, standard care pathway, patient's age and chief
complaint.
Return ONLY a JSON array of numbers in order of preference, e.g. [3,1,5,2].
No explanation.
Ranking: [/INST]

================================================================================






