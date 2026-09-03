## CuraSage: An Agentic AI Engine using Consensus Mechanism for Clinical Pathways Prediction

## Reproducibility

---

## Environment
Hardware used for all reported experiments: NVIDIA RTX A6000

---

## Data
MIMIC-III reference files: D_ICD_DIAGNOSES.csv, D_ICD_PROCEDURES.csv,
D_ITEMS.csv, D_LABITEMS.csv (code-to-text mapping); ADMISSIONS.csv,
ICUSTAYS.csv (ICU-stay ID recovery)

Test cohort: 2,000 patients, held out at unique patient level. (seed=42)

Dataset link: https://www.kaggle.com/datasets/healthcarecg/mimic-iii-10k

---

## Python Package Requirements
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


---

## Models
- **Semantic / patient-profiling embedding**: `emilyalsentzer/Bio_ClinicalBERT`
- **LLM reasoning / validation**: `BioMistral/BioMistral-7B`
  - Precision: float16 (GPU) / float32 (CPU); 8-bit fallback via `BitsAndBytesConfig(load_in_8bit=True)`
- **Evaluation-only embedding**: `all-MiniLM-L6-v2`

---

## Seeds
`RANDOM_SEED = 42` (UMAP fitting, BERTopic hyperparameter grid search)

Train/test split seed: `[42]`

---

## Consensus Weights
| Component | Weight |
|-----------|--------|
| Markov | 0.50 |
| Semantic | 0.30 |
| LLM | 0.10 |
| Guideline | 0.10 |

---

## Configuration Hyperparameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| Min. support threshold | 5 | Minimum patient transitions to include an edge |
| Max. pathway steps | 20 | Maximum steps in a generated pathway |
| Min. steps before terminal | 2 | Guard against premature discharge |
| Max. repetitions per topic | 1 | Loop prevention: max times any topic may recur |
| Repetition window | 4 | Recency window for repetition check |
| Semantic repetition threshold | 0.85 | Jaccard stem-overlap for duplicate detection |
| Vital boost multiplier | 2.5 | Score amplification for vital-matched categories |
| Vital block multiplier | 0.0 | Score suppression for vital-excluded categories |
| Domain boost multiplier | 1.8 | Affinity boost for patient's core domain |
| Topic coherence threshold | 0.35 | Minimum keyword coherence to enter candidate pool |
| LLM reranking top-N | 8 | Candidates submitted to BioMistral reranking |
| Max. alternatives | 3 | Number of alternative pathways generated |
| Max. alternative depth | 6 | Maximum steps in each alternative sub-pathway |
| Min. acceptable confidence | 0.15 | Minimum composite score to include a candidate |

---

## BERTopic / UMAP / HDBSCAN
Selected via grid search (c_v coherence), 15% patient subset, seed = 42.

**Chosen parameters for final model:**
- `n_neighbors`: 15
- `n_components`: 10
- `min_dist`: 0.1
- `min_cluster_size`: 15
- `min_samples`: 10
- `ngram`: (1, 2)

**Fixed:** UMAP `metric="cosine"`, `random_state=42`; HDBSCAN `prediction_data=True`

**Final selected values:** see `best_params.json`

---

## LLM Prompts (BioMistral-7B)

**Shared generation config:** `do_sample=True`, `temperature=0.1`, `top_p=0.9`, `repetition_penalty=1.1`

```
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
```


# CuraSage — Core Algorithm Snippets

The snippets below are the core algorithmic components referenced in the paper,
provided. 

---

## 1. Weighted Consensus Scoring (Section 5.3, Eq. 2)

Configuration weights:

```python
class Config:
    def __init__(self, db_path: str, rules_path: str, guidelines_path: str):
        self.min_support_threshold   = 5

        self.markov_weight           = 0.5
        self.semantic_weight         = 0.3
        self.llm_validation_weight   = 0.1
        self.guideline_weight        = 0.1
```

Core scoring loop (per candidate topic):

```python
markov_score   = self._get_markov_score(current_topic, topic_id)
semantic_score = self._get_semantic_score(current_sequence, topic_id)

guideline_boost = self.guideline_integrator.get_guideline_boost(
    topic_id, patient_data or {})

llm_score = 0.5
if (self.config.use_llm_validation and self.llm_validator
        and patient_data and len(scored) < 3):
    source_name = self.db.get_topic_name(current_topic)
    target_name = self.db.get_topic_name(topic_id)
    llm_score, _ = self.llm_validator.validate_transition(
        source_name, target_name,
        patient_data.get('age', 50),
        patient_data.get('gender', 'Unknown'))

valid_transition, _ = self.rule_engine.is_valid_transition(
    current_topic, topic_id)
if not valid_transition:
    continue

base_score = (
    markov_score   * self.config.markov_weight +
    semantic_score * self.config.semantic_weight +
    llm_score      * self.config.llm_validation_weight
) * guideline_boost

# Apply vital constraints
cat        = self.db.get_topic_category(topic_id)
base_score = self.vital_engine.apply_to_score(
    base_score, cat, vital_constraints,
    boost_mult=self.config.vital_boost_multiplier,
    block_mult=self.config.vital_block_multiplier)

base_score *= self._get_patient_context_score(topic_id, patient_data)

if base_score >= self.config.min_acceptable_confidence:
    scored.append({'topic_id': topic_id, 'probability': base_score, ...})
```

---

## 2. Markov Transition Matrix (Section 4.2, 5.3.1)

Graph construction from stored transition statistics:

```python
def _build_graph(self):
    transitions = self.db.get_topic_transitions()
    for trans in transitions:
        source = trans['source_topic_id']
        target = trans['target_topic_id']

        self.graph.add_node(source)
        self.graph.add_node(target)

        self.graph.add_edge(
            source, target,
            weight=trans['confidence_score'],
            count=trans['transition_count'],
            avg_time=trans['avg_time_gap_hours'],
        )
```

Transition-probability lookup used in scoring:

```python
def _get_markov_score(self, source: int, target: int) -> float:
    if source in self.graph.graph and target in self.graph.graph[source]:
        return self.graph.graph[source][target]['weight']
    return 0.0
```

(Edges below `min_support_threshold = 5` are excluded upstream when the
transition statistics table is populated.)

---

## 3. BioMistral-7B Single-Pair Validation (Section 5.3.3)

Model loading (with 8-bit fallback):

```python
model_name = "BioMistral/BioMistral-7B"

self.tokenizer = AutoTokenizer.from_pretrained(model_name)
self.model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    low_cpu_mem_usage=True,
)
# On failure, falls back to 8-bit quantization via BitsAndBytesConfig
```

Safety-filter shortcuts and LLM call:

```python
def validate_transition(self, source_topic, target_topic,
                         patient_age, patient_gender):
    target_lower = target_topic.lower()

    if patient_age > 18:
        for term in ['neonate', 'infant', 'newborn', 'baby', 'pediatric']:
            if term in target_lower:
                return 0.1, "neonatal topic for adult"
    if patient_age < 40:
        for term in ['elderly', 'geriatric', 'dementia']:
            if term in target_lower:
                return 0.1, "geriatric topic for young"
    if 'discharge' in target_lower:
        return 0.9, "discharge is appropriate"

    prompt = (
        f"<s>[INST] You are a clinical expert. Determine if this medical "
        f"transition makes clinical sense.\n\n"
        f"Patient: {patient_age} year old {patient_gender}\n"
        f"Current state: {source_topic}\nNext step: {target_topic}\n\n"
        f"Answer with a single number 0-1 (0=wrong, 1=appropriate).\n"
        f"Score: [/INST]"
    )

    outputs = self.generator(prompt, max_new_tokens=20,
                              pad_token_id=self.tokenizer.pad_token_id)
    generated = outputs[0]['generated_text'].split('[/INST]')[-1].strip()
    numbers = re.findall(r'0\.\d+|\b[01]\b', generated)
    score = float(min(max(float(numbers[0]), 0), 1)) if numbers else 0.5
    return score, "BioMistral validated"
```

---

## 4. Topic Modeling Configuration (Section 4.1, BERTopic/UMAP/HDBSCAN)

```python
def build_topic_model(params):
    vectorizer = CountVectorizer(
        stop_words=list(STOPWORDS),
        ngram_range=params["ngram"],
        token_pattern=r"(?u)\b[a-z]*[a-z][a-z]{2,}\b",
    )

    umap_model = UMAP(
        n_neighbors=params["n_neighbors"],
        n_components=params["n_components"],
        min_dist=params["min_dist"],
        metric="cosine",
        random_state=RANDOM_SEED,
    )

    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=params["min_cluster_size"],
        min_samples=params["min_samples"],
        prediction_data=True,
    )

    return BERTopic(
        embedding_model=EMBED_MODEL,          # emilyalsentzer/Bio_ClinicalBERT
        vectorizer_model=vectorizer,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        language="english",
    )
```

Best hyperparameters are selected by grid search over `c_v` topic
coherence on a subset of patients (`SUBSET_PATIENT_RATIO = 0.15`,
`RANDOM_SEED = 42`) and stored in `best_params.json`.

---

## 5. Data Preprocessing Example (Section 3.2.1)

Representative table-specific cleaning function (diagnosis codes):

```python
def preprocess_DIAGNOSES_ICD_SORTED(df: pd.DataFrame) -> pd.DataFrame:
    mapping = pd.read_csv("data/raw/D_ICD_DIAGNOSES.csv")

    df = df.dropna(subset=['ICD9_CODE'])

    mapping_dict = dict(zip(mapping['ICD9_CODE'].astype(str),
                             mapping['LONG_TITLE']))
    df['ICD9_CODE'] = df['ICD9_CODE'].astype(str)
    df['ICD9_DESCRIPTION'] = df['ICD9_CODE'].map(mapping_dict)

    df_grouped = (
        df.groupby(['SUBJECT_ID', 'HADM_ID'])['ICD9_DESCRIPTION']
        .apply(lambda x: ', '.join(sorted(set(x.dropna()))))
        .reset_index()
    )

    return df_grouped.rename(columns={'ICD9_DESCRIPTION': 'ICD9_CODES'})
```

Every other clinical table (procedures, lab events, medications, etc.)
follows the same pattern: validate required columns, map coded fields to
their MIMIC-III reference descriptions, and group by patient/admission.

---
