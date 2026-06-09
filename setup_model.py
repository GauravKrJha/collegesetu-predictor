

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib, json, os, warnings, time

warnings.filterwarnings("ignore")

CSAB_CSV   = "csab_cutoffs.csv"
OUTPUT_DIR = "model_artifacts"
SAMPLES    = 20
SEED       = 42

np.random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def log(s, t, m):
    print(f"\n{'='*55}\n  [{s}/{t}] {m}\n{'='*55}")


# ── PHASE 1: LOAD & CLEAN ─────────────────────────────────────────────────────
def load_and_clean():
    log(1, 5, "DATA INGESTION & FEATURE ENGINEERING")

    df = pd.read_csv(CSAB_CSV)
    df.columns = df.columns.str.strip()

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    for rc in ["Opening Rank", "Closing Rank"]:
        df[rc] = pd.to_numeric(df[rc], errors="coerce")

    before = len(df)
    df.dropna(subset=["Opening Rank", "Closing Rank"], inplace=True)
    df["Opening Rank"] = df["Opening Rank"].astype(int)
    df["Closing Rank"] = df["Closing Rank"].astype(int)

    # Ensure Institute_Type exists
    if "Institute_Type" not in df.columns:
        df["Institute_Type"] = df["Institute"].apply(lambda n:
            "NIT"  if "National Institute of Technology" in str(n) else
            "IIIT" if ("Indian Institute of Information" in str(n) or "IIIT" in str(n)) else
            "GFTI")

    # Engineered features
    df["Rank_Spread"]            = df["Closing Rank"] - df["Opening Rank"]
    df["Competitiveness_Index"]  = 1 / (1 + np.log1p(df["Closing Rank"]))

    def get_domain(p):
        p = str(p).lower()
        if any(k in p for k in ["computer","software","information","data","ai","artificial"]): return "CS_IT"
        if any(k in p for k in ["electrical","electronics","ece","eee","vlsi"]):                return "EE_ECE"
        if any(k in p for k in ["mechanical","production","industrial","auto"]):                return "Mechanical"
        if any(k in p for k in ["civil","architecture","planning","structural"]):               return "Civil"
        if any(k in p for k in ["chemical","bio","pharma","food","biotech"]):                   return "Chemical_Bio"
        if any(k in p for k in ["mining","metallur","material","ceramic"]):                     return "Materials"
        if any(k in p for k in ["math","physics","science","chemistry"]):                       return "Sciences"
        return "Other"

    df["Program_Domain"] = df["Program"].apply(get_domain)

    print(f"  Loaded {before} rows, kept {len(df)}")
    print(f"  Institutes: {df['Institute'].nunique()} | Programs: {df['Program'].nunique()}")
    print(f"  Institute types: {dict(df['Institute_Type'].value_counts())}")
    return df


# ── PHASE 2: MONTE CARLO SIMULATION ──────────────────────────────────────────
def generate_synthetic(df):
    log(2, 5, "MONTE CARLO SYNTHETIC DATA GENERATION")
    rows = []
    for _, r in df.iterrows():
        closing = r["Closing Rank"]
        spread  = max(int(closing * 0.20), 500)
        for _ in range(SAMPLES):
            noise    = int(np.random.normal(0, spread * 0.5))
            sim_rank = max(1, closing + noise)
            z        = (closing - sim_rank) / max(spread * 0.3, 100)
            prob     = 1 / (1 + np.exp(-z))
            admitted = 1 if np.random.random() < prob else 0
            rows.append({
                "Institute_Type":        r["Institute_Type"],
                "Institute":             r["Institute"],
                "Program":               r["Program"],
                "Quota":                 r["Quota"],
                "Category":              r["Category"],
                "Gender":                r["Gender"],
                "Program_Domain":        r["Program_Domain"],
                "Simulated_Rank":        sim_rank,
                "Rank_Spread":           r["Rank_Spread"],
                "Competitiveness_Index": r["Competitiveness_Index"],
                "Rank_Ratio":            sim_rank / max(closing, 1),
                "Distance_From_Cutoff":  closing - sim_rank,
                "Admitted":              admitted,
            })
    syn = pd.DataFrame(rows)
    print(f"  Generated {len(syn):,} synthetic records from {len(df):,} cutoff rows")
    print(f"  Admitted: {syn['Admitted'].sum():,} | Rejected: {(len(syn)-syn['Admitted'].sum()):,}")
    return syn


# ── PHASE 3: TRAIN RANDOMFOREST ───────────────────────────────────────────────
def train_model(syn):
    log(3, 5, "RANDOMFOREST MODEL TRAINING")
    encoders = {}
    cat_cols = ["Institute_Type","Institute","Program","Quota","Category","Gender","Program_Domain"]
    for col in cat_cols:
        le = LabelEncoder()
        syn[col+"_enc"] = le.fit_transform(syn[col].astype(str))
        encoders[col] = le

    feature_cols = [c+"_enc" for c in cat_cols] + [
        "Simulated_Rank", "Rank_Spread", "Competitiveness_Index",
        "Rank_Ratio", "Distance_From_Cutoff"
    ]
    X = syn[feature_cols].values
    y = syn["Admitted"].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)

    print("  Training RandomForest (250 trees, max_depth=20)...")
    t0 = time.time()
    model = RandomForestClassifier(
        n_estimators=250, max_depth=20, min_samples_split=5,
        class_weight="balanced", random_state=SEED, n_jobs=-1)
    model.fit(X_tr, y_tr)

    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]
    cv      = cross_val_score(model, X, y, cv=5, scoring="accuracy")

    metrics = {
        "accuracy":   round(accuracy_score(y_te, y_pred), 4),
        "f1_score":   round(f1_score(y_te, y_pred, average="weighted"), 4),
        "auc_roc":    round(roc_auc_score(y_te, y_proba), 4),
        "cv_mean":    round(cv.mean(), 4),
        "cv_std":     round(cv.std(), 4),
        "train_time": round(time.time()-t0, 1),
        "total_records": len(syn),
    }
    imps = {feature_cols[i]: round(float(v), 5)
            for i, v in enumerate(model.feature_importances_)}
    imps = dict(sorted(imps.items(), key=lambda x: x[1], reverse=True))

    print(f"\n  === Evaluation ===")
    print(f"  Accuracy:  {metrics['accuracy']}")
    print(f"  F1 Score:  {metrics['f1_score']}")
    print(f"  AUC-ROC:   {metrics['auc_roc']}")
    print(f"  CV (5-fold): {metrics['cv_mean']} ± {metrics['cv_std']}")
    print(f"  Train time: {metrics['train_time']}s")
    return model, encoders, feature_cols, metrics, imps


# ── PHASE 4: NLP INTENT CLASSIFIER ───────────────────────────────────────────
def train_nlp():
    log(4, 5, "NLP INTENT CLASSIFIER TRAINING")
    data = [
        # ── college_recommendation ──
        ("What colleges can I get with 50000 rank",     "college_recommendation"),
        ("Best NITs for my rank",                        "college_recommendation"),
        ("Which institutes can I get",                   "college_recommendation"),
        ("Suggest colleges for rank 30000",              "college_recommendation"),
        ("CSAB colleges for 80000 rank",                 "college_recommendation"),
        ("Which NIT for 60000 rank OBC",                 "college_recommendation"),
        ("Colleges under 20000 rank open category",      "college_recommendation"),
        ("What options do I have at rank 100000",        "college_recommendation"),
        ("Give me a list of good colleges for 75000",    "college_recommendation"),
        ("Which IIIT can I get with 40000 rank",         "college_recommendation"),
        ("recommend me some colleges",                   "college_recommendation"),
        ("where can I get admission with 25k rank",      "college_recommendation"),
        ("best engineering colleges for my rank ews",    "college_recommendation"),
        ("shortlist colleges for me",                    "college_recommendation"),
        ("which gfti is good for 90000 rank",            "college_recommendation"),
        # ── branch_comparison ──
        ("Which is better CSE or ECE",                   "branch_comparison"),
        ("Compare mechanical and civil engineering",     "branch_comparison"),
        ("Should I choose IT or CSE",                    "branch_comparison"),
        ("CSE vs IT which has better placement",         "branch_comparison"),
        ("Difference between ECE and EEE",               "branch_comparison"),
        ("is AI better than data science",               "branch_comparison"),
        ("chemical vs mechanical which to pick",         "branch_comparison"),
        ("compare electrical and electronics",           "branch_comparison"),
        ("CSE at lower NIT or ECE at top NIT",           "branch_comparison"),
        ("which branch is best for future",              "branch_comparison"),
        ("metallurgy vs civil engineering",              "branch_comparison"),
        # ── cutoff_inquiry ──
        ("What is the cutoff for NIT Surathkal CSE",     "cutoff_inquiry"),
        ("Closing rank for NIT Trichy ECE",              "cutoff_inquiry"),
        ("NIT Warangal last year cutoff OPEN",           "cutoff_inquiry"),
        ("CSAB special round cutoff 2025",               "cutoff_inquiry"),
        ("what was the opening rank for IIIT Hyderabad", "cutoff_inquiry"),
        ("show me cutoff of MNNIT Allahabad CSE",        "cutoff_inquiry"),
        ("closing rank for mechanical at NIT Calicut",   "cutoff_inquiry"),
        ("cutoff for SC category at NIT Rourkela",       "cutoff_inquiry"),
        # ── counseling_process ──
        ("How does CSAB special round work",             "counseling_process"),
        ("What is UPTAC counseling",                     "counseling_process"),
        ("How many rounds in CSAB",                      "counseling_process"),
        ("Difference between JOSAA and CSAB",            "counseling_process"),
        ("What is home state quota in NITs",             "counseling_process"),
        ("how does seat allotment work",                 "counseling_process"),
        ("explain the josaa choice filling process",     "counseling_process"),
        ("when does csab registration start",            "counseling_process"),
        ("what is float freeze and slide",               "counseling_process"),
        ("how to do document verification",              "counseling_process"),
        # ── career_guidance ──
        ("What is the scope of ECE",                     "career_guidance"),
        ("Career options after mechanical engineering",  "career_guidance"),
        ("Which branch has best salary",                 "career_guidance"),
        ("Placements at NIT Trichy",                     "career_guidance"),
        ("is there a good future in civil engineering",  "career_guidance"),
        ("job opportunities after chemical engineering", "career_guidance"),
        ("which branch is best for higher studies abroad","career_guidance"),
        ("average package for CSE graduates",            "career_guidance"),
        ("scope of data science in india",              "career_guidance"),
        # ── category_reservation ──
        ("What is OBC NCL reservation",                  "category_reservation"),
        ("How does EWS quota work in NITs",              "category_reservation"),
        ("SC ST reservation percentage",                 "category_reservation"),
        ("am I eligible for ews certificate",            "category_reservation"),
        ("how much relaxation for obc candidates",       "category_reservation"),
        ("what is pwd reservation",                      "category_reservation"),
        ("female supernumerary seats explanation",       "category_reservation"),
        # ── admission_probability ──
        ("What are my chances of getting NIT",           "admission_probability"),
        ("Predict my admission at rank 55000",           "admission_probability"),
        ("Am I safe at 30000 rank for NIT Trichy CSE",   "admission_probability"),
        ("Will I get NIT with 45000 rank OBC",           "admission_probability"),
        ("how likely am I to get IIIT Allahabad",        "admission_probability"),
        ("chances of CSE at NIT with 12000 rank",        "admission_probability"),
        ("is my rank enough for mechanical at NIT",      "admission_probability"),
        ("probability of getting a seat in round 2",     "admission_probability"),
        # ── greeting ──
        ("Hello",    "greeting"),
        ("Hi there", "greeting"),
        ("Help me",  "greeting"),
        ("Good morning", "greeting"),
        ("hey",      "greeting"),
        ("thanks a lot", "greeting"),
        ("who are you", "greeting"),
        ("what can you do", "greeting"),
        # ── general_query (any other doubt) ──
        ("what are the fees at NIT Trichy",              "general_query"),
        ("is hostel compulsory in first year",          "general_query"),
        ("how is campus life at NIT Warangal",          "general_query"),
        ("can I change my branch after first year",     "general_query"),
        ("tell me about NIT Surathkal",                 "general_query"),
        ("is JEE Main tougher than JEE Advanced",       "general_query"),
        ("how should I prepare for counselling",        "general_query"),
        ("what documents do I need for admission",      "general_query"),
        ("is a private college better than a low NIT",  "general_query"),
        ("explain the difference between IIT and NIT",  "general_query"),
        ("what is a good rank in JEE Main",             "general_query"),
        ("should I drop a year to improve my rank",     "general_query"),
    ]
    texts, labels = zip(*data)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000,
                                   stop_words="english", sublinear_tf=True)),
        ("clf",   MultinomialNB(alpha=0.1)),
    ])
    pipe.fit(texts, labels)
    cv = cross_val_score(pipe, texts, labels, cv=3, scoring="accuracy")
    print(f"  Trained on {len(texts)} examples | {len(set(labels))} intents")
    print(f"  CV Accuracy: {cv.mean():.2f} ± {cv.std():.2f}")

    entity_patterns = {
        "rank_pattern": r'\b(\d{3,7})\b',
        "category_keywords": {
            "OPEN":    ["general", "open", "ur", "unreserved"],
            "OBC-NCL": ["obc", "obc-ncl", "other backward"],
            "SC":      ["sc", "scheduled caste"],
            "ST":      ["st", "scheduled tribe"],
            "EWS":     ["ews", "economically weaker"],
        },
        "gender_keywords": {
            "Gender-Neutral": ["male", "boy", "general gender"],
            "Female-only (including Supernumerary)": ["female", "girl", "women", "woman"],
        },
    }
    return pipe, entity_patterns


# ── PHASE 5: SAVE ─────────────────────────────────────────────────────────────
def save_all(model, encoders, fcols, metrics, imps, intent_pipe, entity_patterns):
    log(5, 5, "SAVING ARTIFACTS")
    joblib.dump(model,    os.path.join(OUTPUT_DIR, "model.pkl"))
    joblib.dump(encoders, os.path.join(OUTPUT_DIR, "label_encoders.pkl"))
    joblib.dump({"feature_cols": fcols}, os.path.join(OUTPUT_DIR, "model_metadata.pkl"))
    with open(os.path.join(OUTPUT_DIR, "model_metrics.json"),      "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "feature_importance.json"), "w") as f:
        json.dump(imps, f, indent=2)
    joblib.dump(intent_pipe,     os.path.join(OUTPUT_DIR, "intent_classifier.pkl"))
    joblib.dump(entity_patterns, os.path.join(OUTPUT_DIR, "entity_patterns.pkl"))
    print(f"  All artifacts saved to ./{OUTPUT_DIR}/")
    print(f"\n  ✓ SETUP COMPLETE → python -m streamlit run app.py")


if __name__ == "__main__":
    t0 = time.time()
    df = load_and_clean()
    syn = generate_synthetic(df)
    model, encoders, fcols, metrics, imps = train_model(syn)
    intent_pipe, entity_patterns = train_nlp()
    save_all(model, encoders, fcols, metrics, imps, intent_pipe, entity_patterns)
    print(f"\n  Total time: {time.time()-t0:.0f}s")
