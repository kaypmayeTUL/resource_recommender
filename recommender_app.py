"""
Acquisition Recommendation Scorer — Standalone Streamlit App
============================================================

A self-contained version of the Acquisition Recommendation Scorer, extracted
from the larger Library Collection Dashboard. Score candidate book lists
(vendor slip lists, GOBI picks, approval-plan exceptions, DDA candidates,
faculty requests) against your checkout history to prioritize purchases.

How it scores:
  * Subject similarity — Jaccard overlap of normalized subject terms,
    optionally stemmed and reduced via a synonym map (built-in + user-supplied)
  * LC classification fit — match strength based on shared LC letter prefix
  * Author popularity — count of past checkouts by the same author
  * Faculty research interest — optional, joins candidate subjects against
    faculty interest text via TF-IDF-like overlap

Inputs:
  * Checkouts file (required) — your historical circulation data
  * Recommendations file (required) — what you're considering buying
  * Faculty research interests (optional) — for the interest-match score
  * Custom synonym groups (optional) — to collapse near-equivalents

Output: a scored, sortable recommendations list with per-component breakdowns,
downloadable as CSV with annotation notes.

Run as a standalone app:
    streamlit run recommender_app.py

Dependencies:
    streamlit, pandas, numpy, plotly, nltk
    (NLTK data: punkt, wordnet, omw-1.4 — auto-downloaded on first run)

Contact: Kay P Maye (kmaye@tulane.edu)
Extracted from Library Collection Dashboard v2.3
"""

# =====================================================================
# IMPORTS
# =====================================================================
import re
import csv
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from io import BytesIO, StringIO

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# NLTK is required for this tool; we degrade gracefully if it's missing.
try:
    import nltk
    from nltk.stem import SnowballStemmer
    from nltk.corpus import wordnet
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


# =====================================================================
# PAGE CONFIG & GLOBAL CSS (Tulane palette)
# =====================================================================
st.set_page_config(
    page_title="Acquisition Recommendation Scorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(""""
<style>
:root {
    --tulane-green: #285C4D;
    --tulane-blue: #71C5E8;
}
.main > div { padding-top: 1.5rem; }
.stButton>button {
    background-color: #285C4D;
    color: white;
    font-weight: bold;
    padding: 0.5rem 1rem;
    border-radius: 5px;
    border: none;
    width: 100%;
}
.stButton>button:hover { background-color: #1e4a3c; }
div[data-testid="metric-container"] {
    background-color: #eef6f3;
    border: 1px solid #285C4D;
    padding: 10px;
    border-radius: 5px;
    margin: 5px 0;
}
.decision-box {
    background-color: #eef6f3;
    border-left: 4px solid #285C4D;
    padding: 15px 20px;
    border-radius: 4px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)


# =====================================================================
# LC classification reference (used for display labels)
# =====================================================================

LC_CLASSES = {
    'A': 'General Works', 'B': 'Philosophy, Psychology, Religion',
    'C': 'Auxiliary Sciences of History', 'D': 'World History',
    'E': 'US History', 'F': 'History of the Americas',
    'G': 'Geography, Anthropology, Recreation', 'H': 'Social Sciences',
    'J': 'Political Science', 'K': 'Law', 'L': 'Education',
    'M': 'Music & Books on Music', 'N': 'Fine Arts', 'P': 'Language & Literature',
    'Q': 'Science', 'R': 'Medicine', 'S': 'Agriculture',
    'T': 'Technology', 'U': 'Military Science', 'V': 'Naval Science',
    'Z': 'Bibliography & Library Science'
}

# LC subclass map — main letter → {two-letter subclass code → human label}
# Sourced from the Library of Congress Classification Outline
# (https://www.loc.gov/aba/cataloging/classification/lcco/) and the LC's
# free per-class PDF schedules. Covers all 21 main classes.
#
# Coverage scope: the two-letter (alpha) subclasses only. The numerical
# ranges below those (e.g., HQ 1000–1999) are not represented here because
# the dashboard's matching only inspects the leading letters of a call
# number — a richer breakdown isn't useful unless we change the parser.
#
# A few classes have notable nuances reflected here:

# =====================================================================
# Shared utilities: text normalization & LC parsing
# =====================================================================

_RE_DATE_PAREN = re.compile(r"\s*\([0-9\-]+\)")
_RE_MULTI_SPACE = re.compile(r"\s+")
_RE_DASH_SPACE = re.compile(r"\s*-\s*")

def normalize_text(text):
    """Lowercase → strip accents → clean punctuation → collapse whitespace."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_subject_term(term):
    """Clean and standardize a single subject term."""
    if pd.isna(term) or not isinstance(term, str) or term.strip() == '':
        return None
    s = term.strip().rstrip('.;- ')
    s = _RE_DATE_PAREN.sub('', s)
    s = s.replace('--', ' - ')
    s = _RE_DASH_SPACE.sub(' - ', s)
    s = _RE_MULTI_SPACE.sub(' ', s).strip()
    return s.lower() if s else None


def split_subjects(raw_subjects):
    """Split on ; | , newline and normalize each piece."""
    if pd.isna(raw_subjects) or not isinstance(raw_subjects, str):
        return []
    parts = re.split(r"[;|,\n]", raw_subjects)
    return [normalize_text(p) for p in parts if normalize_text(p)]


def extract_lc_prefix(lc_class):
    """Extract LC letter prefix from a call number string."""
    if pd.isna(lc_class):
        return None
    match = re.match(r"^([A-Z]{1,3})", str(lc_class).strip().upper())
    return match.group(1) if match else None


# =====================================================================
# Shared utilities: session caching across reruns
# =====================================================================

def _make_file_key(uploaded_file):
    """Build a stable cache key from an uploaded file object."""
    if uploaded_file is None:
        return None
    try:
        return (uploaded_file.name, uploaded_file.size)
    except AttributeError:
        # Fallback for file-like objects without .size
        return (uploaded_file.name, None)


def _cached_df_for_tool(tool_key, uploaded_file):
    """Retrieve a cached processed DataFrame for this tool+file, if it exists.

    Returns the cached df, or None if nothing matches (caller should do the load).
    """
    cache_key = f"_df_cache_{tool_key}"
    file_key = _make_file_key(uploaded_file)
    cached = st.session_state.get(cache_key)
    if cached and cached.get('file_key') == file_key:
        return cached.get('df')
    return None


def _store_cached_df(tool_key, uploaded_file, df):
    """Store a processed DataFrame in session state for this tool+file."""
    cache_key = f"_df_cache_{tool_key}"
    st.session_state[cache_key] = {
        'file_key': _make_file_key(uploaded_file),
        'df': df,
    }


# =====================================================================
# Shared utilities: analysis notes + CSV annotation
# =====================================================================

def _notes_widget(tool_key, label="📝 Analysis notes", placeholder=None):
    """Render a notes text area and return its current value.

    The value persists in session_state so it survives reruns and tool switches.
    Intended to be called near the top of each tool's analysis output so users
    can annotate *before* downloading.
    """
    note_key = f"_notes_{tool_key}"
    if note_key not in st.session_state:
        st.session_state[note_key] = ""

    placeholder = placeholder or (
        "e.g., Prepared for sociology liaison meeting, Nov 2025. "
        "Follow-up: discuss HQ underperformance with Dr. Chen."
    )

    with st.expander(label, expanded=False):
        st.caption("Notes are saved in this session and included as a header comment "
                   "in any CSV you download below. They won't persist if you close "
                   "the browser tab.")
        notes = st.text_area(
            "Add context, rationale, or follow-up items:",
            value=st.session_state[note_key],
            placeholder=placeholder,
            key=f"{note_key}_widget",
            height=100,
        )
        st.session_state[note_key] = notes
    return notes


def _annotate_csv(df, notes, extra_meta=None):
    """Return CSV bytes with an optional notes header block prepended.

    The notes appear as CSV comment lines (prefixed with #) which Excel reads
    as a single first row but most CSV libraries skip. Kept simple and portable.
    """
    from io import StringIO
    from datetime import datetime

    lines = []
    lines.append(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    if extra_meta:
        for k, v in extra_meta.items():
            lines.append(f"# {k}: {v}")
    if notes and notes.strip():
        lines.append("# Notes:")
        for ln in notes.strip().splitlines():
            lines.append(f"#   {ln}")
    lines.append("")  # blank line before CSV body

    buf = StringIO()
    if lines:
        buf.write("\n".join(lines) + "\n")
    df.to_csv(buf, index=False)
    return buf.getvalue()


# =====================================================================
# Shared utilities: download tray
# =====================================================================

def _reset_tray(tool_key):
    """Clear the tray for this tool. Call at the start of a fresh render pass
    so stale artifacts from a previous run don't leak into the ZIP."""
    st.session_state[f"_tray_{tool_key}"] = []


def _add_to_tray(tool_key, filename, data):
    """Register a downloadable artifact (CSV string or bytes) for this tool."""
    tray_key = f"_tray_{tool_key}"
    if tray_key not in st.session_state:
        st.session_state[tray_key] = []
    # De-dup: if this filename is already in the tray, overwrite it
    tray = st.session_state[tray_key]
    tray[:] = [item for item in tray if item[0] != filename]
    tray.append((filename, data))


def _render_download_tray(tool_key, zip_filename="results.zip"):
    """Render a 'Download all' button that bundles everything in the tray."""
    tray = st.session_state.get(f"_tray_{tool_key}", [])
    if not tray:
        return
    import zipfile
    from io import BytesIO as _BIO
    buf = _BIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename, data in tray:
            if isinstance(data, str):
                data = data.encode("utf-8")
            zf.writestr(filename, data)
    buf.seek(0)
    count = len(tray)
    st.download_button(
        f"📦 Download all ({count} file{'s' if count != 1 else ''}) as ZIP",
        buf.getvalue(),
        zip_filename,
        "application/zip",
        key=f"_tray_dl_{tool_key}",
        use_container_width=True,
        type="primary",
    )
    with st.expander(f"Files included ({count})", expanded=False):
        for filename, _ in tray:
            st.caption(f"• {filename}")



# =====================================================================
# ACQUISITION RECOMMENDATION SCORER
# =====================================================================

def _ensure_nltk():
    """Download NLTK data if needed."""
    if not NLTK_AVAILABLE:
        return False
    try:
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
    return True


def normalize_author(author):
    """Normalize author for lookup, generating reversed forms."""
    if pd.isna(author) or not isinstance(author, str):
        return set()
    norm = normalize_text(author)
    if len(norm) <= 2:
        return set()
    candidates = {norm}
    if "," in author:
        parts = [normalize_text(p) for p in author.split(",", 1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            candidates.add(f"{parts[1]} {parts[0]}".strip())
    return candidates


REQUIRED_CHECKOUT_COLS = {"title", "checkouts"}
RECOMMENDED_CHECKOUT_COLS = {"author", "subjects", "lc_classification"}
REQUIRED_REC_COLS = {"title"}
RECOMMENDED_REC_COLS = {"author", "subjects", "lc_classification"}


def _suggest(col, candidates, threshold=0.75):
    best, best_score = None, 0.0
    for c in candidates:
        score = SequenceMatcher(None, col.lower(), c.lower()).ratio()
        if score > best_score:
            best, best_score = c, score
    return best if best_score >= threshold else None


def validate_columns(df, required, recommended, file_label):
    actual = set(df.columns.str.lower())
    df.columns = df.columns.str.lower()
    warnings = []
    valid = True
    for col in required:
        if col not in actual:
            suggestion = _suggest(col, actual)
            hint = f" Did you mean **`{suggestion}`**?" if suggestion else ""
            st.error(f"❌ **{file_label}** is missing required column `{col}`.{hint}")
            valid = False
    for col in recommended:
        if col not in actual:
            suggestion = _suggest(col, actual)
            hint = f" (closest match: `{suggestion}`)" if suggestion else ""
            warnings.append(f"`{col}` not found{hint} — scores for this factor will be 0")
    return valid, warnings


def validate_checkouts_numeric(df):
    if "checkouts" not in df.columns:
        return df
    original = df["checkouts"].copy()
    df["checkouts"] = pd.to_numeric(df["checkouts"], errors="coerce").fillna(0)
    bad = original[df["checkouts"] == 0][original != 0].count()
    if bad > 0:
        st.warning(f"⚠️ {bad} rows in the checkouts column had non-numeric values and were set to 0.")
    return df


def consolidate_checkouts(df):
    key_cols = [c for c in ["title", "author"] if c in df.columns]
    if not key_cols:
        return df
    rows_before = len(df)
    multi_year_titles = df[df.duplicated(subset=key_cols, keep=False)][key_cols[0]].nunique()
    if multi_year_titles == 0:
        return df
    agg_rules = {}
    for col in df.columns:
        if col in key_cols:
            continue
        if col == "checkouts" or pd.api.types.is_numeric_dtype(df[col]):
            agg_rules[col] = "sum"
        else:
            agg_rules[col] = "first"
    consolidated = df.groupby(key_cols, as_index=False, sort=False).agg(agg_rules)
    rows_after = len(consolidated)
    st.info(f"📅 **Multi-year data:** {multi_year_titles} title(s) consolidated "
            f"({rows_before} rows → {rows_after} unique titles). Checkouts summed.")
    return consolidated.reset_index(drop=True)


def check_duplicates_recommendations(df):
    key_cols = [c for c in ["title", "author"] if c in df.columns]
    if not key_cols:
        return df
    dupes = df.duplicated(subset=key_cols, keep="first").sum()
    if dupes > 0:
        st.warning(f"⚠️ Recommendations: {dupes} duplicate row(s) removed.")
        df = df.drop_duplicates(subset=key_cols, keep="first").reset_index(drop=True)
    return df


def extract_all_subjects(df):
    subject_counts = defaultdict(int)
    if "subjects" not in df.columns:
        return subject_counts
    for subjects_str in df["subjects"].dropna():
        for subject in split_subjects(subjects_str):
            if subject:
                subject_counts[subject] += 1
    return dict(subject_counts)


# Synonym map (kept in Tool 3 since only the scorer uses it)
BUILTIN_SYNONYM_GROUPS = {
    "radical_extreme": ["radical", "extreme", "extremist", "fringe", "militant", "revolutionary"],
    "conservative": ["conservative", "traditional", "right-wing", "reactionary"],
    "progressive_liberal": ["progressive", "liberal", "left-wing", "reformist"],
    "equity_justice": ["equity", "equality", "justice", "fairness", "parity"],
    "inclusion_diversity": ["inclusion", "diversity", "belonging", "representation", "multiculturalism"],
    "discrimination": ["discrimination", "bias", "prejudice", "racism", "sexism", "oppression"],
    "climate_change": ["climate change", "global warming", "greenhouse", "carbon", "emissions"],
    "environment_ecology": ["environment", "ecology", "ecosystem", "biodiversity", "conservation", "sustainability"],
    "mental_health": ["mental health", "mental illness", "psychiatric", "psychological", "wellbeing"],
    "chronic_disease": ["chronic disease", "chronic illness", "long-term condition", "comorbidity"],
    "infectious_disease": ["infectious disease", "epidemic", "pandemic", "outbreak", "pathogen"],
    "violence": ["violence", "aggression", "assault", "brutality", "coercion"],
    "war_conflict": ["war", "conflict", "warfare", "combat", "armed conflict", "insurgency"],
    "migration": ["migration", "immigration", "emigration", "diaspora", "mobility"],
    "refugee": ["refugee", "asylum seeker", "displaced person", "exile"],
    "artificial_intelligence": ["artificial intelligence", "machine learning", "deep learning", "AI", "neural network"],
    "data_privacy": ["privacy", "data protection", "surveillance", "tracking"],
    "poverty_inequality": ["poverty", "inequality", "deprivation", "disadvantage", "underserved"],
    "pedagogy_teaching": ["pedagogy", "teaching", "instruction", "education", "curriculum", "learning"],
    "gender_identity": ["gender", "gender identity", "transgender", "nonbinary"],
    "sexuality": ["sexuality", "sexual orientation", "LGBTQ", "queer"],
    "race_ethnicity": ["race", "ethnicity", "racial", "ethnic"],
    "religion_faith": ["religion", "faith", "spirituality", "theology"],
}


def build_synonym_map(stemmer, user_overrides_df=None):
    groups = {label: list(terms) for label, terms in BUILTIN_SYNONYM_GROUPS.items()}
    if user_overrides_df is not None and not user_overrides_df.empty:
        for _, row in user_overrides_df.iterrows():
            term = str(row.get("term", "")).strip()
            label = str(row.get("group_label", "")).strip()
            if term and label:
                groups.setdefault(label, []).append(term)
    synonym_map = {}
    for label, terms in groups.items():
        for term in terms:
            norm = normalize_text(term)
            for word in norm.split():
                stemmed = stemmer.stem(word)
                if len(stemmed) > 2:
                    synonym_map.setdefault(stemmed, label)
    return synonym_map


def apply_synonym_map(terms, synonym_map):
    return [synonym_map.get(t, t) for t in terms]


class FacultyScorer:
    def __init__(self, faculty_df, stemmer, synonym_map):
        self.stemmer = stemmer
        self.synonym_map = synonym_map
        self.faculty_index = self._build_index(faculty_df)

    def _tokenize(self, text):
        norm = normalize_text(text) if text else ""
        if not norm:
            return []
        stemmed = [self.stemmer.stem(w) for w in norm.split() if len(w) > 2]
        return apply_synonym_map(stemmed, self.synonym_map)

    def _build_index(self, faculty_df):
        index = []
        for _, row in faculty_df.iterrows():
            name = str(row.get("name", "")).strip()
            dept = str(row.get("department", "")).strip()
            interests_raw = str(row.get("research_interests", ""))
            tokens = set(self._tokenize(interests_raw))
            if tokens:
                index.append({"name": name, "department": dept, "tokens": tokens})
        return index

    def score(self, recommendation):
        raw_subjects = recommendation.get("subjects", "")
        raw_title = recommendation.get("title", "")
        combined = f"{raw_subjects} {raw_title}"
        rec_tokens = set(self._tokenize(combined))
        if not rec_tokens or not self.faculty_index:
            return 0.0, ""
        best_score = 0.0
        best_label = ""
        for faculty in self.faculty_index:
            fac_tokens = faculty["tokens"]
            if not fac_tokens:
                continue
            intersection = rec_tokens & fac_tokens
            union = rec_tokens | fac_tokens
            if not union:
                continue
            jaccard = len(intersection) / len(union)
            scaled = min(jaccard * 300, 100.0)
            if scaled > best_score:
                best_score = scaled
                dept_str = f" ({faculty['department']})" if faculty["department"] else ""
                best_label = f"{faculty['name']}{dept_str}"
        return round(best_score, 2), best_label


class RecommendationScorer:
    def __init__(self, checkouts_df, synonym_map=None):
        self.checkouts_df = checkouts_df
        self.stemmer = SnowballStemmer("english")
        self.synonym_map = synonym_map or {}
        self.total_docs = len(checkouts_df)
        self.semantic_groups = self._build_semantic_groups()
        self.author_checkout_map = self._build_author_map()
        self.lc_checkout_map = self._build_lc_map()
        self.subject_terms = self._extract_subject_terms_enhanced()
        self.term_frequencies = self._calculate_term_frequencies()

    def _build_semantic_groups(self):
        groups = {
            "computer_science": ["comput", "programm", "softwar", "algorithm", "code"],
            "artificial_intelligence": ["artifici", "intellig", "machin", "learn", "neural", "deep", "ai"],
            "data_analytics": ["data", "analysi", "analyt", "statist", "visual", "databas"],
            "psychology": ["psycholog", "mental", "health", "behavior", "cognit", "psychiatr"],
            "sociology": ["sociolog", "social", "cultur", "commun", "society"],
            "economics": ["econom", "market", "trade", "finance", "financi", "busi"],
            "political_science": ["politic", "govern", "polici", "democrat", "elect", "legisl"],
            "history": ["histor", "histori", "past", "ancient", "mediev", "modern", "war"],
            "philosophy": ["philosoph", "ethic", "moral", "metaphys", "epistemolog"],
            "literature": ["literatur", "novel", "fiction", "poetri", "drama", "narrat"],
            "education": ["educ", "teach", "learn", "pedagog", "curriculum", "school"],
            "law": ["law", "legal", "court", "justic", "judg", "attorney"],
            "medicine": ["medicin", "medic", "health", "clinic", "hospit", "treatment", "diseas"],
            "environmental": ["environ", "climat", "ecolog", "sustain", "conserv", "ecosyst"],
            "biology": ["biolog", "life", "scienc", "organ", "cell", "geneti"],
            "library_science": ["librari", "inform", "catalog", "bibliograph", "archiv", "collect"],
            "gender_studies": ["gender", "feminis", "women", "masculin", "queer", "lgbt"],
            "diversity": ["divers", "inclus", "equiti", "racial", "ethnic", "multicultural"],
        }
        term_to_group = {}
        for group_id, terms in groups.items():
            for term in terms:
                term_to_group.setdefault(term, []).append(group_id)
        return {"groups": groups, "term_to_group": term_to_group}

    def _build_author_map(self):
        author_map = defaultdict(list)
        for _, row in self.checkouts_df.iterrows():
            for candidate in normalize_author(row.get("author", "")):
                author_map[candidate].append(row.get("checkouts", 0))
        return dict(author_map)

    def _build_lc_map(self):
        lc_map = defaultdict(list)
        for _, row in self.checkouts_df.iterrows():
            if pd.notna(row.get("lc_classification")):
                lc_prefix = extract_lc_prefix(row["lc_classification"])
                if lc_prefix:
                    lc_map[lc_prefix].append(row.get("checkouts", 0))
        return dict(lc_map)

    def _extract_subject_terms_enhanced(self):
        all_terms = []
        doc_term_counts = defaultdict(set)
        unique_subject_docs = set()
        for _, row in self.checkouts_df.iterrows():
            if pd.notna(row.get("subjects")):
                subjects = split_subjects(str(row["subjects"]))
                checkouts = row.get("checkouts", 0)
                for i, subject in enumerate(subjects):
                    unique_subject_docs.add(subject)
                    hw = 1.0 if i == 0 else 0.7
                    for term in self._tokenize_and_stem(subject):
                        all_terms.append((term, checkouts * hw, False))
                        doc_term_counts[term].add(subject)
                    for bigram in self._extract_bigrams(subject):
                        all_terms.append((bigram, checkouts * hw * 1.3, True))
                        doc_term_counts[bigram].add(subject)
        total_subject_docs = max(len(unique_subject_docs), 1)
        term_checkouts = defaultdict(list)
        for term, checkout_count, _ in all_terms:
            term_checkouts[term].append(checkout_count)
        term_scores = {}
        for term, counts in term_checkouts.items():
            avg = sum(counts) / len(counts)
            docs_with = len(doc_term_counts[term])
            idf = np.log(total_subject_docs / (1 + docs_with))
            term_scores[term] = avg * (1 + idf * 0.3)
        return term_scores

    def _extract_bigrams(self, text):
        if not text:
            return []
        words = [w for w in text.split() if len(w) > 2]
        return [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]

    def _calculate_term_frequencies(self):
        tf = Counter()
        for _, row in self.checkouts_df.iterrows():
            if pd.notna(row.get("subjects")):
                norm = normalize_text(str(row["subjects"]))
                tf.update(self._tokenize_and_stem(norm))
        return tf

    def _tokenize_and_stem(self, text):
        norm = normalize_text(text) if text else ""
        if not norm:
            return []
        stemmed = [self.stemmer.stem(w) for w in norm.split() if len(w) > 2]
        return apply_synonym_map(stemmed, self.synonym_map)

    def _get_synonyms(self, word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(self.stemmer.stem(lemma.name().lower()))
        return synonyms

    def _get_semantic_matches(self, term):
        matches = []
        ttg = self.semantic_groups["term_to_group"]
        if term in ttg:
            groups = ttg[term]
            for gid in groups:
                for gt in self.semantic_groups["groups"][gid]:
                    if gt in self.subject_terms:
                        if gt in ttg:
                            shared = set(groups) & set(ttg[gt])
                            strength = len(shared) * 0.85
                        else:
                            strength = 0.85
                        matches.append((gt, self.subject_terms[gt], strength))
        return matches

    def _fuzzy_match_terms(self, term, threshold=0.80):
        max_score = 0
        for existing_term in self.subject_terms:
            sim = SequenceMatcher(None, term, existing_term).ratio()
            if sim >= threshold:
                max_score = max(max_score, self.subject_terms[existing_term])
        return max_score

    def _calculate_subject_similarity(self, recommendation):
        raw = recommendation.get("subjects")
        if pd.isna(raw) or not self.subject_terms:
            return 0.0
        norm = normalize_text(str(raw))
        rec_terms = self._tokenize_and_stem(norm)
        rec_bigrams = self._extract_bigrams(norm)
        all_rec = rec_terms + rec_bigrams
        if not all_rec:
            return 0.0
        total_score = 0
        matched = 0
        exact = 0
        for rt in all_rec:
            rec_syns = self._get_synonyms(rt.replace("_", " "))
            rec_syns.add(rt)
            max_ts = 0
            if rt in self.subject_terms:
                max_ts = self.subject_terms[rt] * 1.5
                exact += 1
            if max_ts == 0:
                for syn in rec_syns:
                    if syn in self.subject_terms:
                        max_ts = max(max_ts, self.subject_terms[syn])
            if max_ts == 0:
                for _, ts, strength in self._get_semantic_matches(rt):
                    max_ts = max(max_ts, ts * strength)
            if max_ts == 0:
                max_ts = self._fuzzy_match_terms(rt)
            if max_ts > 0:
                matched += 1
                total_score += max_ts
        if matched == 0:
            return 0.0
        avg = total_score / matched
        max_c = max(self.subject_terms.values())
        coverage = matched / len(all_rec)
        exact_ratio = exact / len(all_rec)
        cw = min(0.6 + 0.4 * coverage + 0.2 * exact_ratio, 1.0)
        return (avg / max_c) * 100 * cw

    def _calculate_lc_score(self, recommendation):
        if pd.isna(recommendation.get("lc_classification")) or not self.lc_checkout_map:
            return 0.0
        lc_prefix = extract_lc_prefix(recommendation["lc_classification"])
        if not lc_prefix or lc_prefix not in self.lc_checkout_map:
            return 0.0
        vals = self.lc_checkout_map[lc_prefix]
        avg = sum(vals) / len(vals)
        max_avg = max(sum(v) / len(v) for v in self.lc_checkout_map.values())
        return (avg / max_avg) * 100

    def _calculate_author_score(self, recommendation):
        candidates = normalize_author(recommendation.get("author", ""))
        if not candidates or not self.author_checkout_map:
            return 0.0
        max_avg = max(sum(v) / len(v) for v in self.author_checkout_map.values())
        best = 0.0
        for c in candidates:
            if c in self.author_checkout_map:
                vals = self.author_checkout_map[c]
                avg = sum(vals) / len(vals)
                best = max(best, (avg / max_avg) * 100)
        return best

    def score_recommendations(self, recommendations_df,
                              subject_weight=0.5, lc_weight=0.3,
                              author_weight=0.2, faculty_weight=0.0,
                              faculty_scorer=None):
        results = []
        for _, rec in recommendations_df.iterrows():
            ss = self._calculate_subject_similarity(rec)
            ls = self._calculate_lc_score(rec)
            aus = self._calculate_author_score(rec)
            fs, mf = 0.0, ""
            if faculty_scorer and faculty_weight > 0:
                fs, mf = faculty_scorer.score(rec)
            likelihood = ss * subject_weight + ls * lc_weight + aus * author_weight + fs * faculty_weight
            d = rec.to_dict()
            d["likelihood_score"] = round(likelihood, 2)
            d["similarity_score"] = round(ss, 2)
            d["checkout_volume_score"] = round(ls, 2)
            d["author_popularity_score"] = round(aus, 2)
            d["faculty_interest_score"] = round(fs, 2)
            d["matched_faculty"] = mf
            results.append(d)
        rdf = pd.DataFrame(results)
        rdf = rdf.sort_values("likelihood_score", ascending=False).reset_index(drop=True)
        return rdf


def generate_report(results_df):
    lines = ["=" * 80, "LIBRARY BOOK RECOMMENDATION REPORT", "=" * 80, "",
             "SUMMARY", "-" * 80,
             f"Total Recommendations Analyzed: {len(results_df)}", ""]
    high = len(results_df[results_df["likelihood_score"] >= 70])
    medium = len(results_df[(results_df["likelihood_score"] >= 40) & (results_df["likelihood_score"] < 70)])
    low = len(results_df[results_df["likelihood_score"] < 40])
    lines += [
        f"High Priority (70-100):   {high} books  ({high/max(1,len(results_df))*100:.1f}%)",
        f"Medium Priority (40-69):  {medium} books  ({medium/max(1,len(results_df))*100:.1f}%)",
        f"Low Priority (0-39):      {low} books  ({low/max(1,len(results_df))*100:.1f}%)",
        "", "TOP 20 RECOMMENDATIONS", "=" * 80, "",
    ]
    for idx, row in results_df.head(20).iterrows():
        lines += [
            f"#{idx + 1}: {row['title']}",
            f"   Author: {row.get('author', 'N/A')}",
            f"   Overall Score: {row['likelihood_score']:.1f}/100",
            f"   - Subject Similarity:    {row['similarity_score']:.1f}",
            f"   - Checkout Volume:       {row['checkout_volume_score']:.1f}",
            f"   - Author Popularity:     {row['author_popularity_score']:.1f}",
            f"   - Faculty Interest:      {row.get('faculty_interest_score', 0.0):.1f}",
            f"   - Matched Faculty:       {row.get('matched_faculty', 'N/A')}", "",
        ]
    return "\n".join(lines)


def page_recommendation_scorer():
    """Tool 3: Acquisition Recommendation Scorer."""
    if not NLTK_AVAILABLE:
        st.error("The `nltk` package is required for this tool. Install with: `pip install nltk`")
        return
    _ensure_nltk()

    st.header("📊 Acquisition Recommendation Scorer")
    st.markdown(
        "**What should we buy next?** Score candidate books against your checkout "
        "history to prioritize purchases."
    )
    with st.expander("ℹ️ When to use this tool"):
        st.markdown(
            "- **Collections:** Evaluating vendor slip lists and GOBI picks, approval-plan "
            "exceptions, triaging faculty requests, flipping DDA candidates to purchase, "
            "reviewing author/publisher lists for standing orders.\n"
            "- **Instruction:** Occasionally — clusters of high-scoring recommendations in "
            "one area can reveal curricular momentum worth a targeted info-lit session.\n"
            "- **Outreach:** Showing faculty *why* a recommendation scored high (with "
            "the faculty-interest score naming their research match) makes this a strong "
            "conversation-starter at liaison meetings."
        )


    # File uploads
    st.subheader("Step 1: Upload your data")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Checkouts File** — Required: `title`, `checkouts`; "
                    "Recommended: `author`, `subjects`, `lc_classification`")
        checkouts_file = st.file_uploader("Upload checkouts CSV", type=["csv"], key="rec_checkouts")
    with c2:
        st.markdown("**Recommendations File** — Required: `title`; "
                    "Recommended: `author`, `subjects`, `lc_classification`")
        recommendations_file = st.file_uploader("Upload recommendations CSV", type=["csv"], key="rec_recs")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Faculty Research Interests** *(optional)* — "
                    "Columns: `name`, `department`, `research_interests`")
        faculty_file = st.file_uploader("Upload faculty CSV", type=["csv"], key="rec_faculty")
    with c4:
        st.markdown("**Custom Synonym Groups** *(optional)* — "
                    "Columns: `term`, `group_label`")
        synonym_file = st.file_uploader("Upload synonym CSV", type=["csv"], key="rec_synonyms")

    if checkouts_file and recommendations_file:
        try:
            # Check session cache for each file independently (different shape/roles)
            cached_co = _cached_df_for_tool("rec_checkouts", checkouts_file)
            cached_rec = _cached_df_for_tool("rec_recommendations", recommendations_file)

            if cached_co is not None and cached_rec is not None:
                checkouts_df = cached_co.copy()
                recommendations_df = cached_rec.copy()
                st.success(f"✅ Using cached data for both files")
            else:
                with st.spinner("Loading and validating data..."):
                    checkouts_df = pd.read_csv(checkouts_file)
                    recommendations_df = pd.read_csv(recommendations_file)
                _store_cached_df("rec_checkouts", checkouts_file, checkouts_df)
                _store_cached_df("rec_recommendations", recommendations_file, recommendations_df)

            co_valid, co_warns = validate_columns(checkouts_df, REQUIRED_CHECKOUT_COLS,
                                                   RECOMMENDED_CHECKOUT_COLS, "Checkouts file")
            re_valid, re_warns = validate_columns(recommendations_df, REQUIRED_REC_COLS,
                                                   RECOMMENDED_REC_COLS, "Recommendations file")

            if co_warns or re_warns:
                with st.expander("⚠️ Column warnings"):
                    for w in co_warns + re_warns:
                        st.markdown(f"- {w}")
            if not (co_valid and re_valid):
                st.stop()

            checkouts_df = validate_checkouts_numeric(checkouts_df)
            checkouts_df = consolidate_checkouts(checkouts_df)
            recommendations_df = check_duplicates_recommendations(recommendations_df)

            # Faculty
            faculty_df = None
            if faculty_file:
                faculty_df = pd.read_csv(faculty_file)
                faculty_df.columns = faculty_df.columns.str.lower()
                missing_fac = [c for c in ["name", "department", "research_interests"]
                               if c not in faculty_df.columns]
                if missing_fac:
                    st.warning(f"⚠️ Faculty file missing: {missing_fac}. Faculty scoring disabled.")
                    faculty_df = None
                else:
                    st.success(f"✅ Loaded {len(faculty_df)} faculty records")

            # Synonyms
            synonym_overrides_df = None
            if synonym_file:
                synonym_overrides_df = pd.read_csv(synonym_file)
                synonym_overrides_df.columns = synonym_overrides_df.columns.str.lower()
                if not {"term", "group_label"}.issubset(synonym_overrides_df.columns):
                    st.warning("⚠️ Synonym file needs `term` and `group_label` columns.")
                    synonym_overrides_df = None
                else:
                    st.success(f"✅ Loaded {len(synonym_overrides_df)} synonym mappings")

            st.success(f"✅ Loaded {len(checkouts_df)} checkout records and "
                       f"{len(recommendations_df)} recommendations")

            with st.expander("📋 Preview Data"):
                pc1, pc2 = st.columns(2)
                with pc1:
                    st.write("**Checkouts:**")
                    st.dataframe(checkouts_df.head())
                with pc2:
                    st.write("**Recommendations:**")
                    st.dataframe(recommendations_df.head())

            # --- (Collection Insights panel removed — see Collection Profiler instead) ---

            # Scoring configuration — preset-first, with manual override in advanced
            st.subheader("Step 2: Choose scoring approach")

            PRESETS = {
                "Balanced": {"subject": 0.50, "lc": 0.30, "author": 0.20, "faculty": 0.00},
                "Subject-focused": {"subject": 0.70, "lc": 0.20, "author": 0.10, "faculty": 0.00},
                "Faculty-driven": {"subject": 0.35, "lc": 0.20, "author": 0.10, "faculty": 0.35},
            }
            # If faculty file is loaded, default to the preset that uses it
            default_preset = "Faculty-driven" if faculty_df is not None else "Balanced"

            # Initialize preset selection in session state
            if "rec_preset_choice" not in st.session_state:
                st.session_state["rec_preset_choice"] = default_preset

            preset_options = list(PRESETS.keys()) + ["Advanced (custom weights)"]
            # Disable Faculty-driven if no faculty file
            preset_help = ("Pick how to weight the four scoring factors. "
                           "Faculty-driven needs a faculty CSV (upload one above).")
            preset_choice = st.radio(
                "Scoring approach:",
                preset_options,
                index=preset_options.index(st.session_state["rec_preset_choice"])
                      if st.session_state["rec_preset_choice"] in preset_options else 0,
                horizontal=True,
                key="rec_preset_choice",
                help=preset_help,
            )

            if preset_choice in PRESETS:
                w = PRESETS[preset_choice]
                # If a preset wants faculty weight but no faculty file is loaded, warn and fall back
                if w["faculty"] > 0 and faculty_df is None:
                    st.warning(
                        "**Faculty-driven** needs a faculty CSV. Upload one above or "
                        "pick a different approach. Falling back to **Balanced** for now."
                    )
                    w = PRESETS["Balanced"]
                subject_weight = w["subject"]
                lc_weight = w["lc"]
                author_weight = w["author"]
                faculty_weight = w["faculty"]
                # Show what the preset does in a compact caption
                parts = []
                parts.append(f"Subject {int(subject_weight*100)}%")
                parts.append(f"LC {int(lc_weight*100)}%")
                parts.append(f"Author {int(author_weight*100)}%")
                if faculty_weight > 0:
                    parts.append(f"Faculty {int(faculty_weight*100)}%")
                st.caption(" · ".join(parts))
            else:
                # Advanced mode — keep the four sliders, with auto-normalize help
                with st.container(border=True):
                    st.caption(
                        "Set any values you want; they'll be normalized to sum to 1.0 "
                        "before scoring."
                    )
                    _fac_default = 0.15 if faculty_df is not None else 0.0
                    wc1, wc2, wc3, wc4 = st.columns(4)
                    with wc1:
                        subject_weight = st.slider(
                            "Subject Similarity", 0.0, 1.0,
                            0.45 if faculty_df else 0.5, 0.05, key="rec_sw"
                        )
                    with wc2:
                        lc_weight = st.slider(
                            "LC Classification", 0.0, 1.0,
                            0.25 if faculty_df else 0.3, 0.05, key="rec_lw"
                        )
                    with wc3:
                        author_weight = st.slider(
                            "Author Popularity", 0.0, 1.0,
                            0.15 if faculty_df else 0.2, 0.05, key="rec_aw"
                        )
                    with wc4:
                        faculty_weight = st.slider(
                            "Faculty Interest", 0.0, 1.0, _fac_default, 0.05,
                            disabled=(faculty_df is None), key="rec_fw"
                        )
                    raw_total = subject_weight + lc_weight + author_weight + faculty_weight
                    if raw_total <= 0:
                        st.error("All weights are zero — set at least one above 0.")
                    else:
                        # Auto-normalize silently
                        subject_weight = subject_weight / raw_total
                        lc_weight = lc_weight / raw_total
                        author_weight = author_weight / raw_total
                        faculty_weight = faculty_weight / raw_total
                        st.caption(
                            f"Normalized: Subject {subject_weight:.0%} · "
                            f"LC {lc_weight:.0%} · Author {author_weight:.0%} · "
                            f"Faculty {faculty_weight:.0%}"
                        )

            for key in ("rec_results", "rec_checkouts_scored", "rec_recs_scored",
                        "rec_faculty_scored", "rec_weights"):
                if key not in st.session_state:
                    st.session_state[key] = None

            if st.button("Score recommendations", type="primary", key="rec_score_btn"):
                with st.spinner("Analyzing..."):
                    _stemmer = SnowballStemmer("english")
                    syn_map = build_synonym_map(_stemmer, synonym_overrides_df)
                    scorer = RecommendationScorer(checkouts_df, synonym_map=syn_map)
                    _faculty_scorer = None
                    if faculty_df is not None and faculty_weight > 0:
                        _faculty_scorer = FacultyScorer(faculty_df, _stemmer, syn_map)
                    results_df = scorer.score_recommendations(
                        recommendations_df,
                        subject_weight=subject_weight, lc_weight=lc_weight,
                        author_weight=author_weight, faculty_weight=faculty_weight,
                        faculty_scorer=_faculty_scorer,
                    )
                st.success("✅ Analysis complete!")
                st.session_state["rec_results"] = results_df
                st.session_state["rec_checkouts_scored"] = checkouts_df
                st.session_state["rec_recs_scored"] = recommendations_df
                st.session_state["rec_faculty_scored"] = faculty_df
                st.session_state["rec_weights"] = {
                    "subject": subject_weight, "lc": lc_weight,
                    "author": author_weight, "faculty": faculty_weight,
                }

            # Results display
            if st.session_state["rec_results"] is not None:
                results_df = st.session_state["rec_results"]
                st.subheader("Step 3: Review results")

                # Fresh tray for this render pass
                _reset_tray("rec_scorer")

                # Notes — annotate before downloading
                notes = _notes_widget(
                    "recommendation_scorer",
                    placeholder="e.g., YBP slip list Nov 2025, sociology liaison review. "
                                "Weights adjusted to favor faculty interest (Dr. Chen's lab)."
                )

                tab_r1, tab_r2, tab_r3, tab_r4 = st.tabs([
                    "Scored recommendations", "Score distribution",
                    "Subject analysis", "Faculty analysis"
                ])

                with tab_r1:
                    high_p = results_df[results_df["likelihood_score"] >= 70]
                    med_p = results_df[(results_df["likelihood_score"] >= 40) &
                                        (results_df["likelihood_score"] < 70)]
                    low_p = results_df[results_df["likelihood_score"] < 40]
                    tc1, tc2, tc3, tc4 = st.columns(4)
                    tc1.metric("Total Scored", len(results_df))
                    tc2.metric("🟢 High (70+)", len(high_p))
                    tc3.metric("🟡 Medium (40-69)", len(med_p))
                    tc4.metric("🔴 Low (<40)", len(low_p))

                    search = st.text_input("Search by title or author", "", key="rec_search")
                    min_score = st.slider("Minimum score", 0, 100, 0, key="rec_min")
                    filtered = results_df.copy()
                    if search:
                        mask = (filtered["title"].str.contains(search, case=False, na=False) |
                                filtered.get("author", pd.Series(dtype=str))
                                .str.contains(search, case=False, na=False))
                        filtered = filtered[mask]
                    filtered = filtered[filtered["likelihood_score"] >= min_score]

                    def get_priority(s):
                        if s >= 70: return "🟢 High"
                        if s >= 40: return "🟡 Medium"
                        return "🔴 Low"

                    display = filtered.copy()
                    display["Priority"] = display["likelihood_score"].apply(get_priority)
                    pcols = ["Priority", "title", "author", "likelihood_score",
                             "similarity_score", "checkout_volume_score",
                             "author_popularity_score", "faculty_interest_score", "matched_faculty"]
                    others = [c for c in display.columns if c not in pcols]
                    display = display[[c for c in pcols if c in display.columns] + others]
                    st.dataframe(display, use_container_width=True, height=600)

                with tab_r2:
                    scores = results_df["likelihood_score"]
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(x=scores, nbinsx=20, marker_color="#285C4D"))
                    fig_hist.add_vline(x=70, line_dash="dash", line_color="#2ecc71",
                                       annotation_text="High (70)")
                    fig_hist.add_vline(x=40, line_dash="dash", line_color="#f39c12",
                                       annotation_text="Medium (40)")
                    fig_hist.update_layout(title="Score Distribution",
                                           xaxis_title="Score", yaxis_title="Count",
                                           height=400, showlegend=False)
                    st.plotly_chart(fig_hist, use_container_width=True)
                    sc1, sc2, sc3 = st.columns(3)
                    sc1.metric("Mean", f"{scores.mean():.1f}")
                    sc2.metric("Median", f"{scores.median():.1f}")
                    sc3.metric("Std Dev", f"{scores.std():.1f}")

                with tab_r3:
                    co_subj = extract_all_subjects(st.session_state["rec_checkouts_scored"])
                    rec_subj = extract_all_subjects(st.session_state["rec_recs_scored"])
                    sa1, sa2, sa3 = st.columns(3)
                    sa1.metric("Checkout Subjects", len(co_subj))
                    sa2.metric("Recommendation Subjects", len(rec_subj))
                    overlap = len(set(co_subj) & set(rec_subj))
                    sa3.metric("Common Subjects", overlap)

                    common = {s: {"co": co_subj[s], "rec": rec_subj[s]}
                              for s in set(co_subj) & set(rec_subj)}
                    if common:
                        cdf = pd.DataFrame([
                            {"Subject": s, "In Checkouts": d["co"], "In Recommendations": d["rec"],
                             "Total": d["co"] + d["rec"]}
                            for s, d in common.items()
                        ]).sort_values("Total", ascending=False)
                        st.dataframe(cdf.head(30), use_container_width=True, height=400)

                    gap_subj = {k: v for k, v in co_subj.items() if k not in rec_subj and v >= 2}
                    if gap_subj:
                        st.subheader("Recommendation gaps")
                        st.markdown("High-circulation subjects missing from your recommendations list:")
                        gdf = pd.DataFrame([
                            {"Subject": s, "Checkout Occurrences": c}
                            for s, c in sorted(gap_subj.items(), key=lambda x: -x[1])[:30]
                        ])
                        st.dataframe(gdf, use_container_width=True, height=300)

                with tab_r4:
                    fac_scored = st.session_state.get("rec_faculty_scored")
                    if fac_scored is None:
                        st.info("No faculty file uploaded. Upload one and re-score to see this analysis.")
                    elif ("faculty_interest_score" not in results_df.columns
                          or results_df["faculty_interest_score"].sum() == 0):
                        st.warning("Faculty scores are zero — set faculty weight > 0 and re-score.")
                    else:
                        fac_results = results_df[results_df["matched_faculty"].str.strip() != ""].copy()
                        fc1, fc2 = st.columns(2)
                        fc1.metric("Faculty Members", len(fac_scored))
                        fc2.metric("Matched Recommendations", len(fac_results))
                        if len(fac_results) > 0:
                            st.dataframe(
                                fac_results.sort_values("faculty_interest_score", ascending=False)
                                .head(20)[["title", "author", "likelihood_score",
                                           "faculty_interest_score", "matched_faculty"]],
                                use_container_width=True, height=400
                            )

                # Downloads
                st.subheader("Downloads")
                dc1, dc2, dc3 = st.columns(3)
                _weights = st.session_state.get("rec_weights", {})
                _weights_str = (f"subject={_weights.get('subject', 0)}, "
                                f"lc={_weights.get('lc', 0)}, "
                                f"author={_weights.get('author', 0)}, "
                                f"faculty={_weights.get('faculty', 0)}")
                with dc1:
                    _full_bytes = _annotate_csv(results_df, notes,
                                                extra_meta={'Tool': 'Recommendation Scorer',
                                                            'View': 'Full Results',
                                                            'Weights': _weights_str})
                    st.download_button("📥 Full results (CSV)",
                                       _full_bytes,
                                       "recommendations_scored.csv", "text/csv",
                                       key="rec_dl_full")
                    _add_to_tray("rec_scorer", "recommendations_scored.csv", _full_bytes)
                with dc2:
                    high_df = results_df[results_df["likelihood_score"] >= 70]
                    _high_bytes = _annotate_csv(high_df, notes,
                                                extra_meta={'Tool': 'Recommendation Scorer',
                                                            'View': 'High Priority (≥70)',
                                                            'Weights': _weights_str})
                    st.download_button("📥 High priority only (CSV)",
                                       _high_bytes,
                                       "recommendations_high_priority.csv", "text/csv",
                                       key="rec_dl_high")
                    _add_to_tray("rec_scorer", "recommendations_high_priority.csv", _high_bytes)
                with dc3:
                    # TXT report: prepend notes as a header block (not CSV, so different format)
                    report_body = generate_report(results_df)
                    if notes and notes.strip():
                        from datetime import datetime
                        notes_block = (
                            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                            f"Weights: {_weights_str}\n\n"
                            f"NOTES\n{'-' * 80}\n{notes.strip()}\n\n"
                        )
                        report_body = notes_block + report_body
                    st.download_button("📄 Report (TXT)", report_body,
                                       "recommendation_report.txt", "text/plain",
                                       key="rec_dl_txt")
                    _add_to_tray("rec_scorer", "recommendation_report.txt", report_body)

                # ZIP-all option
                _render_download_tray("rec_scorer",
                                      zip_filename="recommendation_scorer_results.zip")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Check that your CSV files have the required columns.")
    else:
        st.info("Upload both a checkouts file and a recommendations file to begin.")

    # Sidebar instructions
    with st.sidebar:
        st.markdown("---")
        st.subheader("Scorer instructions")
        st.markdown("""
        1. Upload **checkouts CSV** *(required)*
        2. Upload **recommendations CSV** *(required)*
        3. Optionally add **faculty** and **synonym** CSVs
        4. Adjust scoring weights
        5. Click **Score Recommendations**

        **Scores:** 🟢 70+ High · 🟡 40-69 Medium · 🔴 <40 Low
        """)




# =====================================================================
# =====================================================================


# =====================================================================
# Entry point
# =====================================================================

def main():
    """Standalone app entry point. Calls the page function defined above."""
    page_recommendation_scorer()


if __name__ == "__main__":
    main()

