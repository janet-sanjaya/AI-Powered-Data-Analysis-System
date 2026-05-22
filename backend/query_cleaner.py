import difflib
import re

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

REGIONS = ["West", "East", "Central", "South"]
CATEGORIES = ["Electronics", "Clothing", "Furniture", "Grocery"]
BRANCHES = ["A", "B", "C", "D", "E"]

CANONICAL_TERMS = {
    "sales": "sales",
    "revenue": "sales",
    "turnover": "sales",
    "profit": "profit",
    "income": "profit",
    "earnings": "profit",
    "quantity": "quantity",
    "qty": "quantity",
    "orders": "orders",
    "order": "orders",
    "branch": "branch",
    "store": "branch",
    "stores": "branch",
    "location": "branch",
    "locations": "branch",
    "outlet": "branch",
    "outlets": "branch",
    "region": "region",
    "area": "region",
    "zone": "region",
    "territory": "region",
    "territories": "region",
    "category": "category",
    "product": "category",
    "products": "category",
    "month-over-month": "month-over-month",
    "month-to-date": "month-to-date",
    "year-to-date": "year-to-date",
    "year-over-year": "year-over-year",
    "mom": "month-over-month",
    "mtd": "month-to-date",
    "ytd": "year-to-date",
    "yoy": "year-over-year",
    "compare": "compare",
    "comparison": "compare",
    "change": "change",
    "trend": "trend",
    "drop": "drop",
    "decrease": "decrease",
    "decline": "decrease",
    "fall": "decrease",
    "increase": "increase",
    "growth": "increase",
    "rise": "increase",
    "highest": "highest",
    "lowest": "lowest",
    "largest": "largest",
    "smallest": "smallest",
    "total": "total",
    "average": "average",
    "avg": "average",
}

ENTITY_CANONICAL = {
    **{m.lower(): m for m in MONTHS},
    **{r.lower(): r for r in REGIONS},
    **{c.lower(): c for c in CATEGORIES},
    **{b.lower(): b for b in BRANCHES},
}

STOPWORDS = {
    "what", "is", "the", "a", "an", "for", "of", "in", "to", "by", "on",
    "from", "with", "and", "or", "which", "who", "how", "did", "does",
    "do", "was", "were", "be", "been", "this", "that", "these", "those",
    "show", "tell", "give", "me", "all", "each", "every", "any", "than",
    "vs", "versus", "excluding", "excluding", "including", "please",
    "compare", "compared", "caused", "cause"
}

VOCAB = sorted(
    set(
        list(CANONICAL_TERMS.keys())
        + list(CANONICAL_TERMS.values())
        + list(ENTITY_CANONICAL.keys())
        + list(ENTITY_CANONICAL.values())
        + ["sales", "profit", "quantity", "orders", "branch", "region", "category"]
    )
)

PHRASE_PATTERNS = [
    (r"\bmonth\s+over\s+month\b", "month-over-month"),
    (r"\bmonth[-\s]?over[-\s]?month\b", "month-over-month"),
    (r"\bmonth\s+to\s+date\b", "month-to-date"),
    (r"\bmonth[-\s]?to[-\s]?date\b", "month-to-date"),
    (r"\byear\s+to\s+date\b", "year-to-date"),
    (r"\byear[-\s]?to[-\s]?date\b", "year-to-date"),
    (r"\byear\s+over\s+year\b", "year-over-year"),
    (r"\byear[-\s]?over[-\s]?year\b", "year-over-year"),
    (r"\bmonth on month\b", "month-over-month"),
    (r"\bmonths? over months?\b", "month-over-month"),
    (r"\bmom\b", "month-over-month"),
    (r"\bmtd\b", "month-to-date"),
    (r"\bytd\b", "year-to-date"),
    (r"\byoy\b", "year-over-year"),
]

def _fuzzy_correct(word: str) -> str:
    if len(word) < 4:
        return word

    match = difflib.get_close_matches(word, VOCAB, n=1, cutoff=0.84)
    return match[0] if match else word


def _normalize_token(token: str) -> str:
    low = token.lower()

    if low in STOPWORDS:
        return low

    if low in CANONICAL_TERMS:
        return CANONICAL_TERMS[low]

    if low in ENTITY_CANONICAL:
        return ENTITY_CANONICAL[low]

    if len(token) == 1 and token.upper() in BRANCHES:
        return token.upper()

    corrected = _fuzzy_correct(low)

    if corrected in ENTITY_CANONICAL:
        return ENTITY_CANONICAL[corrected]

    if corrected in CANONICAL_TERMS:
        return CANONICAL_TERMS[corrected]

    return corrected


def normalize_query(query: str) -> str:
    """
    Normalize a user query so the LLM sees a cleaner version:
    - lower/upper case normalization
    - synonym replacement
    - simple typo correction
    """
    if not query:
        return ""

    text = query.strip().lower()

    for pattern, replacement in PHRASE_PATTERNS:
        text = re.sub(pattern, replacement, text)

    tokens = re.findall(r"[a-zA-Z]+(?:-[a-zA-Z]+)?|\d+|[^\w\s]", text)
    normalized = []

    for token in tokens:
        if re.fullmatch(r"[^\w\s]", token):
            normalized.append(token)
        elif token.isdigit():
            normalized.append(token)
        else:
            normalized.append(_normalize_token(token))

    cleaned = " ".join(normalized)
    cleaned = re.sub(r"\s+([?.!,;:])", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned