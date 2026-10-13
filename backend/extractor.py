"""Extract skills, contact fields, and experience duration from resume text."""

from __future__ import annotations

import re
from datetime import date
from functools import lru_cache
from typing import Final

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span

SPACY_MODEL: Final[str] = "en_core_web_sm"

SKILL_TAXONOMY: Final[dict[str, frozenset[str]]] = {
    "languages": frozenset(
        {
            "Python",
            "Java",
            "JavaScript",
            "TypeScript",
            "C",
            "C++",
            "C#",
            "Go",
            "Golang",
            "Rust",
            "Ruby",
            "PHP",
            "Swift",
            "Kotlin",
            "Scala",
            "R",
            "MATLAB",
            "SQL",
            "HTML",
            "CSS",
            "Bash",
            "Shell",
            "PowerShell",
            "Dart",
            "Elixir",
            "Haskell",
            "Perl",
            "Lua",
            "Objective-C",
            "Groovy",
            "Julia",
            "SAS",
            "VBA",
        }
    ),
    "web_frameworks": frozenset(
        {
            "React",
            "React Native",
            "Angular",
            "Vue",
            "Vue.js",
            "Next.js",
            "Nuxt",
            "Svelte",
            "SvelteKit",
            "Django",
            "Flask",
            "FastAPI",
            "Express",
            "Express.js",
            "Spring",
            "Spring Boot",
            "ASP.NET",
            ".NET",
            "Ruby on Rails",
            "Laravel",
            "Symfony",
            "NestJS",
            "Remix",
            "Gatsby",
            "jQuery",
            "Bootstrap",
            "Tailwind CSS",
            "Streamlit",
            "Gradio",
            "Jinja",
            "Starlette",
            "Uvicorn",
        }
    ),
    "ml_ai": frozenset(
        {
            "TensorFlow",
            "PyTorch",
            "Keras",
            "scikit-learn",
            "Hugging Face",
            "Transformers",
            "spaCy",
            "NLTK",
            "OpenCV",
            "Pandas",
            "NumPy",
            "SciPy",
            "Matplotlib",
            "Seaborn",
            "XGBoost",
            "LightGBM",
            "CatBoost",
            "LangChain",
            "LlamaIndex",
            "OpenAI",
            "BERT",
            "GPT",
            "YOLO",
            "MLflow",
            "Weights & Biases",
            "Jupyter",
            "Spark",
            "PySpark",
            "Hadoop",
            "Airflow",
            "Kubeflow",
            "ONNX",
            "Sentence Transformers",
            "ChromaDB",
            "FAISS",
            "Pinecone",
            "Weaviate",
            "RAG",
            "NLP",
            "Computer Vision",
            "Deep Learning",
            "Machine Learning",
            "LLM",
            "scikit-learn",
        }
    ),
    "databases": frozenset(
        {
            "PostgreSQL",
            "MySQL",
            "MariaDB",
            "SQLite",
            "MongoDB",
            "Redis",
            "Elasticsearch",
            "Cassandra",
            "DynamoDB",
            "Oracle",
            "SQL Server",
            "Snowflake",
            "BigQuery",
            "Redshift",
            "Neo4j",
            "CouchDB",
            "Firestore",
            "Supabase",
            "ClickHouse",
            "InfluxDB",
            "Prisma",
            "SQLAlchemy",
        }
    ),
    "cloud": frozenset(
        {
            "AWS",
            "Azure",
            "GCP",
            "Google Cloud",
            "Amazon Web Services",
            "Microsoft Azure",
            "S3",
            "EC2",
            "Lambda",
            "Cloud Functions",
            "Heroku",
            "DigitalOcean",
            "Vercel",
            "Netlify",
            "Cloudflare",
            "IBM Cloud",
            "Oracle Cloud",
            "SageMaker",
            "Vertex AI",
            "Cloud Run",
        }
    ),
    "devops": frozenset(
        {
            "Docker",
            "Kubernetes",
            "Jenkins",
            "GitHub Actions",
            "GitLab CI",
            "CircleCI",
            "Terraform",
            "Ansible",
            "Chef",
            "Puppet",
            "Prometheus",
            "Grafana",
            "Nginx",
            "Apache",
            "Linux",
            "Git",
            "CI/CD",
            "Helm",
            "ArgoCD",
            "Vault",
            "Packer",
            "Vagrant",
            "Maven",
            "Gradle",
            "npm",
            "Yarn",
            "pnpm",
            "Webpack",
            "Vite",
            "GitHub",
            "GitLab",
            "Bitbucket",
        }
    ),
}

SKILL_ALIASES: Final[dict[str, str]] = {
    "js": "JavaScript",
    "ts": "TypeScript",
    "nodejs": "Node.js",
    "node.js": "Node.js",
    "node": "Node.js",
    "golang": "Go",
    "csharp": "C#",
    "cpp": "C++",
    "c plus plus": "C++",
    "objective c": "Objective-C",
    "vuejs": "Vue.js",
    "vue.js": "Vue.js",
    "nextjs": "Next.js",
    "next.js": "Next.js",
    "expressjs": "Express.js",
    "express.js": "Express.js",
    "reactjs": "React",
    "rails": "Ruby on Rails",
    "ror": "Ruby on Rails",
    "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "tf": "TensorFlow",
    "huggingface": "Hugging Face",
    "hugging face": "Hugging Face",
    "spacy": "spaCy",
    "pytorch": "PyTorch",
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "mongo": "MongoDB",
    "mssql": "SQL Server",
    "microsoft sql server": "SQL Server",
    "gcp": "GCP",
    "google cloud platform": "GCP",
    "google cloud": "Google Cloud",
    "amazon web services": "AWS",
    "k8s": "Kubernetes",
    "ci cd": "CI/CD",
    "github actions": "GitHub Actions",
    "gitlab ci": "GitLab CI",
    "tailwind": "Tailwind CSS",
    "tailwindcss": "Tailwind CSS",
    "machine learning": "Machine Learning",
    "deep learning": "Deep Learning",
    "computer vision": "Computer Vision",
    "natural language processing": "NLP",
    "llms": "LLM",
    "sentence-transformers": "Sentence Transformers",
    "chromadb": "ChromaDB",
    "fastapi": "FastAPI",
}

_MONTHS: Final[dict[str, int]] = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

_MONTH_ALT: Final[str] = "|".join(sorted(_MONTHS.keys(), key=len, reverse=True))

_EMAIL_RE: Final[re.Pattern[str]] = re.compile(
    r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b",
    re.IGNORECASE,
)
_PHONE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:(?:\+|00)\d{1,3}[\s\-.]*)?(?:\(?\d{2,4}\)?[\s\-.]*){2,4}\d{2,4}"
)
_DATE_RANGE_RE: Final[re.Pattern[str]] = re.compile(
    rf"""
    (?:
        (?P<start_month>{_MONTH_ALT})\.?\s+
    )?
    (?:
        (?P<start_md>\d{{1,2}})[/\-]
    )?
    (?P<start_year>(?:19|20)\d{{2}})
    \s*(?:[\-–—]|to|until|thru|through)\s*
    (?:
        (?P<end_month>{_MONTH_ALT})\.?\s+
    )?
    (?:
        (?P<end_md>\d{{1,2}})[/\-]
    )?
    (?P<end_year>(?:19|20)\d{{2}}|present|current|now|today)
    """,
    re.IGNORECASE | re.VERBOSE,
)
_NAME_LINE_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}$"
)


@lru_cache(maxsize=1)
def load_nlp() -> Language:
    """Load and cache the spaCy English pipeline."""
    return spacy.load(SPACY_MODEL)


def extract_skills(text: str) -> list[str]:
    """Return taxonomy skills found via case-insensitive token/phrase matching."""
    if not text or not text.strip():
        return []

    haystack: str = text
    found: list[str] = []
    seen: set[str] = set()

    for skill, pattern in _skill_patterns():
        if pattern.search(haystack) is None:
            continue
        canonical: str = _canonicalize_skill(skill)
        key: str = canonical.casefold()
        if key in seen:
            continue
        seen.add(key)
        found.append(canonical)

    return found


def extract_contact_info(text: str) -> dict[str, str | None]:
    """Extract email, phone, and candidate name from resume text."""
    contact: dict[str, str | None] = {"email": None, "phone": None, "name": None}
    if not text or not text.strip():
        return contact

    email_match: re.Match[str] | None = _EMAIL_RE.search(text)
    if email_match is not None:
        contact["email"] = email_match.group(0)

    contact["phone"] = _first_valid_phone(text)
    contact["name"] = _extract_candidate_name(text, contact["email"])
    return contact


def _first_valid_phone(text: str) -> str | None:
    date_ranges: list[tuple[int, int]] = [
        match.span() for match in _DATE_RANGE_RE.finditer(text)
    ]
    for match in _PHONE_RE.finditer(text):
        digit_count: int = sum(character.isdigit() for character in match.group(0))
        if not 7 <= digit_count <= 15:
            continue
        if any(
            match.start() < range_end and match.end() > range_start
            for range_start, range_end in date_ranges
        ):
            continue
        return match.group(0).strip()
    return None


def _extract_candidate_name(text: str, email: str | None) -> str | None:
    for line in text.splitlines()[:5]:
        candidate: str = line.strip()
        if email is not None and email in candidate:
            continue
        if _NAME_LINE_RE.fullmatch(candidate):
            return candidate

    document = load_nlp()(text)
    for entity in document.ents:
        candidate = entity.text.strip()
        if entity.label_ == "PERSON" and (email is None or email not in candidate):
            return candidate
    return None


def estimate_experience_years(experience_text: str) -> float:
    """Estimate years of experience from year/month date ranges in text."""
    if not experience_text or not experience_text.strip():
        return 0.0

    intervals: list[tuple[date, date]] = []
    today: date = date.today()

    for match in _DATE_RANGE_RE.finditer(experience_text):
        start: date | None = _parse_bound(
            year_text=match.group("start_year"),
            month_text=match.group("start_month"),
            md_text=match.group("start_md"),
            is_end=False,
            today=today,
        )
        end: date | None = _parse_bound(
            year_text=match.group("end_year"),
            month_text=match.group("end_month"),
            md_text=match.group("end_md"),
            is_end=True,
            today=today,
        )
        if start is None or end is None or end < start:
            continue
        intervals.append((start, end))

    if not intervals:
        return 0.0

    merged: list[tuple[date, date]] = _merge_intervals(intervals)
    total_days: int = sum((finish - start).days + 1 for start, finish in merged)
    years: float = total_days / 365.25
    return round(min(years, 50.0), 1)

import re
import re
from typing import Any

def _canonicalize_skill(skill: str) -> str:
    """Map skill aliases or variations to their standard canonical name."""
    s = skill.strip()
    # Check if SKILL_ALIASES exists, else return the skill itself
    if "SKILL_ALIASES" in globals() and s in SKILL_ALIASES:
        return SKILL_ALIASES[s]
    return s

def _skill_to_pattern(skill: str) -> Any:
    """Compile skill string into a case-insensitive word-boundary regex pattern."""
    escaped = re.escape(skill.strip())
    return re.compile(rf"\b{escaped}\b", re.IGNORECASE)

def _skill_to_pattern(skill: str) -> re.Pattern:
    """Compile skill string into a case-insensitive word-boundary regex pattern."""
    escaped = re.escape(skill.strip())
    return re.compile(rf"\b{escaped}\b", re.IGNORECASE)

def _skill_patterns() -> list[tuple[str, re.Pattern[str]]]:
    phrases: set[str] = set()
    for group in SKILL_TAXONOMY.values():
        phrases.update(group)
    phrases.update(SKILL_ALIASES.keys())
    ordered: list[str] = sorted(phrases, key=len, reverse=True)
    return [(skill, _skill_to_pattern(skill)) for skill in ordered]


@lru_cache(maxsize=1)
def _compiled_skill_patterns() -> tuple[tuple[str, re.Pattern[str]], ...]:
    return tuple(_skill_patterns())


def extract_skills(text: str) -> list[str]:  # noqa: F811
    if not text or not text.strip():
        return []

    found: list[str] = []
    seen: set[str] = set()
    for skill, pattern in _compiled_skill_patterns():
        if pattern.search(text) is None:
            continue
        canonical: str = _canonicalize_skill(skill)
        key: str = canonical.casefold()
        if key in seen:
            continue
        seen.add(key)
        found.append(canonical)
    return found
