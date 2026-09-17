from __future__ import annotations
import os
import re
import json
import queue
import faiss
import threading
import time
import numpy as np
from typing import Literal
from collections import OrderedDict, Counter
from datetime import datetime
from sentence_transformers import SentenceTransformer

"""
Hybrid Memory System for Conversational AI / Agents
==================================================
Combines FAISS vector similarity search with BM25 keyword matching (Hybrid Search),
includes automatic entity extraction, decay-based importance ranking, and background 
thread persistence.
"""

# ══════════════════════════════════════════════════════════════════════════════
# Configuration & Hyperparameters
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_THRESHOLD = 0.60  # Base similarity threshold for search filtering
DEDUP_THRESHOLD = 0.92  # Cosine similarity threshold to drop duplicate memories
DEFAULT_K = 6  # Default number of memories to retrieve
ENTITY_BOOST = 1.20  # Score multiplier for extracted entity memories
IMPORTANCE_DECAY = 0.95  # Daily decay rate applied to memory importance scores
MAX_MEMORIES = 2000  # Maximum number of stored memory entries
DEFAULT_TTL_DAYS = 90  # Default Time-To-Live for raw conversation turns (days)
BM25_WEIGHT = 0.35  # Weight of BM25 lexical score in hybrid fusion
VECTOR_WEIGHT = 0.65  # Weight of FAISS vector score in hybrid fusion
EMBED_CACHE_SIZE = 512  # Max capacity for the LRU embedding cache
RECENCY_WEIGHT = 0.15  # Weight factor applied to recency/decay score boosting

# ══════════════════════════════════════════════════════════════════════════════
# Multilingual Stopwords & Question Signals
# ══════════════════════════════════════════════════════════════════════════════

_STOP_WORDS: set[str] = {
    # ── Vietnamese ────────────────────────────────────────────────────────────
    "đâu", "gì", "ai", "nào", "không", "chưa", "sao", "thế", "vậy",
    "rồi", "được", "có", "là", "ở", "tại", "của", "cho", "với",
    "một", "các", "những", "này", "đó", "kia", "cũng", "đã", "sẽ",
    # ── English ───────────────────────────────────────────────────────────────
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "what", "who", "where", "when", "why", "how", "which", "whose",
    "yes", "no", "not", "and", "or", "of", "to", "in", "on", "at",
    "for", "with", "from", "by", "this", "that", "these", "those",
    # ── Chinese ───────────────────────────────────────────────────────────────
    "的", "了", "是", "在", "我", "你", "他", "她", "它", "们",
    "什么", "哪里", "哪儿", "谁", "怎么", "怎样", "为什么", "吗", "呢", "吧",
    "这", "那", "有", "没", "不", "和", "与", "或", "也", "就",
    # ── Japanese ──────────────────────────────────────────────────────────────
    "の", "は", "が", "を", "に", "で", "と", "も", "から", "まで",
    "です", "ます", "ある", "いる", "なる",
    "何", "どこ", "誰", "いつ", "なぜ", "どう", "どの",
    "これ", "それ", "あれ", "この", "その", "あの",
    "か", "ね", "よ", "な",
    # ── Korean ────────────────────────────────────────────────────────────────
    "은", "는", "이", "가", "을", "를", "에", "의", "와", "과",
    "있다", "없다", "하다",
    "무엇", "어디", "누구", "언제", "왜", "어떻게", "어느",
    "이것", "그것", "저것",
    # ── French ────────────────────────────────────────────────────────────────
    "le", "la", "les", "un", "une", "des", "de", "du",
    "est", "sont", "être", "été",
    "quoi", "qui", "où", "quand", "pourquoi", "comment", "quel", "quelle",
    "oui", "non", "pas", "et", "ou", "ce", "cette", "ces",
    # ── Spanish ───────────────────────────────────────────────────────────────
    "el", "los", "las", "unos", "unas",
    "es", "son", "ser", "estar",
    "qué", "quien", "quién", "dónde", "cuándo", "cómo",
    "sí", "y", "o", "este", "esta", "estos", "estas",
    # ── German ────────────────────────────────────────────────────────────────
    "der", "die", "das", "den", "dem", "des", "ein", "eine",
    "ist", "sind", "sein", "war", "waren",
    "was", "wer", "wo", "wann", "warum", "wie", "welche",
    "ja", "nein", "nicht", "und", "oder", "dieser", "diese", "dieses",
    # ── Russian ───────────────────────────────────────────────────────────────
    "и", "в", "не", "на", "что", "я", "с", "он", "она", "оно",
    "это", "как", "где", "когда", "почему", "кто", "да", "нет",
    # ── Portuguese ────────────────────────────────────────────────────────────
    "é", "são", "ser", "estar",
    "quem", "onde", "quando", "como",
    "sim", "não", "e", "ou",
}

_QUESTION_SIGNALS: set[str] = {
    "?", "？",
    "gì", "nào", "đâu", "ai", "chưa", "không", "sao", "thế nào",
    "what", "where", "who", "when", "why", "how", "which",
    "什么", "哪里", "哪儿", "谁", "怎么", "为什么", "吗", "呢",
    "何", "どこ", "誰", "いつ", "なぜ", "どう", "どの",
    "무엇", "어디", "누구", "언제", "왜", "어떻게",
    "quoi", "qui", "où", "quand", "pourquoi", "comment",
    "qué", "dónde", "cuándo", "cómo",
    "was", "wer", "wo", "wann", "warum", "wie",
    "что", "где", "когда", "почему", "кто", "как",
    "onde", "quando", "como",
}

_FILLER_SUFFIX = re.compile(
    r"(?:\s+(?:"
    r"thì|à|ạ|nhé|nha|ha|hả|thôi|vậy|đó|đây|ơi|ừ|uh|uhm"
    r"|please|thanks|thank you|ok|okay"
    r"|吧|呢|啊|呀|哦|嗯"
    r"|ね|よ|な|か"
    r"|요|네"
    r"|s'il vous plaît|merci"
    r"|por favor|gracias"
    r"|bitte|danke"
    r"|пожалуйста|спасибо"
    r"))$",
    re.IGNORECASE,
)

# ══════════════════════════════════════════════════════════════════════════════
# Multilingual Entity Extraction Patterns
# ══════════════════════════════════════════════════════════════════════════════

_ENTITY_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    # ══ Identity ══════════════════════════════════════════════════════════════
    (re.compile(r"(?:tên (?:của )?(?:tôi|mình|ta) là|tôi tên là|mọi người gọi tôi là) (?!(?:gì|ai|nào))(.+)"),
     "User name: {0}", "identity"),
    (re.compile(r"(?:tôi|mình|ta) (?:năm nay|bây giờ|hiện tại)? ?(\d{1,3}) tuổi"), "Age: {0}", "identity"),
    (re.compile(r"(?:tôi|mình|ta) sinh (?:năm|ngày) (.+)"), "Date of birth: {0}", "identity"),
    (re.compile(r"(?:tôi|mình|ta) (?:là|đang là) (nam|nữ|boy|girl|con trai|con gái)"), "Gender: {0}", "identity"),
    (re.compile(r"(?:my name is|i am called|i'm called|call me|you can call me)\s+(?!(?:what|who|which))(.+)", re.I),
     "User name: {0}", "identity"),
    (re.compile(r"(?:i am|i'm)\s+(\d{1,3})\s+(?:years?\s+old|yo)", re.I), "Age: {0}", "identity"),
    (re.compile(r"(?:my birthday is|i was born (?:on|in))\s+(.+)", re.I), "Date of birth: {0}", "identity"),
    (re.compile(r"(?:i am|i'm)\s+(male|female|a man|a woman|a boy|a girl)", re.I), "Gender: {0}", "identity"),
    (re.compile(r"(?:我叫|我的名字是|我的名字叫)\s*(.+)"), "User name: {0}", "identity"),
    (re.compile(r"我(?:今年)?\s*(\d{1,3})\s*岁"), "Age: {0}", "identity"),
    (re.compile(r"我(?:是)\s*(男|女)(?:生|的)?"), "Gender: {0}", "identity"),
    (re.compile(r"(?:私の名前は|僕の名前は|俺の名前は|名前は)\s*(.+?)(?:です|だ)?$"), "User name: {0}", "identity"),
    (re.compile(r"私(?:は)?\s*(\d{1,3})\s*歳"), "Age: {0}", "identity"),
    (re.compile(r"(?:제 이름은|내 이름은|나의 이름은)\s*(.+?)(?:입니다|이에요|예요|야)?$"), "User name: {0}", "identity"),
    (re.compile(r"(?:저는|나는)\s*(\d{1,3})\s*살"), "Age: {0}", "identity"),
    (re.compile(r"(?:je m'appelle|mon nom est)\s+(.+)", re.I), "User name: {0}", "identity"),
    (re.compile(r"j'ai\s+(\d{1,3})\s+ans", re.I), "Age: {0}", "identity"),
    (re.compile(r"(?:me llamo|mi nombre es)\s+(.+)", re.I), "User name: {0}", "identity"),
    (re.compile(r"tengo\s+(\d{1,3})\s+años", re.I), "Age: {0}", "identity"),
    (re.compile(r"(?:ich heiße|mein name ist)\s+(.+)", re.I), "User name: {0}", "identity"),
    (re.compile(r"ich bin\s+(\d{1,3})\s+jahre alt", re.I), "Age: {0}", "identity"),

    # ══ Location ══════════════════════════════════════════════════════════════
    (re.compile(r"(?:tôi|mình|ta) (?:sống|ở|đang ở|đang sống|sinh sống) (?:tại|ở) (?!(?:đâu|nào))(.+)"),
     "Location: {0}", "location"),
    (re.compile(r"(?:tôi|mình|ta) (?:quê|quê hương|quê ở|người) (?!(?:đâu|nào|gì))(.+)"), "Hometown: {0}", "location"),
    (re.compile(r"i (?:live|am living|stay|am staying|reside)\s+(?:in|at)\s+(.+)", re.I), "Location: {0}", "location"),
    (re.compile(r"i(?:'m| am) from\s+(.+)", re.I), "Hometown: {0}", "location"),
    (re.compile(r"我住在\s*(.+)"), "Location: {0}", "location"),
    (re.compile(r"我来自\s*(.+)"), "Hometown: {0}", "location"),
    (re.compile(r"我是\s*(.+?)人"), "Hometown: {0}", "location"),
    (re.compile(r"私(?:は)?\s*(.+?)(?:に住んでいます|に住む)"), "Location: {0}", "location"),
    (re.compile(r"私(?:は)?\s*(.+?)(?:出身です|の出身)"), "Hometown: {0}", "location"),
    (re.compile(r"(?:저는|나는)\s*(.+?)(?:에 살아요|에 삽니다)"), "Location: {0}", "location"),
    (re.compile(r"(?:저는|나는)\s*(.+?)(?:출신입니다|출신이에요)"), "Hometown: {0}", "location"),
    (re.compile(r"(?:j'habite|je vis)\s+(?:à|en|au|aux)\s+(.+)", re.I), "Location: {0}", "location"),
    (re.compile(r"(?:vivo|resido)\s+en\s+(.+)", re.I), "Location: {0}", "location"),
    (re.compile(r"ich (?:wohne|lebe)\s+in\s+(.+)", re.I), "Location: {0}", "location"),

    # ══ Career / Education ════════════════════════════════════════════════════
    (re.compile(r"(?:tôi|ta|mình) (?:làm nghề|làm công việc|đang làm|làm) (?!(?:gì|ai|nào|người|không))(.+)"),
     "Occupation: {0}", "career"),
    (re.compile(r"(?:tôi|ta|mình) (?:học|đang học) (?:tại|ở|trường|ngành|chuyên ngành) (?!(?:đâu|nào|gì))(.+)"),
     "Education: {0}", "education"),
    (re.compile(r"(?:tôi|mình|ta) (?:tốt nghiệp|ra trường) (?:từ|ở|tại) (?!(?:đâu|nào))(.+)"), "Graduated from: {0}",
     "education"),
    (re.compile(r"i (?:work as|work at|am working as|am a)\s+(?:a |an )?(.+)", re.I), "Occupation: {0}", "career"),
    (re.compile(r"i (?:study|am studying|major in)\s+(?:at |in )?(.+)", re.I), "Education: {0}", "education"),
    (re.compile(r"i graduated from\s+(.+)", re.I), "Graduated from: {0}", "education"),
    (re.compile(r"我(?:在做|工作是|职业是)\s*(.+)"), "Occupation: {0}", "career"),
    (re.compile(r"我在\s*(.+?)(?:上学|学习|读书)"), "Education: {0}", "education"),
    (re.compile(r"私(?:は)?\s*(.+?)(?:として働いています|の仕事をしています)"), "Occupation: {0}", "career"),
    (re.compile(r"私(?:は)?\s*(.+?)(?:を専攻しています|に通っています)"), "Education: {0}", "education"),
    (re.compile(r"je travaille (?:comme|en tant que)\s+(.+)", re.I), "Occupation: {0}", "career"),
    (re.compile(r"j'étudie\s+(?:à |en )?(.+)", re.I), "Education: {0}", "education"),
    (re.compile(r"trabajo (?:como|de)\s+(.+)", re.I), "Occupation: {0}", "career"),
    (re.compile(r"estudio\s+(?:en )?(.+)", re.I), "Education: {0}", "education"),
    (re.compile(r"ich arbeite als\s+(.+)", re.I), "Occupation: {0}", "career"),
    (re.compile(r"ich studiere\s+(.+)", re.I), "Education: {0}", "education"),

    # ══ Preferences ═══════════════════════════════════════════════════════════
    (re.compile(r"(?:tôi|ta|mình) (?:thích|yêu thích|rất thích|mê) (?!(?:gì|ai|nào))(.+)"), "Likes: {0}", "preference"),
    (re.compile(r"(?:tôi|ta|mình) (?:ghét|không thích|không muốn|tránh) (?!(?:gì|ai|nào))(.+)"), "Dislikes: {0}",
     "preference"),
    (re.compile(r"(?:tôi|ta|mình) (?:hay|thường|thường xuyên|thường hay) (.+)"), "Habit: {0}", "habit"),
    (re.compile(r"i (?:like|love|enjoy|prefer|am into)\s+(.+)", re.I), "Likes: {0}", "preference"),
    (re.compile(r"i (?:hate|dislike|don't like|do not like|avoid)\s+(.+)", re.I), "Dislikes: {0}", "preference"),
    (re.compile(r"i (?:usually|often|always|tend to)\s+(.+)", re.I), "Habit: {0}", "habit"),
    (re.compile(r"我(?:喜欢|爱|偏好)\s*(.+)"), "Likes: {0}", "preference"),
    (re.compile(r"我(?:讨厌|不喜欢)\s*(.+)"), "Dislikes: {0}", "preference"),
    (re.compile(r"私(?:は)?\s*(.+?)(?:が好きです|が好き)"), "Likes: {0}", "preference"),
    (re.compile(r"私(?:は)?\s*(.+?)(?:が嫌いです|が嫌い)"), "Dislikes: {0}", "preference"),
    (re.compile(r"(?:저는|나는)\s*(.+?)(?:을 좋아해요|를 좋아해요|좋아합니다)"), "Likes: {0}", "preference"),
    (re.compile(r"(?:저는|나는)\s*(.+?)(?:을 싫어해요|를 싫어해요|싫어합니다)"), "Dislikes: {0}", "preference"),
    (re.compile(r"j'(?:aime|adore|préfère)\s+(.+)", re.I), "Likes: {0}", "preference"),
    (re.compile(r"je (?:déteste|n'aime pas)\s+(.+)", re.I), "Dislikes: {0}", "preference"),
    (re.compile(r"me (?:gusta|encanta|prefiero)\s+(.+)", re.I), "Likes: {0}", "preference"),
    (re.compile(r"(?:odio|no me gusta)\s+(.+)", re.I), "Dislikes: {0}", "preference"),
    (re.compile(r"ich (?:mag|liebe|bevorzuge)\s+(.+)", re.I), "Likes: {0}", "preference"),
    (re.compile(r"ich (?:hasse|mag nicht)\s+(.+)", re.I), "Dislikes: {0}", "preference"),

    # ══ Goals / Issues ════════════════════════════════════════════════════════
    (re.compile(r"(?:tôi|ta|mình) (?:muốn|cần|đang cần|đang muốn|đang tìm) (?!(?:gì|nào|ai))(.+)"), "Goal: {0}",
     "goal"),
    (re.compile(r"(?:tôi|ta|mình) (?:đang gặp|gặp phải|bị|đang bị) (?:vấn đề|lỗi|khó khăn|tình trạng) (.+)"),
     "Issue: {0}", "issue"),
    (re.compile(r"i (?:want|need|am looking for|would like|am trying to)\s+(.+)", re.I), "Goal: {0}", "goal"),
    (re.compile(r"i (?:have|am having|am facing)\s+(?:a |an )?(?:problem|issue|trouble|bug)\s+(?:with\s+)?(.+)", re.I),
     "Issue: {0}", "issue"),
    (re.compile(r"我(?:想要|想|需要|正在找)\s*(.+)"), "Goal: {0}", "goal"),
    (re.compile(r"我(?:遇到了|有)\s*(.+?)(?:问题|麻烦)"), "Issue: {0}", "issue"),
    (re.compile(r"私(?:は)?\s*(.+?)(?:が欲しい|が必要|を探しています)"), "Goal: {0}", "goal"),
    (re.compile(r"(?:저는|나는)\s*(.+?)(?:을 원해요|를 원해요|이 필요해요|가 필요해요)"), "Goal: {0}", "goal"),
    (re.compile(r"je (?:veux|voudrais|cherche|ai besoin de)\s+(.+)", re.I), "Goal: {0}", "goal"),
    (re.compile(r"(?:quiero|necesito|busco)\s+(.+)", re.I), "Goal: {0}", "goal"),
    (re.compile(r"ich (?:möchte|will|brauche|suche)\s+(.+)", re.I), "Goal: {0}", "goal"),

    # ══ Relationships ═════════════════════════════════════════════════════════
    (re.compile(r"(?:vợ|chồng|bạn gái|bạn trai|con|bố|mẹ|anh|chị|em) (?:của tôi|tôi|mình) (?:là|tên là|tên) (.+)"),
     "Relationship: {0}", "relationship"),
    (re.compile(
        r"my (?:wife|husband|girlfriend|boyfriend|son|daughter|father|mother|brother|sister)(?:'s name)? is\s+(.+)",
        re.I), "Relationship: {0}", "relationship"),
    (re.compile(r"我的(?:妻子|丈夫|女朋友|男朋友|儿子|女儿|爸爸|妈妈|哥哥|姐姐|弟弟|妹妹)(?:叫|是)\s*(.+)"),
     "Relationship: {0}", "relationship"),
    (re.compile(r"私の(?:妻|夫|彼女|彼氏|息子|娘|父|母|兄|姉|弟|妹)(?:の名前)?は\s*(.+?)(?:です)?$"),
     "Relationship: {0}", "relationship"),
]


def _extract_entities(text: str) -> list[tuple[str, str]]:
    """Extract structured entities (e.g., Name, Age, Location) using regex pattern matching.

    Args:
        text (str): Input query or conversational turn.

    Returns:
        list[tuple[str, str]]: List of formatted entity strings and their sub-types.
    """
    lower = text.lower().strip().rstrip("?？").strip()
    if not lower:
        return []
    out: list[tuple[str, str]] = []
    for pattern, template, etype in _ENTITY_PATTERNS:
        m = pattern.search(lower)
        if not m:
            continue
        val = m.groups()[-1].strip()
        if len(val) < 2 or val in _STOP_WORDS or re.fullmatch(r"[\d\W]+", val):
            continue
        val = _FILLER_SUFFIX.sub("", val).strip()
        if len(val) < 2:
            continue
        out.append((template.format(val), etype))
    return out


def _is_question(text: str) -> bool:
    """Check if the text contains explicit question marks or signal words."""
    if "?" in text or "？" in text:
        return True
    lower = text.lower()
    return any(s in lower for s in _QUESTION_SIGNALS)


# ══════════════════════════════════════════════════════════════════════════════
# BM25 Lite — Inverted Index Implementation
# ══════════════════════════════════════════════════════════════════════════════

# Regular expression tokenizer supporting CJK, Latin, Cyrillic, and Arabic alphabets
_TOKEN_RE = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf"
    r"\u3040-\u309f\u30a0-\u30ff"
    r"\uac00-\ud7af\u1100-\u11ff"
    r"\u0e00-\u0e7f"
    r"]"
    r"|[^\W\d_\u4e00-\u9fff\u3400-\u4dbf\u3040-\u309f\u30a0-\u30ff"
    r"\uac00-\ud7af\u1100-\u11ff\u0e00-\u0e7f]+",
    re.UNICODE,
)


class _BM25:
    """Lightweight in-memory BM25 lexical retriever with inverted indexing."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._N: int = 0
        self._total_len: int = 0
        self._df: dict[str, int] = {}
        self._postings: dict[str, dict[int, int]] = {}
        self._doc_len: list[int] = []

    @staticmethod
    def _tok(text: str) -> list[str]:
        return _TOKEN_RE.findall(text.lower())

    @property
    def N(self) -> int:
        return self._N

    @property
    def _avgdl(self) -> float:
        return self._total_len / self._N if self._N else 1.0

    def fit(self, texts: list[str]):
        """Rebuild the entire inverted index from a given list of document texts."""
        self._N = 0
        self._total_len = 0
        self._df = {}
        self._postings = {}
        self._doc_len = []
        for t in texts:
            self.add(t)

    def add(self, text: str):
        """Add a single text entry into the inverted index."""
        toks = self._tok(text)
        doc_id = self._N
        self._N += 1
        dl = len(toks)
        self._doc_len.append(dl)
        self._total_len += dl
        if dl == 0:
            return
        for tok, c in Counter(toks).items():
            self._df[tok] = self._df.get(tok, 0) + 1
            post = self._postings.get(tok)
            if post is None:
                self._postings[tok] = {doc_id: c}
            else:
                post[doc_id] = c

    def score(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Calculate BM25 relevancy scores for a query across indexed documents."""
        if self._N == 0:
            return []
        q_toks = set(self._tok(query))
        if not q_toks:
            return []

        N, k1, b = self._N, self.k1, self.b
        avgdl = self._avgdl
        scores: dict[int, float] = {}

        for tok in q_toks:
            postings = self._postings.get(tok)
            if not postings:
                continue
            df = self._df[tok]
            # Pre-compute Inverse Document Frequency (IDF) outside inner document loop
            idf = float(np.log((N - df + 0.5) / (df + 0.5) + 1.0))

            for doc_id, tf in postings.items():
                dl = self._doc_len[doc_id]
                numer = tf * (k1 + 1.0)
                denom = tf + k1 * (1.0 - b + b * dl / avgdl)
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * (numer / denom)

        if not scores:
            return []
        if len(scores) > top_k:
            items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        else:
            items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return items


# ══════════════════════════════════════════════════════════════════════════════
# LRU Embedding Cache
# ══════════════════════════════════════════════════════════════════════════════

class _EmbedCache:
    """Least Recently Used (LRU) Cache for storing pre-calculated text embeddings."""

    def __init__(self, maxsize: int = EMBED_CACHE_SIZE):
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._max = maxsize

    def get(self, key: str) -> np.ndarray | None:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: str, val: np.ndarray):
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._max:
                self._cache.popitem(last=False)
            self._cache[key] = val

    def __len__(self) -> int:
        return len(self._cache)


# ══════════════════════════════════════════════════════════════════════════════
# Timestamp & Entry Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _to_timestamp(val) -> float | None:
    """Convert various timestamp input representations to standard Unix epoch float seconds."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return datetime.fromisoformat(val).timestamp()
    except Exception:
        return 0.0


def _to_iso(ts: float | None) -> str | None:
    """Convert Unix epoch float seconds into an ISO 8601 formatted date-time string."""
    if ts is None:
        return None
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def _make_entry(
        text: str,
        etype: Literal["entity", "raw"] = "raw",
        subtype: str = "",
        importance: float = 1.0,
        ttl_days: int | None = DEFAULT_TTL_DAYS,
) -> dict:
    """Create a structured dictionary representing a memory record."""
    now_ts = time.time()
    expire_at = None
    if ttl_days is not None and etype == "raw":
        expire_at = now_ts + (ttl_days * 86400.0)
    return {
        "text": text,
        "timestamp": now_ts,
        "last_accessed": now_ts,
        "access_count": 0,
        "type": etype,
        "subtype": subtype,
        "importance": round(importance, 4),
        "expire_at": expire_at,
    }


def _decay_score(entry: dict, now_ts: float) -> float:
    """Calculate the decay-adjusted importance score based on time elapsed since last access."""
    # Uses raw numerical timestamps to avoid datetime instantiation overhead
    days = (now_ts - entry.get("last_accessed", now_ts)) / 86400.0
    if days < 0:
        days = 0.0
    return entry.get("importance", 1.0) * (IMPORTANCE_DECAY ** days)


def _convert_for_save(texts: list[dict]) -> list[dict]:
    """Convert floating-point Unix timestamps back into human-readable ISO strings for JSON storage."""
    out = []
    for t in texts:
        new_t = t.copy()
        new_t["timestamp"] = _to_iso(new_t.get("timestamp"))
        new_t["last_accessed"] = _to_iso(new_t.get("last_accessed"))
        new_t["expire_at"] = _to_iso(new_t.get("expire_at"))
        out.append(new_t)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Async Save Worker Thread
# ══════════════════════════════════════════════════════════════════════════════

class _SaveWorker(threading.Thread):
    """Background daemon thread to persist FAISS index and JSON metadata asynchronously."""

    def __init__(self, index_file: str, data_file: str):
        super().__init__(daemon=True, name="MemorySaveWorker")
        self.index_file = index_file
        self.data_file = data_file
        self._q: queue.Queue = queue.Queue(maxsize=1)
        self.start()

    def enqueue(self, index: faiss.Index, texts: list[dict]):
        """Enqueue state data for saving. Overwrites unsaved non-processed items."""
        try:
            self._q.get_nowait()
        except queue.Empty:
            pass
        self._q.put((index, list(texts)))

    def run(self):
        while True:
            item = self._q.get()
            if item is None:
                break
            index, texts = item
            try:
                faiss.write_index(index, self.index_file)
                save_texts = _convert_for_save(texts)
                with open(self.data_file, "w", encoding="utf-8") as f:
                    json.dump(save_texts, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[MEM][SaveWorker] {e}")

    def stop(self):
        self._q.put(None)


# ══════════════════════════════════════════════════════════════════════════════
# Main Memory Engine Class
# ══════════════════════════════════════════════════════════════════════════════

class Memory:
    """Long-term memory module utilizing vector search (FAISS) and lexical search (BM25)."""

    def __init__(
            self,
            index_file: str = "memory.index",
            data_file: str = "memory.json",
            model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
            dimension: int = 384,
            auto_save: bool = True,
            ttl_days: int = DEFAULT_TTL_DAYS,
    ):
        self.index_file = index_file
        self.data_file = data_file
        self.dimension = dimension
        self.auto_save = auto_save
        self.ttl_days = ttl_days

        self._lock = threading.RLock()
        self._cache = _EmbedCache(EMBED_CACHE_SIZE)
        self._bm25 = _BM25()

        device = "cpu"
        print(f"[MEM] Loading '{model_name}' on {device} ...")
        self.model = SentenceTransformer(model_name, device=device)

        self.memory_texts: list[dict] = []
        self.index = faiss.IndexFlatIP(self.dimension)
        self._text_set: set[str] = set()

        self._load()
        self._worker = _SaveWorker(index_file, data_file) if auto_save else None

    # ── Internal Storage Methods ──────────────────────────────────────────────

    def _load(self):
        """Load index file and metadata from disk if available."""
        if not (os.path.exists(self.index_file) and os.path.exists(self.data_file)):
            print("[MEM] Fresh start.")
            return
        try:
            self.index = faiss.read_index(self.index_file)
            with open(self.data_file, "r", encoding="utf-8") as f:
                raw = json.load(f)

            self.memory_texts = []
            for m in raw:
                if isinstance(m, dict):
                    # Parse old ISO timestamp strings into Unix floats
                    m["timestamp"] = _to_timestamp(m.get("timestamp"))
                    m["last_accessed"] = _to_timestamp(m.get("last_accessed"))
                    m["expire_at"] = _to_timestamp(m.get("expire_at"))
                    self.memory_texts.append(m)
                else:
                    self.memory_texts.append(_make_entry(m))

            self._text_set = {e["text"] for e in self.memory_texts}
            self._prune_expired(silent=True)
            self._bm25.fit([e["text"] for e in self.memory_texts])
            print(f"[MEM] Loaded {len(self.memory_texts)} memories.")
        except Exception as e:
            print(f"[MEM] Load error ({e}), starting fresh.")
            self.index = faiss.IndexFlatIP(self.dimension)
            self.memory_texts = []
            self._text_set = set()

    def save(self):
        """Synchronously persist memory index and metadata to disk."""
        with self._lock:
            try:
                faiss.write_index(self.index, self.index_file)
                save_texts = _convert_for_save(self.memory_texts)
                with open(self.data_file, "w", encoding="utf-8") as f:
                    json.dump(save_texts, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[MEM] Save error: {e}")

    def _save_async(self):
        """Asynchronously trigger saving to background worker."""
        if self._worker:
            with self._lock:
                self._worker.enqueue(self.index, self.memory_texts)

    def close(self):
        """Safely save and terminate background thread workers."""
        self.save()
        if self._worker:
            self._worker.stop()

    # ── Embedding Methods ─────────────────────────────────────────────────────

    def _encode(self, text: str) -> np.ndarray:
        """Encode single string into a normalized dense embedding vector."""
        hit = self._cache.get(text)
        if hit is not None:
            return hit
        vec = self.model.encode([text], normalize_embeddings=True)
        arr = np.array(vec, dtype="float32")
        self._cache.put(text, arr)
        return arr

    def _encode_batch_cached(self, texts: list[str]) -> np.ndarray:
        """Perform batch embedding computation with cache utilization."""
        n = len(texts)
        if n == 0:
            return np.zeros((0, self.dimension), dtype="float32")
        result = np.zeros((n, self.dimension), dtype="float32")
        miss_idx: list[int] = []
        miss_txt: list[str] = []
        for i, t in enumerate(texts):
            hit = self._cache.get(t)
            if hit is not None:
                result[i] = hit[0]
            else:
                miss_idx.append(i)
                miss_txt.append(t)
        if miss_txt:
            vecs = self.model.encode(
                miss_txt, normalize_embeddings=True,
                batch_size=64, show_progress_bar=False,
            )
            vecs = np.asarray(vecs, dtype="float32")
            for j, i in enumerate(miss_idx):
                result[i] = vecs[j]
                self._cache.put(miss_txt[j], vecs[j:j + 1].copy())
        return result

    # ── Maintenance & Maintenance Operations ──────────────────────────────────

    def _rebuild_index(self):
        """Reconstruct both FAISS vector index and BM25 inverted index."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self._text_set = {e["text"] for e in self.memory_texts}
        if self.memory_texts:
            vecs = self._encode_batch_cached([e["text"] for e in self.memory_texts])
            self.index.add(vecs)
        self._bm25 = _BM25()
        for e in self.memory_texts:
            self._bm25.add(e["text"])

    def _prune_expired(self, silent: bool = False):
        """Purge raw memories whose Time-To-Live (TTL) has expired."""
        now_ts = time.time()
        prev = len(self.memory_texts)
        self.memory_texts = [
            e for e in self.memory_texts
            if not (e["type"] == "raw" and e.get("expire_at") and e["expire_at"] < now_ts)
        ]
        n = prev - len(self.memory_texts)
        if n > 0:
            self._rebuild_index()
            if not silent:
                print(f"[MEM] Pruned {n} expired entries.")

    def _prune_to_cap(self):
        """Enforce maximum capacity limits by purging low-priority decayed memories."""
        if len(self.memory_texts) <= MAX_MEMORIES:
            return

        now_ts = time.time()

        def _priority(e: dict) -> float:
            s = _decay_score(e, now_ts)
            if e["type"] == "entity":
                s *= 2.0
            return s

        self.memory_texts.sort(key=_priority, reverse=True)
        removed = len(self.memory_texts) - MAX_MEMORIES
        self.memory_texts = self.memory_texts[:MAX_MEMORIES]
        self._rebuild_index()
        print(f"[MEM] Cap: pruned {removed} entries, kept {len(self.memory_texts)}.")

    def _is_duplicate(self, vec: np.ndarray) -> bool:
        """Check if identical or highly similar memory vector exists in FAISS."""
        if self.index.ntotal == 0:
            return False
        D, _ = self.index.search(vec, min(3, self.index.ntotal))
        return bool(D[0].max() > DEDUP_THRESHOLD)

    def _add(
            self,
            text: str,
            etype: Literal["entity", "raw"] = "raw",
            subtype: str = "",
            importance: float = 1.0,
    ):
        """Internal worker function to add memory entry."""
        with self._lock:
            if etype == "entity":
                val = text.split(": ", 1)[-1].strip()
                if len(val) < 2 or val in _STOP_WORDS:
                    return

            if text in self._text_set:
                return

            vec = self._encode(text)
            if self._is_duplicate(vec):
                return

            ttl = None if etype == "entity" else self.ttl_days
            entry = _make_entry(text, etype, subtype, importance, ttl)
            self.index.add(vec)
            self.memory_texts.append(entry)
            self._text_set.add(text)
            self._bm25.add(text)

    # ── Public APIs: Interaction Turn Ingestion ──────────────────────────────

    def add_user_turn(self, user_text: str):
        """Extract structured entities and save raw user message turns.

        Args:
            user_text (str): Input text provided by user.
        """
        entities = _extract_entities(user_text)
        for formatted, subtype in entities:
            self._add(formatted, etype="entity", subtype=subtype, importance=2.0)

        if not _is_question(user_text) and len(user_text.split()) >= 3:
            self._add(
                f"User nói: {user_text}",
                etype="raw",
                importance=1.5 if entities else 1.0,
            )

        self._prune_to_cap()
        self._save_async()

    def add_assistant_turn(self, response_text: str):
        """Add assistant turn / output response text into raw memory.

        Args:
            response_text (str): Output text generated by AI assistant.
        """
        if len(response_text.split()) >= 5:
            self._add(
                f"Holo đã nói: {response_text[:300]}",
                etype="raw",
                importance=0.8,
            )
        self._save_async()

    # ── Public APIs: Retrieval and Search ─────────────────────────────────────

    def search(
            self,
            query: str,
            k: int = DEFAULT_K,
            threshold: float = DEFAULT_THRESHOLD,
            use_hybrid: bool = True,
    ) -> list[str]:
        """Perform hybrid search over stored memories using FAISS + BM25 scores.

        Args:
            query (str): The search input string.
            k (int): Maximum number of records to retrieve.
            threshold (float): Minimum score similarity threshold.
            use_hybrid (bool): Whether to fuse lexical BM25 scores with dense vectors.

        Returns:
            list[str]: Filtered list of matching memory text representations.
        """
        with self._lock:
            if not self.memory_texts:
                return []

            n = len(self.memory_texts)
            k_fetch = min(max(k * 4, 16), n)
            now_ts = time.time()

            # ── Vector Search (FAISS) ─────────────────────────────────────────
            vec = self._encode(query)
            D, I = self.index.search(vec, k_fetch)
            vec_scores: dict[int, float] = {
                int(idx): float(sim)
                for sim, idx in zip(D[0], I[0]) if idx >= 0
            }

            # ── Lexical Search (BM25) ─────────────────────────────────────────
            bm25_scores: dict[int, float] = {}
            if use_hybrid and self._bm25.N == n:
                raw_bm25 = self._bm25.score(query, top_k=k_fetch)
                if raw_bm25:
                    max_b = max(s for _, s in raw_bm25) or 1.0
                    bm25_scores = {idx: s / max_b for idx, s in raw_bm25}

            # ── Hybrid Fusion & Recency Decay ─────────────────────────────────
            candidates = set(vec_scores.keys()) | set(bm25_scores.keys())
            scored: list[tuple[int, float, float]] = []

            for idx in candidates:
                e = self.memory_texts[idx]
                if e.get("expire_at") and e["expire_at"] < now_ts:
                    continue

                vs = max(0.0, vec_scores.get(idx, 0.0))
                bs = bm25_scores.get(idx, 0.0)
                fused = VECTOR_WEIGHT * vs + BM25_WEIGHT * bs

                if e["type"] == "entity":
                    fused *= ENTITY_BOOST

                rec = _decay_score(e, now_ts)
                delta = max(-1.0, min(rec - 1.0, 1.0))
                fused *= 1.0 + RECENCY_WEIGHT * delta

                scored.append((idx, fused, vs))

            scored.sort(key=lambda x: x[1], reverse=True)

            # ── Post-Filtering and Result Generation ──────────────────────────
            results: list[str] = []
            seen: set[str] = set()

            for idx, fused, cos in scored:
                if len(results) >= k:
                    break
                e = self.memory_texts[idx]
                thr = threshold * 0.85 if e["type"] == "entity" else threshold
                if cos < thr and fused < 0.45:
                    continue

                raw_text = e["text"]
                if raw_text in seen:
                    continue
                seen.add(raw_text)

                display = raw_text
                for prefix in ("Holo đã nói: ", "User nói: "):
                    if display.startswith(prefix):
                        display = display[len(prefix):]
                        break

                results.append(display)
                e["last_accessed"] = now_ts
                e["access_count"] = e.get("access_count", 0) + 1

            return results

    # ── Utilities & Debugging ─────────────────────────────────────────────────

    def forget(self, keyword: str) -> int:
        """Delete memories containing a specific sub-string keyword.

        Args:
            keyword (str): Substring to match for deletion.

        Returns:
            int: Total count of deleted entries.
        """
        with self._lock:
            kw = keyword.lower()
            prev = len(self.memory_texts)
            self.memory_texts = [e for e in self.memory_texts if kw not in e["text"].lower()]
            n_del = prev - len(self.memory_texts)
            if n_del > 0:
                self._rebuild_index()
                self._save_async()
                print(f"[MEM] Forgot {n_del} entries matching '{keyword}'.")
            return n_del

    def boost_importance(self, keyword: str, delta: float = 0.5):
        """Increase the base importance factor of memories containing a specific string keyword."""
        with self._lock:
            kw = keyword.lower()
            for e in self.memory_texts:
                if kw in e["text"].lower():
                    e["importance"] = round(min(e.get("importance", 1.0) + delta, 5.0), 4)

    def stats(self) -> dict:
        """Return diagnostic metrics and summary counters regarding storage status."""
        with self._lock:
            now_ts = time.time()
            entities = [e for e in self.memory_texts if e["type"] == "entity"]
            raws = [e for e in self.memory_texts if e["type"] == "raw"]
            expired = sum(
                1 for e in raws
                if e.get("expire_at") and e["expire_at"] < now_ts
            )
            subtypes: dict[str, int] = {}
            for e in entities:
                st = e.get("subtype", "")
                if st:
                    subtypes[st] = subtypes.get(st, 0) + 1
            return {
                "total": len(self.memory_texts),
                "entity": len(entities),
                "raw": len(raws),
                "expired": expired,
                "entity_types": subtypes,
                "embed_cache": len(self._cache),
                "faiss_total": self.index.ntotal,
            }

    def debug_search(self, query: str, k: int = DEFAULT_K) -> list[dict]:
        """Return raw scored outputs with decay metrics for debugging search behavior."""
        with self._lock:
            if not self.memory_texts:
                return []
            n = len(self.memory_texts)
            k_fetch = min(max(k * 4, 16), n)
            vec = self._encode(query)
            D, I = self.index.search(vec, k_fetch)
            bm25 = dict(self._bm25.score(query, top_k=k_fetch))
            out = []
            now_ts = time.time()
            for cos, idx in zip(D[0], I[0]):
                if idx < 0:
                    continue
                e = self.memory_texts[idx]
                out.append({
                    "text": e["text"],
                    "type": e["type"],
                    "subtype": e.get("subtype", ""),
                    "cosine": round(float(cos), 4),
                    "bm25": round(bm25.get(int(idx), 0.0), 4),
                    "importance": e.get("importance", 1.0),
                    "decay": round(_decay_score(e, now_ts), 4),
                    "accesses": e.get("access_count", 0),
                })
            return out[:k]

    def clear(self):
        """Wipe all stored data completely and reset indexes."""
        with self._lock:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.memory_texts = []
            self._text_set = set()
            self._bm25 = _BM25()
            self._cache = _EmbedCache(EMBED_CACHE_SIZE)
            self.save()
            print("[MEM] Cleared.")