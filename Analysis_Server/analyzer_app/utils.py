import re
import urllib.parse as up

import requests
import torch
import unicodedata
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

tok = TweetTokenizer(strip_handles=True, reduce_len=True)
stop_en = set(stopwords.words('english'))

SAFE_CHARS = "-._~:/?#[]@!$&'()*+,;=%"
ALLOWED    = set("abcdefghijklmnopqrstuvwxyz0123456789" + SAFE_CHARS)

def canonical_redirect(url: str, timeout: int = 3) -> str:
    try:
        r = requests.head(url, allow_redirects=True, timeout=timeout)
        return r.url.lower()
    except requests.RequestException:
        return url.lower()


def clean_url(raw: str) -> str:
    raw = raw.strip()
    if not re.match(r'^[a-z]+://', raw):
        raw = 'http://' + raw

    u = up.unquote(raw).lower()
    u = re.sub(r'^(https?://)www\.', r'\1', u)
    u = re.sub(r':(80|443)(?=/)', '', u)
    u = canonical_redirect(u, timeout=10)
    if u.endswith('/') and u.count('/') > 2:
        u = u[:-1]
    return ''.join(c if c in ALLOWED else '_' for c in u)


def strip_html(text: str) -> str:
    text = BeautifulSoup(text, 'html.parser').get_text(separator=' ', strip=True)
    return re.sub(r"&[a-z]+;", ' ', text)


def mask_entities(text: str) -> str:
    return re.sub(r'http[s]?://\S+', ' ___URL___ ', text)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize('NFKD', text)
    return text.lower()


def remove_punctuation(text: str) -> str:
    return re.sub(r"[^a-z0-9_ ]+", ' ', text)


def tokenize_and_remove_stopwords(text: str) -> list[str]:
    tokens = tok.tokenize(text)
    return [t for t in tokens if t not in stop_en]


def clean_text_pipeline(raw: str) -> list[str]:
    text = strip_html(raw)
    text = mask_entities(text)
    text = normalize_text(text)
    text = remove_punctuation(text)
    return tokenize_and_remove_stopwords(text)

from .text_vocab import word2idx

def preprocess_message(text: str, max_len: int = 200):
    tokens = clean_text_pipeline(text)
    ids    = [word2idx.get(t, word2idx['<unk>']) for t in tokens]
    seq    = torch.tensor(ids[:max_len], dtype=torch.long)
    length = torch.tensor(min(len(ids), max_len), dtype=torch.long)
    if seq.size(0) < max_len:
        pad_id     = word2idx['<pad>']
        pad_tensor = torch.full((max_len - seq.size(0),), pad_id, dtype=torch.long)
        seq        = torch.cat([seq, pad_tensor], dim=0)
    return seq, length

from .char_vocab import char2idx, PAD

def preprocess_url(url: str, max_len: int = 100):
    u = clean_url(url)
    ids = [char2idx.get(c, 1) for c in u]
    seq = torch.tensor(ids[:max_len], dtype=torch.long)
    if seq.size(0) < max_len:
        pad_tensor = torch.full((max_len - seq.size(0),), PAD, dtype=torch.long)
        seq = torch.cat([seq, pad_tensor], dim=0)
    return seq
