import re
import math
import urllib.parse
from collections import Counter
from typing import Dict, Union
import tldextract

# Pre-compile regex and sets for speed
IP_PATTERN = re.compile(
    r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}'
    r'(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
)
SUSPICIOUS_KEYWORDS = {
    'login', 'secure', 'verify', 'account', 'update', 
    'banking', 'support', 'service', 'auth', 'confirm'
}
VOWELS = set('aeiou')
CONSONANTS = set('bcdfghjklmnpqrstvwxyz')

def calculate_entropy(text: str) -> float:
    if not text: return 0.0
    length = len(text)
    counts = Counter(text)
    return round(-sum((count / length) * math.log2(count / length) for count in counts.values()), 4)

def get_longest_consonant_sequence(text: str) -> int:
    """Finds the longest string of consecutive consonants (common in DGA domains)."""
    text = text.lower()
    max_len = 0
    current_len = 0
    for char in text:
        if char in CONSONANTS:
            current_len += 1
            max_len = max(max_len, current_len)
        else:
            current_len = 0
    return max_len

def extract_features(url: str) -> Dict[str, Union[int, float]]:
    """Extracts high-dimensional security features from a raw URL."""
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url

    parsed = urllib.parse.urlparse(url)
    ext = tldextract.extract(url)
    url_lower = url.lower()
    
    # 1. Base Metrics
    url_len = len(url)
    domain_len = len(ext.domain)
    
    # 2. NLP & Keyword Analysis
    keyword_count = sum(1 for word in SUSPICIOUS_KEYWORDS if word in url_lower)
    
    # 3. Character Ratio Analysis
    vowel_count = sum(1 for c in url_lower if c in VOWELS)
    consonant_count = sum(1 for c in url_lower if c in CONSONANTS)
    vc_ratio = round(vowel_count / consonant_count, 4) if consonant_count > 0 else 0.0
    special_char_count = sum(1 for c in url_lower if not c.isalnum())
    special_ratio = round(special_char_count / url_len, 4) if url_len > 0 else 0.0

    features = {
        "URL_Length": url_len,
        "Domain_Length": domain_len,
        "Is_HTTPS": 1 if parsed.scheme == 'https' else 0,
        "Num_Dots": url.count('.'),
        "Num_Hyphens": url.count('-'),
        "Num_Subdomains": len(ext.subdomain.split('.')) if ext.subdomain else 0,
        "Entropy": calculate_entropy(url),
        "Has_IP": 1 if parsed.hostname and IP_PATTERN.match(parsed.hostname) else 0,
        "Punycode": 1 if 'xn--' in url_lower else 0,
        
        # New High-Value Features
        "Suspicious_Keywords": keyword_count,
        "VC_Ratio": vc_ratio,
        "Longest_Consonant_Seq": get_longest_consonant_sequence(ext.domain),
        "Special_Char_Ratio": special_ratio
    }
    
    return features