import re
import math
import urllib.parse
from collections import Counter
from typing import Dict, Union

import tldextract

def calculate_entropy(text: str) -> float:
    """
    Calculates the Shannon Entropy of a given string.
    
    Measures information density to detect randomly generated domains 
    (DGAs) frequently used in zero-day phishing attacks.
    
    Args:
        text (str): The raw string to analyze.
        
    Returns:
        float: The calculated entropy score, rounded to 4 decimal places.
    """
    if not text:
        return 0.0
    
    length = len(text)
    counts = Counter(text)
    
    # H(X) = -sum(P(x) * log2(P(x)))
    entropy = -sum(
        (count / length) * math.log2(count / length) 
        for count in counts.values()
    )
    
    return round(entropy, 4)


def extract_features(url: str) -> Dict[str, Union[int, float]]:
    """
    Parses a raw URL and extracts structural, lexical, and mathematical 
    features for the machine learning classification pipeline.
    
    Args:
        url (str): The raw URL string provided by the user or dataset.
        
    Returns:
        Dict[str, Union[int, float]]: A dictionary of extracted feature metrics.
    """
    # 1. Edge Case Handling: Ensure URL has a scheme for accurate parsing
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url

    # 2. Parsing
    parsed = urllib.parse.urlparse(url)
    ext = tldextract.extract(url)
    
    # 3. Subdomain formatting check
    subdomain_count = len(ext.subdomain.split('.')) if ext.subdomain else 0
    
    # 4. Strict IPv4 validation (Prevents false positives on standard numbers)
    ip_pattern = re.compile(
        r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}'
        r'(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
    )
    # parsed.hostname handles cases where the URL port is attached (e.g., :8080)
    has_ip = 1 if parsed.hostname and ip_pattern.match(parsed.hostname) else 0

    # 5. Feature Assembly
    features = {
        "URL_Length": len(url),
        "Domain_Length": len(ext.domain),
        "Is_HTTPS": 1 if parsed.scheme == 'https' else 0,
        "Num_Dots": url.count('.'),
        "Num_Hyphens": url.count('-'),
        "Num_Subdomains": subdomain_count,
        "Entropy": calculate_entropy(url),
        "Has_IP": has_ip,
        "Punycode": 1 if 'xn--' in url.lower() else 0
    }
    
    return features