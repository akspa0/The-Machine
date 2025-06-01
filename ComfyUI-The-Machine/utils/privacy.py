"""
Privacy and PII removal utilities for ComfyUI-The-Machine nodes.
Implement all logic according to the privacy_and_manifest specification.
"""
import re
import os
import unicodedata

def scrub_filename(filename):
    """
    Remove PII from filename by stripping emails, phone numbers, names, and non-alphanumeric characters (except dash/underscore/dot).
    Handles unicode and edge cases. Returns a sanitized filename.
    """
    # Normalize unicode
    filename = unicodedata.normalize('NFKD', filename)
    # Remove email addresses
    filename = re.sub(r"[\w\.-]+@[\w\.-]+", "", filename)
    # Remove phone numbers (various patterns)
    filename = re.sub(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "", filename)
    filename = re.sub(r"\b\d{10,}\b", "", filename)
    # Remove anything that looks like a name (capitalized words, initials)
    filename = re.sub(r"\b[A-Z][a-z]+\b", "", filename)
    filename = re.sub(r"\b[A-Z]{2,}\b", "", filename)
    # Remove any remaining non-alphanumeric (except dash, underscore, dot, and extension)
    name, ext = os.path.splitext(filename)
    name = re.sub(r"[^a-zA-Z0-9._-]", "", name)
    sanitized = name + ext
    sanitized = sanitized.strip(".-_")
    if not sanitized:
        sanitized = "file"
    return sanitized

def scrub_metadata(metadata):
    """
    Remove PII from audio metadata (ID3, etc.).
    Strips common fields: artist, album, composer, comments, etc.
    Handles unicode and edge cases.
    """
    pii_fields = [
        "artist", "album", "composer", "comment", "comments", "author", "email", "phone", "location", "copyright",
        "contact", "address", "organization", "owner", "user", "description", "website", "url"
    ]
    clean = dict(metadata)
    for field in pii_fields:
        if field in clean:
            clean[field] = ""
    # Remove any field value that looks like PII
    for k, v in clean.items():
        if isinstance(v, str) and contains_pii(v):
            clean[k] = ""
    return clean

def scrub_directory(directory):
    """
    Scrub all files in a directory for PII in filenames. Renames files in place.
    Returns a list of (old, new) filename pairs.
    """
    renamed = []
    for fname in os.listdir(directory):
        old_path = os.path.join(directory, fname)
        if os.path.isfile(old_path):
            new_fname = scrub_filename(fname)
            new_path = os.path.join(directory, new_fname)
            if new_fname != fname:
                os.rename(old_path, new_path)
                renamed.append((fname, new_fname))
    return renamed

def contains_pii(s):
    """
    Returns True if the string contains likely PII (email, phone, etc.), else False.
    """
    if re.search(r"[\w\.-]+@[\w\.-]+", s):
        return True
    if re.search(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", s):
        return True
    if re.search(r"\b\d{10,}\b", s):
        return True
    if re.search(r"\b[A-Z][a-z]+\b", s):
        return True
    return False
# Future: add more advanced PII detection (NER, etc.) 