"""
Security layer for LocMemory.

Provides PII detection and AES-GCM encryption for sensitive memory nodes.
"""

from core.security.security import (
    detect_pii,
    MemoryEncryptor,
    process_before_store,
    decrypt_for_retrieval,
    get_encryptor,
)

__all__ = [
    "detect_pii",
    "MemoryEncryptor", 
    "process_before_store",
    "decrypt_for_retrieval",
    "get_encryptor",
]