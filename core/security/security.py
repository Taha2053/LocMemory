"""
Security layer for LocMemory.

Provides PII detection and AES-GCM encryption for sensitive memory nodes.
"""

import base64
import re
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from core.settings.config import get_config


# PII detection patterns
PII_PATTERNS = {
    # Email: standard email pattern
    "email": re.compile(
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    ),
    # Phone: international + local formats
    "phone": re.compile(
        r"(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}"
    ),
    # SSN: XXX-XX-XXXX pattern
    "ssn": re.compile(r"\b[0-9]{3}-[0-9]{2}-[0-9]{4}\b"),
    # API keys: common patterns
    "api_key": re.compile(
        r"\b(?:sk-|pk_)[a-zA-Z0-9]{32,}|"
        r"\bBearer\s+[a-zA-Z0-9]{32,}|"
        r"\bghp_[a-zA-Z0-9]{36}"
    ),
    # Password: lines containing password, pwd, pass
    "password": re.compile(
        r"(?i)\b(?:password|pwd|pass)[:\s][^\s]{4,}"
    ),
    # Credit card: 16-digit with optional dashes/spaces
    "credit_card": re.compile(
        r"\b[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{4}\b"
    ),
    # IPv4 address
    "ip_address": re.compile(
        r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"
    ),
}


def detect_pii(text: str) -> dict[str, list[str]]:
    """
    Detect PII in text using regex patterns.

    Returns dict of {pii_type: [matches]}
    Returns empty dict if no PII found.

    Args:
        text: The text to scan for PII

    Returns:
        Dict mapping PII type to list of matches found
    """
    if not text:
        return {}

    results = {}

    for pii_type, pattern in PII_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            results[pii_type] = matches

    return results


class MemoryEncryptor:
    """
    AES-GCM encryptor for memory nodes.

    Uses 256-bit AES in GCM mode for authenticated encryption.
    """

    NONCE_SIZE = 12  # 96 bits
    KEY_SIZE = 32    # 256 bits

    def __init__(self, key_path: Optional[str] = None):
        """
        Initialize the encryptor.

        Auto-generates key if not exists, loads if exists.

        Args:
            key_path: Path to key file. Defaults to config.
        """
        self._config = get_config()
        key_path = key_path or self._config.get(
            "security", "key_path", "data/secret.key"
        )
        self._key_path = Path(key_path)

        self._key = self._load_or_generate_key()
        self._aesgcm = AESGCM(self._key)

    def _load_or_generate_key(self) -> bytes:
        """
        Load existing key or generate new one.

        Returns:
            32-byte encryption key
        """
        if self._key_path.exists():
            with open(self._key_path, "rb") as f:
                key = f.read()
                if len(key) == self.KEY_SIZE:
                    return key
                # Invalid key size, regenerate
        # Generate new key
        key = AESGCM.generate_key(bit_length=self.KEY_SIZE * 8)
        # Save to file with restricted permissions
        self._key_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._key_path, "wb") as f:
            f.write(key)
        # Set restrictive permissions (owner only)
        try:
            import os
            os.chmod(self._key_path, 0o600)
        except Exception:
            pass
        return key

    def encrypt(self, text: str) -> str:
        """
        Encrypt text using AES-GCM.

        Returns base64-encoded ciphertext with nonce prepended.

        Args:
            text: Plain text to encrypt

        Returns:
            Base64-encoded encrypted string
        """
        if not text:
            return text

        # Generate random nonce
        nonce = AESGCM.generate_key(bit_length=self.NONCE_SIZE * 8)

        # Encrypt
        ciphertext = self._aesgcm.encrypt(nonce, text.encode("utf-8"), None)

        # Combine nonce + ciphertext and base64 encode
        combined = nonce + ciphertext
        return base64.b64encode(combined).decode("utf-8")

    def decrypt(self, encrypted: str) -> str:
        """
        Decrypt encrypted text.

        Args:
            encrypted: Base64-encoded encrypted string

        Returns:
            Decrypted plain text
        """
        if not encrypted:
            return encrypted

        try:
            # Decode base64
            combined = base64.b64decode(encrypted.encode("utf-8"))

            # Split nonce and ciphertext
            nonce = combined[: self.NONCE_SIZE]
            ciphertext = combined[self.NONCE_SIZE :]

            # Decrypt
            plaintext = self._aesgcm.decrypt(nonce, ciphertext, None)
            return plaintext.decode("utf-8")
        except Exception:
            # Return original if decryption fails
            return encrypted

    def is_encrypted(self, text: str) -> bool:
        """
        Check if text appears to be encrypted.

        Heuristic check: base64-decodable with correct length
        and valid AES-GCM structure.

        Args:
            text: Text to check

        Returns:
            True if text appears encrypted
        """
        if not text:
            return False

        # Must be base64-encodable
        try:
            combined = base64.b64decode(text.encode("utf-8"))
        except Exception:
            return False

        # Minimum length: nonce (12) + tag (16) + at least 1 byte
        if len(combined) < 29:
            return False

        # Should start with random-looking nonce, not printable ASCII
        # Check first byte is non-printable (nonce is random)
        if combined[0] >= 32 and combined[0] < 127:
            # Could be plain text starting with printable char
            # Do additional check: try to decode as AES-GCM would fail on plain text
            pass
        else:
            return True

        # Try actual decryption to verify
        try:
            nonce = combined[: self.NONCE_SIZE]
            ciphertext = combined[self.NONCE_SIZE :]
            self._aesgcm.decrypt(nonce, ciphertext, None)
            return True
        except Exception:
            return False


def process_before_store(text: str, encryptor: Optional[MemoryEncryptor] = None) -> tuple[str, bool]:
    """
    Process text before storing in graph.

    1. Detect PII
    2. If PII found → encrypt the text
    3. Return (processed_text, was_encrypted)

    Args:
        text: The text to process
        encryptor: Optional encryptor instance. Creates one if not provided.

    Returns:
        Tuple of (processed_text, was_encrypted)
    """
    if not text:
        return text, False

    # Detect PII
    pii = detect_pii(text)

    if pii:
        # Encrypt the text if PII found
        if encryptor is None:
            encryptor = MemoryEncryptor()
        encrypted_text = encryptor.encrypt(text)
        return encrypted_text, True

    # No PII found, return unchanged
    return text, False


def decrypt_for_retrieval(text: str, encryptor: Optional[MemoryEncryptor] = None) -> str:
    """
    Decrypt text during retrieval if needed.

    Args:
        text: Potentially encrypted text
        encryptor: Optional encryptor instance

    Returns:
        Decrypted text if encrypted, original otherwise
    """
    if not text:
        return text

    if encryptor is None:
        encryptor = MemoryEncryptor()

    if encryptor.is_encrypted(text):
        return encryptor.decrypt(text)

    return text


# Singleton encryptor instance
_encryptor: Optional[MemoryEncryptor] = None


def get_encryptor() -> MemoryEncryptor:
    """Get the singleton encryptor instance."""
    global _encryptor
    if _encryptor is None:
        _encryptor = MemoryEncryptor()
    return _encryptor