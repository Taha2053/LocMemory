"""
Tests for security layer - PII detection and encryption.
"""

import pytest
import tempfile
from pathlib import Path

from core.security import (
    detect_pii,
    MemoryEncryptor,
    process_before_store,
    decrypt_for_retrieval,
    get_encryptor,
)


def test_detect_email():
    """Email detection."""
    text = "Contact me at foo@bar.com"
    result = detect_pii(text)

    assert "email" in result
    assert "foo@bar.com" in result["email"]


def test_detect_phone():
    """Phone detection."""
    text = "Call me at +1 555-123-4567"
    result = detect_pii(text)

    assert "phone" in result
    # Should match the phone number
    assert any("555" in p for p in result.get("phone", []))


def test_detect_ssn():
    """SSN detection."""
    text = "My SSN is 123-45-6789"
    result = detect_pii(text)

    assert "ssn" in result
    assert "123-45-6789" in result["ssn"]


def test_detect_api_key():
    """API key detection."""
    text = "API key: sk-abc123def456gh789012ijk345lmn678"
    result = detect_pii(text)

    assert "api_key" in result


def test_no_false_positive_normal():
    """Normal sentences should not trigger PII detection."""
    normal_texts = [
        "I went to the gym today",
        "Working on a Python project",
        "Reading a book about machine learning",
        "Meeting with the team tomorrow",
        "Feeling good about progress",
        "Learning new algorithms",
        "Debugging some code",
        "Writing documentation",
        "Testing the system",
        "Deploying to production",
        "Reading API docs",
        "Code review scheduled",
        "Client call at 3pm",
        "Weekend plans with family",
        "Movie night on Saturday",
        "Cooking dinner tonight",
        "Running 5k this morning",
        "Yoga session yesterday",
        "Diet includes vegetables",
        "Sleep schedule adjusted",
    ]

    false_positives = 0
    for text in normal_texts:
        result = detect_pii(text)
        if result:
            false_positives += 1

    # Should be less than 5% (1 out of 20)
    assert false_positives < 1


def test_encrypt_decrypt_roundtrip(tmp_path):
    """Encrypt then decrypt returns original text."""
    key_path = tmp_path / "test_key.key"
    encryptor = MemoryEncryptor(str(key_path))

    original = "This is a secret message"
    encrypted = encryptor.encrypt(original)
    decrypted = encryptor.decrypt(encrypted)

    assert decrypted == original
    assert encrypted != original


def test_is_encrypted_detects_cipher(tmp_path):
    """Encrypted text should be detected as encrypted."""
    key_path = tmp_path / "test_key.key"
    encryptor = MemoryEncryptor(str(key_path))

    original = "Secret text"
    encrypted = encryptor.encrypt(original)

    assert encryptor.is_encrypted(encrypted)


def test_normal_text_not_encrypted():
    """Plain text should not be detected as encrypted."""
    encryptor = MemoryEncryptor()

    assert not encryptor.is_encrypted("This is normal text")


def test_process_before_store_pii(tmp_path):
    """Text with email should be encrypted."""
    key_path = tmp_path / "test_key.key"
    encryptor = MemoryEncryptor(str(key_path))

    text = "Contact me at test@example.com"
    processed, was_encrypted = process_before_store(text, encryptor)

    assert was_encrypted is True
    assert processed != text  # Should be encrypted
    # Verify we can decrypt
    decrypted = encryptor.decrypt(processed)
    assert "test@example.com" in decrypted


def test_process_before_store_clean(tmp_path):
    """Normal text should not be encrypted."""
    key_path = tmp_path / "test_key.key"
    encryptor = MemoryEncryptor(str(key_path))

    text = "I went to the gym today"
    processed, was_encrypted = process_before_store(text, encryptor)

    assert was_encrypted is False
    assert processed == text


def test_process_before_store_no_encryptor():
    """Should work without explicit encryptor (uses singleton)."""
    text = "test@example.com"

    # Should not raise
    processed, was_encrypted = process_before_store(text)

    # Just verify no crash
    assert isinstance(processed, str)


def test_detect_password():
    """Password pattern detection."""
    text = "Password: mysecretpassword123"
    result = detect_pii(text)

    assert "password" in result


def test_detect_credit_card():
    """Credit card detection."""
    text = "Card number 1234 5678 9012 3456"
    result = detect_pii(text)

    assert "credit_card" in result


def test_detect_ip_address():
    """IP address detection."""
    text = "Server at 192.168.1.100"
    result = detect_pii(text)

    assert "ip_address" in result
    assert "192.168.1.100" in result["ip_address"]