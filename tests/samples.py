"""Sample strings shared by the tests.

Mirrors the constants in ``fixtures/generate_fixtures.py`` so tests can assert
on the exact content the fixtures were built from. See
``fixtures/README.md`` for provenance.
"""

from __future__ import annotations

#: Real FM-Abhaya legacy bytes: "Educational Publications Department".
LEGACY_SINHALA = "wOHdmk m%ldYk fomd¾;fïka;=j"

#: The ASCII control string. The prior prototype's blanket conversion turned
#: this into "අඅඅගැාමචමඉගටදඩගකන".
ASCII_CONTROL = "www.edupub.gov.lk"

#: The same phrase as LEGACY_SINHALA, already in correct Unicode.
UNICODE_SINHALA = "අධ්‍යාපන ප්‍රකාශන දෙපාර්තමේන්තුව"
UNICODE_SINHALA_2 = "සියලු ම පෙළපොත් නොමිලේ බෙදා දෙනු ලැබේ"

#: What LEGACY_SINHALA must become once normalised.
LEGACY_SINHALA_CONVERTED = UNICODE_SINHALA
