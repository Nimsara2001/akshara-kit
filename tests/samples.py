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

#: A correct Sinhala paragraph, long enough to clear the consonant floor the
#: well-formedness probe applies before it will judge anything.
WELL_FORMED_PARAGRAPH = (
    "උපාධි පිරිනමන ආයතන වල ප්‍රථම උපාධි පාඨමාලා වලට ඇතුලත් වීම "
    "සඳහා සූදානම් විය යුතු ආකාරය පිළිබඳව වැදගත් උපදෙස්"
)

#: The same sentence as the text layer of ``sample_unicode.pdf`` actually
#: yields it. The PDF's embedded ToUnicode cmap is wrong, so every text-stream
#: backend returns this identically: "උපාධි" as "උපොධි", "ආකාරය" as "ආ ොරය",
#: leaving vowel signs stranded after spaces where a consonant was dropped.
#: Confirmed against OCR of the same page, which reads correctly.
GARBLED_CMAP_SINHALA = (
    "උපොධි පිරිනමන ආයතන වල ප්‍රථම උපොධි පොඨමොලො වලට ඇතුලේ ීම "
    "සඳහො සූදොනම් විය යුතු ආ ොරය"
)
