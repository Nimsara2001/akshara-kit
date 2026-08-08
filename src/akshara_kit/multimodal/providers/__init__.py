"""Vision-language provider implementations.

One module per provider, each independently importable so that installing a
single SDK is enough to use that provider. Modules never import one another;
provider selection lives in :mod:`akshara_kit.multimodal.fallback`, mirroring
the rule that keeps the text extractors decoupled (spec Section 12).
"""
