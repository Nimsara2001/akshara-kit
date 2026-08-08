"""One module per extraction backend.

Every module here exposes exactly one public function::

    extract(file_path: str) -> ExtractionResult

Adapters raise on failure; they never encode an error into the returned text.
"""
