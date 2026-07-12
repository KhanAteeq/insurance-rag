"""Insurance RAG backend application package."""

# ChromaDB requires sqlite3 >= 3.35. Some hosts (e.g. Azure App Service's Linux
# image) ship an older system sqlite3. When the pysqlite3 binary is available
# (installed on the cloud via pysqlite3-binary), transparently replace the
# stdlib sqlite3 module with it BEFORE chromadb is imported anywhere.
try:  # pragma: no cover - environment dependent
    __import__("pysqlite3")
    import sys as _sys

    _sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")
except Exception:
    pass  # local dev (e.g. Windows) uses the stdlib sqlite3, which is new enough

