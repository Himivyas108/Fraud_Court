"""
Point the whole test session at a throwaway SQLite file BEFORE any
`server.*` module is imported (env var must be set before server.db is
first imported, since it reads FRAUDCOURT_DB_PATH at import time).
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_tmp_dir = tempfile.mkdtemp(prefix="fraudcourt_test_")
os.environ["FRAUDCOURT_DB_PATH"] = os.path.join(_tmp_dir, "test.db")

from server.db import init_db  # noqa: E402
init_db()
