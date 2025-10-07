import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)


from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import quote_plus
from dotenv import load_dotenv

from LSTM_Autoencoder.utils.utilities import config_param, get_profile, resolve_auth
from LSTM_Autoencoder.definitions import CONFIG_DIR
load_dotenv()
from sqlalchemy import create_engine, inspect, Inspector, text
from sqlalchemy.engine import Engine, Connection
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.event import listens_for
from LSTM_Autoencoder.utils.logger import get_logger
import pandas as pd

def odbc_conn_str(db_profile: Mapping[str, Any], auth: Mapping[str, Any]) -> str:
    """
    Returns a Windows/Unix-compatible ODBC connection string for MSSQL.
    All values are inserted as-is; the URL quoting is done at the SQLAlchemy URL layer.
    """
    parts = {
    "Driver": db_profile.get("driver"),
    "Server": db_profile.get("server"),
    "Database": db_profile.get("database"),
    "Uid": auth.get("user"),
    "Pwd": auth.get("pwd"),
    "Encrypt": db_profile.get("encrypt", "no"),  # Add this
    "TrustServerCertificate": db_profile.get("trust_server_certificate", "yes"),  # Add this
    "Connection Timeout": db_profile.get("timeout", 30),}

    return ";".join(f"{k}={v}" for k, v in parts.items()) + ";"


@dataclass
class SQLConnector(object):

    def __init__(self, database: str, db_profile: str = None):
        self.config = config_param(config_file=f'{CONFIG_DIR}/sql_config.yaml', field=database)
        profile_name, self.db_profile = get_profile(self.config, name=db_profile)
        self.auth = resolve_auth(profile=self.db_profile)
        self.echo: bool = False
        self.log = get_logger(self.__class__.__name__)
        self.log.info("Database profile: %r", profile_name)
        # self.log.warning("Database name: %r", self.db_profile['database'])

        odbc = odbc_conn_str(self.db_profile, self.auth)
        url = f"mssql+pyodbc:///?odbc_connect={quote_plus(odbc)}"

        self.engine: Engine = create_engine(
            url,
            echo=self.echo,
            pool_pre_ping=True,           # auto-reconnect on stale connections
            fast_executemany=True,        # speeds up executemany with pyodbc
        )

        @listens_for(self.engine, "before_cursor_execute")
        def _set_fast_executemany(_conn, cursor, _statement, _parameters, _context, executemany):
            if executemany and hasattr(cursor, "fast_executemany"):
                cursor.fast_executemany = True

        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)

    def get_engine(self) -> Engine:
        return self.engine

    def connect(self) -> Connection:
        return self.engine.connect()

    def session(self) -> Session:
        return self.SessionLocal()

    def inspector(self) -> Inspector:
        insp = inspect(self.engine)
        schemas = insp.get_schema_names()
        tables_dbo = insp.get_table_names(schema="dbo")
        print('schema', schemas)
        print('tables', tables_dbo)
        return insp

    @staticmethod
    def _is_safe_identifier(name: str) -> bool:
        """
        Minimal guard for identifiers (table/column). Allows letters, digits, underscore, and dot (schema.table).
        Adjust to your naming policy as needed.
        """
        return bool(name) and all(c.isalnum() or c in ("_", ".") for c in name)

    def sql_query(self, table_name: str, columns: list[str] | None = None,
                  limit: int | None = None) -> dict:
        """
        Execute a simple SELECT against a table and return {"columns": [...], "rows": [[...], ...]}.

        - table_name: required table to select from.
        - columns: optional list of column names to return. Defaults to *.
        - limit: optional integer row limit.
        """
        if not self._is_safe_identifier(table_name):
            raise ValueError("Invalid table name")

        if columns:
            if not all(self._is_safe_identifier(col) for col in columns):
                raise ValueError("Invalid column name(s)")
            select_cols = ", ".join(columns)
        else:
            select_cols = "*"

        # Use OFFSET/FETCH for MSSQL-compatible limiting without dynamic TOP injection.
        stmt = f"SELECT {select_cols} FROM {table_name}"
        params: dict[str, Any] = {}
        if limit is not None:
            stmt += " ORDER BY (SELECT NULL) OFFSET 0 ROWS FETCH NEXT :limit ROWS ONLY"
            params["limit"] = int(limit)

        # context-managed connection to avoid leaks
        with self.engine.connect() as conn:
            df = pd.read_sql(text(stmt), conn, params=params if params else None)

        split = df.to_dict(orient="split")  # {'index': [...], 'columns': [...], 'data': [[...], ...]}
        query_dict = {"columns": split["columns"], "rows": split["data"]}
        return query_dict

    def mssql_ts_query(self, _params: dict) -> dict:
        """
        Timeseries query facade. Returns rows from the 'asset_data' table filtered by:
          - asset_id: exact match
          - start/end: inclusive time window on 'timestamp' column (if provided)
          - signals: optional list of column names to project
        orders by timestamp when available.
        """
        table = "asset_data"
        if not self._is_safe_identifier(table):
            raise ValueError("Invalid table name")

        signals = _params.get("signals")
        select_cols = "*"
        if signals and isinstance(signals, list) and all(isinstance(c, str) for c in signals):
            # Validate identifiers to avoid SQL injection
            bad = [c for c in signals if not self._is_safe_identifier(c)]
            if bad:
                raise ValueError(f"Invalid signal/column names: {bad}")
            select_cols = ", ".join(signals)

        where_clauses: list[str] = []
        params: dict[str, Any] = {}

        asset_id = _params.get("asset_id")
        if asset_id is not None and str(asset_id).strip():
            where_clauses.append("asset_id = :asset_id")
            params["asset_id"] = str(asset_id).strip()

        start = _params.get("start")
        end = _params.get("end")
        # Accept strings or datetime; bind as strings for MSSQL
        if start:
            where_clauses.append("[timestamp] >= :start_ts")
            params["start_ts"] = str(start)
        if end:
            where_clauses.append("[timestamp] <= :end_ts")
            params["end_ts"] = str(end)

        stmt = f"SELECT {select_cols} FROM {table}"
        if where_clauses:
            stmt += " WHERE " + " AND ".join(where_clauses)

        # Order by timestamp when it is selected or exists
        stmt += " ORDER BY [timestamp]"

        with self.engine.connect() as conn:
            df = pd.read_sql(text(stmt), conn, params=params if params else None)

        split = df.to_dict(orient="split")
        return {"columns": split["columns"], "rows": split["data"]}


    def mssql_metadata_query(self, _params: dict) -> dict:
        """
        Metadata query facade. Returns all rows from the asset_meta table.
        You may filter by params (asset_id) inside this method later.
        """
        return self.sql_query(table_name="asset_meta")

_CONNECTOR_CACHE: dict[tuple[str, str], SQLConnector] = {}

def get_connector(db_profile: str | None = None, database: str = "mssql") -> SQLConnector:
    """
    Return a cached SQLConnector for the given (database, db_profile).
    If db_profile is None, falls back to 'DatabaseLocal'.
    """
    key = (database, db_profile)
    if key not in _CONNECTOR_CACHE:
        _CONNECTOR_CACHE[key] = SQLConnector(database=database, db_profile=key[1])
    return _CONNECTOR_CACHE[key]


if __name__ == '__main__':
    sql_conn = SQLConnector(database='mssql', db_profile=None) #db_profile = None, Default profile
    sql_conn.inspector()
    read_db = sql_conn.sql_query("compressor_normal_dataset3")
    print('read\n', read_db)