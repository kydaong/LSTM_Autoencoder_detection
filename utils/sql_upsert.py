# Created by aaronkueh on 9/22/2025
import pandas as pd
from sqlalchemy import Inspector
from sqlalchemy.dialects.mssql import NVARCHAR, FLOAT, DATETIME2, INTEGER
from sqlalchemy.engine import Engine
from aom.tools.sql_connect import SQLConnector
from LSTM_Autoencoder.definitions import DATA_DIR
conn = SQLConnector(database='mssql', profile='DatabaseLocal')
conn.inspector()

def read_table(conn_eng: Engine, table_name:str) -> pd.DataFrame:
    query = f"""
            SELECT * FROM {table_name}
            """
    df = pd.read_sql(query, conn_eng)
    return df

def create_table(conn_eng: Engine, table_name:str, ddl_create: str = None, replace: bool = False) -> None:
    """Create dbo.asset_data with required schema. Set replace=True to DROP+CREATE."""
    ddl_drop = f"IF OBJECT_ID('dbo.{table_name}','U') IS NOT NULL DROP TABLE dbo.{table_name};"
    ddl_create = f"""
    IF OBJECT_ID('dbo.{table_name}','U') IS NULL
    CREATE TABLE dbo.{table_name} (
        timestamp    DATETIME2    NOT NULL,
        asset_id     NVARCHAR(64) NOT NULL,
        temperature_c  FLOAT        NULL,
        vibration_mm_s    FLOAT        NULL,
        anomaly      INT          NULL
    );
    """
    with conn_eng.begin() as connection:
        if replace:
            connection.exec_driver_sql(ddl_drop)
        connection.exec_driver_sql(ddl_create)

def insert_data(conn_eng: Engine, df, table_name:str, sch_name:str,
                dtype: dict, insp, append: bool = False) -> None:
    if append:
        flg = "append"
    else:
        flg = "replace"

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%m/%d/%Y %H:%M", errors="raise")
        # strip timezone if any
        if getattr(df["timestamp"].dt, "tz", None) is not None:
            df["timestamp"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)

    exists = insp.has_table(table_name, schema=sch_name)

    if not exists:
        df.head(0).to_sql(name=table_name, con=conn_eng, schema=sch_name,
                          if_exists="fail", index=False, dtype=dtype)

    df.to_sql(name=table_name,
              con=conn_eng,
              schema=sch_name,
              if_exists=flg,
              index=False,
              dtype=dtype)