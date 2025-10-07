# Created by aaronkueh on 10/1/2025
# aom/tools/mssql_checkpointer.py
"""
Custom LangGraph checkpointer for MSSQL Server using sql_connect.
"""
from __future__ import annotations

import pickle
from typing import Any, Dict, Iterator, Optional, Tuple, Sequence

from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata, CheckpointTuple
from sqlalchemy import text
from aom.definitions import CONFIG_DIR
from aom.tools.sql_connect import get_connector
from aom.utils.logger import get_logger
from aom.utils.utilities import config_param, get_profile


class MSSQLSaver(BaseCheckpointSaver):
    """LangGraph checkpointer that uses MSSQL Server via sql_connect."""

    def __init__(self, db_profile: Optional[str] = None, database: str = "mssql", schema: str = "dbo"):
        """
        Initialize MSSQL checkpointer.

        Args:
            db_profile: SQL profile name from sql_config.yaml (uses default if None)
            database: Database type (default: "mssql")
            schema: Database schema name (default: dbo)
        """
        super().__init__()
        self.sql_connector = get_connector(db_profile=db_profile, database=database)
        self.schema = schema
        self.log = get_logger(self.__class__.__name__)

    @classmethod
    def from_profile(cls, db_profile: Optional[str] = None, schema: str = "dbo") -> "MSSQLSaver":
        """
        Create checkpointer from SQL profile.

        Args:
            db_profile: SQL profile name from sql_config.yaml (uses default if None)
            schema: Database schema name
        """
        return cls(db_profile=db_profile, schema=schema)

    def get_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple for a given config."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"].get("checkpoint_id")

        with self.sql_connector.engine.connect() as conn:
            if checkpoint_id:
                query = text(f"""
                    SELECT checkpoint_id, parent_checkpoint_id, [type], checkpoint_data, metadata_data
                    FROM [{self.schema}].[checkpoints]
                    WHERE thread_id = :thread_id AND checkpoint_ns = :checkpoint_ns AND checkpoint_id = :checkpoint_id
                """)
                params = {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns, "checkpoint_id": checkpoint_id}
            else:
                query = text(f"""
                    SELECT TOP 1 checkpoint_id, parent_checkpoint_id, [type], checkpoint_data, metadata_data
                    FROM [{self.schema}].[checkpoints]
                    WHERE thread_id = :thread_id AND checkpoint_ns = :checkpoint_ns
                    ORDER BY created_at DESC
                """)
                params = {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}

            result = conn.execute(query, params)
            row = result.fetchone()

            if not row:
                return None

            checkpoint_id, parent_id, type_, checkpoint_blob, metadata_blob = row

            checkpoint = pickle.loads(checkpoint_blob)
            metadata = pickle.loads(metadata_blob) if metadata_blob else {}

            return CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                    }
                },
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": parent_id,
                    }
                } if parent_id else None,
            )

    def list(
            self,
            config: Optional[Dict[str, Any]] = None,
            *,
            filter: Optional[Dict[str, Any]] = None,
            before: Optional[str] = None,
            limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints matching criteria."""
        thread_id = config["configurable"]["thread_id"] if config else None
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "") if config else ""

        with self.sql_connector.engine.connect() as conn:
            query_str = f"SELECT checkpoint_id, parent_checkpoint_id, [type], checkpoint_data, metadata_data FROM [{self.schema}].[checkpoints] WHERE 1=1"
            params = {}

            if thread_id:
                query_str += " AND thread_id = :thread_id"
                params["thread_id"] = thread_id

            if checkpoint_ns:
                query_str += " AND checkpoint_ns = :checkpoint_ns"
                params["checkpoint_ns"] = checkpoint_ns

            if before:
                query_str += " AND checkpoint_id < :before"
                params["before"] = before

            query_str += " ORDER BY created_at DESC"

            if limit:
                query_str = f"SELECT TOP {limit} * FROM ({query_str}) AS t"

            result = conn.execute(text(query_str), params)

            for row in result:
                checkpoint_id, parent_id, type_, checkpoint_blob, metadata_blob = row
                checkpoint = pickle.loads(checkpoint_blob)
                metadata = pickle.loads(metadata_blob) if metadata_blob else {}

                yield CheckpointTuple(
                    config={
                        "configurable": {
                            "thread_id": thread_id or "",
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": checkpoint_id,
                        }
                    },
                    checkpoint=checkpoint,
                    metadata=metadata,
                    parent_config={
                        "configurable": {
                            "thread_id": thread_id or "",
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": parent_id,
                        }
                    } if parent_id else None,
                )

    def put(
            self,
            config: Dict[str, Any],
            checkpoint: Checkpoint,
            metadata: CheckpointMetadata,
            new_versions: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Save a checkpoint."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]
        parent_id = config["configurable"].get("checkpoint_id")

        try:
            with self.sql_connector.engine.connect() as conn:
                checkpoint_blob = pickle.dumps(checkpoint)
                metadata_blob = pickle.dumps(metadata) if metadata else None

                merge_query = text(f"""
                    MERGE [{self.schema}].[checkpoints] AS target
                    USING (SELECT :thread_id AS thread_id, :checkpoint_ns AS checkpoint_ns, :checkpoint_id AS checkpoint_id) AS source
                    ON target.thread_id = source.thread_id 
                       AND target.checkpoint_ns = source.checkpoint_ns 
                       AND target.checkpoint_id = source.checkpoint_id
                    WHEN MATCHED THEN
                        UPDATE SET 
                            checkpoint_data = :checkpoint_data, 
                            metadata_data = :metadata_data, 
                            parent_checkpoint_id = :parent_id
                    WHEN NOT MATCHED THEN
                        INSERT (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint_data, metadata_data)
                        VALUES (:thread_id, :checkpoint_ns, :checkpoint_id, :parent_id, :checkpoint_data, :metadata_data);
                """)

                conn.execute(merge_query, {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                    "parent_id": parent_id,
                    "checkpoint_data": checkpoint_blob,
                    "metadata_data": metadata_blob
                })
                conn.commit()
        except Exception as e:
            self.log.error(f"Failed to save checkpoint: {e}")

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(self, config: Dict[str, Any],
                   writes: Sequence[Tuple[str, Any]], task_id: str) -> None:
        """
        Store intermediate writes during graph execution.

        Args:
            config: Configuration with thread_id
            writes: Sequence of (channel, value) tuples
            task_id: Unique task identifier
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"].get("checkpoint_id")

        if not checkpoint_id:
            return

        try:
            with self.sql_connector.engine.connect() as conn:
                for idx, (channel, value) in enumerate(writes):
                    try:
                        value_blob = pickle.dumps(value)

                        insert_query = text(f"""
                            INSERT INTO [{self.schema}].[checkpoint_writes] 
                            (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, [value])
                            VALUES (:thread_id, :checkpoint_ns, :checkpoint_id, :task_id, :idx, :channel, :value_blob)
                        """)

                        conn.execute(insert_query, {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": checkpoint_id,
                            "task_id": task_id,
                            "idx": idx,
                            "channel": channel,
                            "value_blob": value_blob
                        })
                    except Exception as e:
                        self.log.error(f"Failed to save write for channel {channel}: {e}")

                conn.commit()
                self.log.debug(f"Saved {len(writes)} writes for checkpoint {checkpoint_id}")
        except Exception as e:
            self.log.error(f"Failed to save checkpoint writes: {e}")


if __name__ == '__main__':
    profile_name = "DatabaseLocal"
    checkpoint_check = MSSQLSaver.from_profile(profile_name)
    print(checkpoint_check)

