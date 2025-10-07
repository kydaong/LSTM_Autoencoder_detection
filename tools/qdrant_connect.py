# Created by aaronkueh on 9/30/2025
# aom/tools/qdrant_connect.py
from __future__ import annotations

from typing import Dict, Any, Optional, List
import hashlib
import json
import os
import yaml
from datetime import datetime, timedelta

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, OptimizersConfigDiff
from qdrant_client.http.models import CollectionInfo

from aom.utils.logger import get_logger
from aom.definitions import CONFIG_DIR


class QdrantConnector:
    """
    Qdrant connection manager with configuration-based collection management.

    Features:
    - Singleton client pattern
    - Configuration-based collection creation
    - Session-based conversation context storage
    - Automatic session expiration
    """

    _instance: Optional[QdrantConnector] = None
    _client: Optional[QdrantClient] = None

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Qdrant connector with configuration.

        Args:
            config_path: Path to qdrant_config.yaml (default: config/qdrant_config.yaml)
        """
        self.log = get_logger("qdrant_connect")

        # Load configuration
        if config_path is None:
            config_path = os.path.join(CONFIG_DIR, "qdrant_config.yaml")

        self.config = self._load_config(config_path)

        # Extract configuration values
        self.url = self.config["url"]
        self.api_key = self.config["api_key"]
        self.timeout = self.config["timeout"]
        self.session_timeout_minutes = self.config["session_timeout_minutes"]
        self.collections_config = self.config["collections"]
        self.performance_config = self.config["performance"]

        # Default collection for conversation memory
        self.default_collection = self.collections_config.get("conversation_memory", {}).get("name",
                                                                                             "conversation_memory")

        # Initialize client
        self._initialize_client()

    @classmethod
    def get_instance(cls, config_path: Optional[str] = None) -> QdrantConnector:
        """Get or create a singleton instance"""
        if cls._instance is None:
            cls._instance = cls(config_path)
        return cls._instance

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load Qdrant configuration from YAML file"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

            # Substitute environment variables
            qdrant_config = config.get("qdrant", {})
            qdrant_url = os.path.expandvars(qdrant_config.get("url", ""))
            qdrant_api_key = os.path.expandvars(qdrant_config.get("api_key", ""))

            # Remove an empty api_key if not set in env
            if qdrant_api_key == "${QDRANT_API_KEY}" or not qdrant_api_key:
                qdrant_api_key = None

            return {
                "url": qdrant_url,
                "api_key": qdrant_api_key,
                "timeout": qdrant_config.get("timeout", 30),
                "session_timeout_minutes": qdrant_config.get("session", {}).get("timeout_minutes", 30),
                "collections": config.get("collections", {}),
                "performance": config.get("performance", {})
            }
        except Exception as e:
            self.log.error(f"Failed to load Qdrant configuration: {e}")
            raise

    def _initialize_client(self) -> None:
        """Initialize Qdrant client connection"""
        try:
            # Initialize a client with or without an API key
            if self.api_key:
                self._client = QdrantClient(
                    url=self.url,
                    api_key=self.api_key,
                    timeout=self.timeout
                )
            else:
                self._client = QdrantClient(
                    url=self.url,
                    timeout=self.timeout
                )

            self.log.info(f"Qdrant client initialized successfully at {self.url}")
        except Exception as e:
            self.log.error(f"Failed to initialize Qdrant client: {e}")
            raise

    @property
    def client(self) -> QdrantClient:
        """Get the Qdrant client instance"""
        if self._client is None:
            self._initialize_client()
        return self._client

    def _ensure_collection(self, collection_name: str, collection_config: Dict[str, Any]) -> None:
        """
        Ensure a collection exists with a proper configuration.

        Args:
            collection_name: Name of the collection
            collection_config: Configuration dictionary for the collection
        """
        try:
            collections_response = self.client.get_collections()
            collections: List[CollectionInfo] = collections_response.collections

            if not any(c.name == collection_name for c in collections):
                # Get configuration
                vector_size = collection_config.get("vector_size", 1024)
                distance_str = collection_config.get("distance", "cosine").upper()
                distance = getattr(Distance, distance_str, Distance.COSINE)

                # Get performance config
                hnsw_config = self.performance_config.get("hnsw", {})
                indexing_threshold = self.performance_config.get("indexing_threshold", 20000)

                # Prepare optimizer config
                optimizer_config = OptimizersConfigDiff(indexing_threshold=indexing_threshold)

                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=distance),
                    hnsw_config=hnsw_config if hnsw_config else None,
                    optimizers_config=optimizer_config
                )
                self.log.info(f"Created Qdrant collection: {collection_name}")
            else:
                self.log.info(f"Using existing Qdrant collection: {collection_name}")
        except Exception as e:
            self.log.error(f"Failed to ensure collection {collection_name}: {e}")
            raise

    def create_collection(self, collection_key: str) -> bool:
        """
        Create a collection based on a configuration key from qdrant_config.yaml

        Args:
            collection_key: Key name in collections config (e.g., 'knowledge_base')

        Returns:
            True if successful, False otherwise
        """
        try:
            if collection_key not in self.collections_config:
                self.log.error(f"Collection '{collection_key}' not found in configuration")
                return False

            collection_config = self.collections_config[collection_key]
            actual_name = collection_config.get("name", collection_key)

            self._ensure_collection(actual_name, collection_config)
            return True
        except Exception as e:
            self.log.error(f"Error creating collection {collection_key}: {e}")
            return False

    def list_collections(self) -> List[str]:
        """
        Get a list of all collections in Qdrant.

        Returns:
            List of collection names
        """
        try:
            collections_response = self.client.get_collections()
            collections: List[CollectionInfo] = collections_response.collections
            return [c.name for c in collections]
        except Exception as e:
            self.log.error(f"Error listing collections: {e}")
            return []

    @staticmethod
    def _get_point_id(session_id: str) -> str:
        """Convert session_id to valid Qdrant point ID"""
        return hashlib.md5(session_id.encode()).hexdigest()

    def get_conversation_context(self, session_id: str, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve conversation context from Qdrant for a given session.

        Args:
            session_id: Unique session identifier
            collection_name: Optional collection name (uses default if not provided)

        Returns:
            Dictionary containing conversation context, or empty dict if not found/expired
        """
        if collection_name is None:
            collection_name = self.default_collection

        try:
            point_id = self._get_point_id(session_id)

            points = self.client.retrieve(
                collection_name=collection_name,
                ids=[point_id],
                with_payload=True,
            )

            if points:
                payload = points[0].payload
                # Check if a session is still valid
                timestamp_str = payload.get("timestamp", "2000-01-01T00:00:00")
                timestamp = datetime.fromisoformat(timestamp_str)

                if datetime.now() - timestamp < timedelta(minutes=self.session_timeout_minutes):
                    context_str = payload.get("context", "{}")
                    return json.loads(context_str) if isinstance(context_str, str) else context_str
                else:
                    # Session expired, delete it
                    self.log.info(f"Session {session_id} expired, deleting")
                    self.delete_conversation_context(session_id, collection_name)

            return {}
        except Exception as e:
            self.log.error(f"Error retrieving conversation context for session {session_id}: {e}")
            return {}

    def save_conversation_context(
            self,
            session_id: str,
            context: Dict[str, Any],
            collection_name: Optional[str] = None,
            vector: Optional[List[float]] = None
    ) -> bool:
        """
        Save conversation context to Qdrant.

        Args:
            session_id: Unique session identifier
            context: Dictionary containing conversation context
            collection_name: Optional collection name (uses default if not provided)
            vector: Optional embedding vector (uses zeros if not provided)

        Returns:
            True if successful, False otherwise
        """
        if collection_name is None:
            collection_name = self.default_collection

        try:
            point_id = self._get_point_id(session_id)

            # Get vector size from config
            collection_config = None
            for key, config in self.collections_config.items():
                if config.get("name") == collection_name:
                    collection_config = config
                    break

            vector_size = collection_config.get("vector_size", 1024) if collection_config else 1024

            # Create embedding vector (zeros by default for key-value storage)
            if vector is None:
                vector = [0.0] * vector_size

            point = PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "session_id": session_id,
                    "context": json.dumps(context, default=str),
                    "timestamp": datetime.now().isoformat(),
                }
            )

            self.client.upsert(
                collection_name=collection_name,
                points=[point]
            )
            self.log.debug(f"Saved context for session {session_id}")
            return True
        except Exception as e:
            self.log.error(f"Error saving conversation context for session {session_id}: {e}")
            return False

    def delete_conversation_context(self, session_id: str, collection_name: Optional[str] = None) -> bool:
        """
        Delete conversation context from Qdrant.

        Args:
            session_id: Unique session identifier
            collection_name: Optional collection name (uses default if not provided)

        Returns:
            True if successful, False otherwise
        """
        if collection_name is None:
            collection_name = self.default_collection

        try:
            point_id = self._get_point_id(session_id)

            self.client.delete(
                collection_name=collection_name,
                points_selector=[point_id]
            )
            self.log.info(f"Deleted context for session {session_id}")
            return True
        except Exception as e:
            self.log.error(f"Error deleting conversation context for session {session_id}: {e}")
            return False

    def health_check(self) -> Dict[str, str]:
        """
        Check Qdrant connection health.

        Returns:
            Dictionary with status information
        """
        try:
            collections_response = self.client.get_collections()
            collections: List[CollectionInfo] = collections_response.collections

            return {
                "status": "connected",
                "url": self.url,
                "default_collection": self.default_collection,
                "collections_count": str(len(collections)),
                "session_timeout_minutes": str(self.session_timeout_minutes),
                "available_collections": ", ".join([c.name for c in collections])
            }
        except Exception as e:
            self.log.error(f"Qdrant health check failed: {e}")
            return {
                "status": "disconnected",
                "url": self.url,
                "error": str(e)
            }


# Convenience functions for backward compatibility and easy access
def get_qdrant_connector() -> QdrantConnector:
    """Get or create a QdrantConnector singleton instance"""
    return QdrantConnector.get_instance()


def get_conversation_context(session_id: str) -> Dict[str, Any]:
    """Retrieve conversation context for a session"""
    connector = get_qdrant_connector()
    return connector.get_conversation_context(session_id)


def save_conversation_context(session_id: str, context: Dict[str, Any]) -> bool:
    """Save conversation context for a session"""
    connector = get_qdrant_connector()
    return connector.save_conversation_context(session_id, context)


def delete_conversation_context(session_id: str) -> bool:
    """Delete conversation context for a session"""
    connector = get_qdrant_connector()
    return connector.delete_conversation_context(session_id)


def check_qdrant_health() -> Dict[str, str]:
    """Check Qdrant connection health"""
    connector = get_qdrant_connector()
    return connector.health_check()