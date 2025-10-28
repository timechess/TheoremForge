"""
MongoDB client for TheoremForge.

This module provides an async MongoDB client to replace Prisma.
"""

import os
from typing import Optional, Dict, Any, List
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

class MongoDBClient:
    """Async MongoDB client for TheoremForge."""

    def __init__(self, connection_url: Optional[str] = None):
        """
        Initialize MongoDB client.

        Args:
            connection_url: MongoDB connection URL. If not provided, uses DATABASE_URL env var.
        """
        self.connection_url = connection_url or os.getenv(
            "DATABASE_URL",
            "mongodb://admin:password@localhost:27017/theoremforge?authSource=admin"
        )
        self.client: Optional[AsyncIOMotorClient] = None
        self.db: Optional[AsyncIOMotorDatabase] = None
        self._connected = False

        # Cache collection instances to avoid recreation
        self._collections: Dict[str, Any] = {}

    async def connect(self):
        """Connect to MongoDB."""
        if self._connected:
            logger.warning("MongoDB client already connected")
            return

        try:
            self.client = AsyncIOMotorClient(self.connection_url)
            # Extract database name from URL or use default
            db_name = "theoremforge"
            if "?" in self.connection_url:
                path = self.connection_url.split("?")[0].split("/")[-1]
                if path:
                    db_name = path

            self.db = self.client[db_name]

            # Test connection
            await self.client.admin.command('ping')
            self._connected = True
            logger.info(f"Connected to MongoDB database: {db_name}")

        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    async def disconnect(self):
        """Disconnect from MongoDB."""
        if not self._connected:
            return

        if self.client:
            self.client.close()
            self._connected = False
            logger.info("Disconnected from MongoDB")

    @property
    def theoremforgestate(self):
        """Access TheoremForgeState collection."""
        if "theoremforgestate" not in self._collections:
            self._collections["theoremforgestate"] = TheoremForgeStateCollection(self.db)
        return self._collections["theoremforgestate"]

    @property
    def provertrace(self):
        """Access ProverTrace collection."""
        if "provertrace" not in self._collections:
            self._collections["provertrace"] = ProverTraceCollection(self.db)
        return self._collections["provertrace"]

    @property
    def selfcorrectiontrace(self):
        """Access SelfCorrectionTrace collection."""
        if "selfcorrectiontrace" not in self._collections:
            self._collections["selfcorrectiontrace"] = SelfCorrectionTraceCollection(self.db)
        return self._collections["selfcorrectiontrace"]

    @property
    def theoremretrievaltrace(self):
        """Access TheoremRetrievalTrace collection."""
        if "theoremretrievaltrace" not in self._collections:
            self._collections["theoremretrievaltrace"] = TheoremRetrievalTraceCollection(self.db)
        return self._collections["theoremretrievaltrace"]

    @property
    def informalprooftrace(self):
        """Access InformalProofTrace collection."""
        if "informalprooftrace" not in self._collections:
            self._collections["informalprooftrace"] = InformalProofTraceCollection(self.db)
        return self._collections["informalprooftrace"]

    @property
    def proofsketchtrace(self):
        """Access ProofSketchTrace collection."""
        if "proofsketchtrace" not in self._collections:
            self._collections["proofsketchtrace"] = ProofSketchTraceCollection(self.db)
        return self._collections["proofsketchtrace"]

    @property
    def proofassemblytrace(self):
        """Access ProofAssemblyTrace collection."""
        if "proofassemblytrace" not in self._collections:
            self._collections["proofassemblytrace"] = ProofAssemblyTraceCollection(self.db)
        return self._collections["proofassemblytrace"]


class BaseCollection:
    """Base class for MongoDB collections."""

    def __init__(self, db: AsyncIOMotorDatabase, collection_name: str):
        self.collection = db[collection_name]

    async def create(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new document."""
        # Make a copy to avoid mutating the input
        doc_data = data.copy()

        # Add timestamps
        now = datetime.utcnow()
        doc_data["createdAt"] = now
        doc_data["updatedAt"] = now

        result = await self.collection.insert_one(doc_data)
        doc_data["_id"] = str(result.inserted_id)

        # Log only the ID, not the entire document (which can be huge)
        logger.debug(f"Created document with id={doc_data.get('id', doc_data['_id'])} in {self.collection.name}")

        return doc_data

    async def find_one(self, filter: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find a single document."""
        return await self.collection.find_one(filter)

    async def find_many(self, filter: Dict[str, Any] = None, limit: int = 0) -> List[Dict[str, Any]]:
        """Find multiple documents."""
        cursor = self.collection.find(filter or {})
        if limit > 0:
            cursor = cursor.limit(limit)
        return await cursor.to_list(length=None)

    async def update_one(self, filter: Dict[str, Any], update: Dict[str, Any]) -> bool:
        """Update a single document."""
        update["updatedAt"] = datetime.utcnow()
        result = await self.collection.update_one(
            filter,
            {"$set": update}
        )
        return result.modified_count > 0

    async def delete_one(self, filter: Dict[str, Any]) -> bool:
        """Delete a single document."""
        result = await self.collection.delete_one(filter)
        return result.deleted_count > 0


class TheoremForgeStateCollection(BaseCollection):
    """Collection for TheoremForge states."""

    def __init__(self, db: AsyncIOMotorDatabase):
        super().__init__(db, "theorem_forge_states")


class ProverTraceCollection(BaseCollection):
    """Collection for prover traces."""

    def __init__(self, db: AsyncIOMotorDatabase):
        super().__init__(db, "prover_traces")


class SelfCorrectionTraceCollection(BaseCollection):
    """Collection for self-correction traces."""

    def __init__(self, db: AsyncIOMotorDatabase):
        super().__init__(db, "self_correction_traces")


class TheoremRetrievalTraceCollection(BaseCollection):
    """Collection for theorem retrieval traces."""

    def __init__(self, db: AsyncIOMotorDatabase):
        super().__init__(db, "theorem_retrieval_traces")


class InformalProofTraceCollection(BaseCollection):
    """Collection for informal proof traces."""

    def __init__(self, db: AsyncIOMotorDatabase):
        super().__init__(db, "informal_proof_traces")


class ProofSketchTraceCollection(BaseCollection):
    """Collection for proof sketch traces."""

    def __init__(self, db: AsyncIOMotorDatabase):
        super().__init__(db, "proof_sketch_traces")


class ProofAssemblyTraceCollection(BaseCollection):
    """Collection for proof assembly traces."""

    def __init__(self, db: AsyncIOMotorDatabase):
        super().__init__(db, "proof_assembly_traces")

