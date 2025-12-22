"""
SQLite client for TheoremForge.

This module provides an async SQLite client with two data models:
- TheoremForgeState: Stores theorem proving states
- AgentTrace: Stores agent execution traces
"""

import os
import json
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path
import aiosqlite
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


class SQLiteClient:
    """Async SQLite client for TheoremForge."""

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize SQLite client.

        Args:
            db_path: Path to SQLite database file. If not provided, uses DATABASE_PATH env var
                    or defaults to ./theoremforge.db
        """
        if db_path:
            self.db_path = db_path
        else:
            self.db_path = os.getenv("DATABASE_PATH", "./theoremforge.db")

        # Ensure directory exists
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)

        self.connection: Optional[aiosqlite.Connection] = None
        self._connected = False

    async def connect(self):
        """Connect to SQLite database and create tables."""
        if self._connected:
            logger.warning("SQLite client already connected")
            return

        try:
            self.connection = await aiosqlite.connect(self.db_path)
            # Enable foreign keys
            await self.connection.execute("PRAGMA foreign_keys = ON")
            # Set WAL mode for better concurrency
            await self.connection.execute("PRAGMA journal_mode = WAL")

            # Create tables
            await self._create_tables()

            self._connected = True
            logger.info(f"Connected to SQLite database: {self.db_path}")

        except Exception as e:
            logger.error(f"Failed to connect to SQLite: {e}")
            raise

    async def _create_tables(self):
        """Create database tables."""
        async with self.connection.cursor() as cursor:
            # Create TheoremForgeState table
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS theorem_forge_states (
                    id TEXT PRIMARY KEY,
                    informal_statement TEXT,
                    formal_statement TEXT,
                    formal_proof TEXT,
                    subgoals TEXT,
                    parent_id TEXT,
                    success INTEGER DEFAULT 0,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)
            await cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_states_parent_id ON theorem_forge_states(parent_id)"
            )
            await cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_states_success ON theorem_forge_states(success)"
            )

            # Create AgentTrace table
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_traces (
                    id TEXT PRIMARY KEY,
                    state_id TEXT,
                    prompt TEXT,
                    response TEXT,
                    input_token INTEGER DEFAULT 0,
                    output_token INTEGER DEFAULT 0,
                    agent_name TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)
            await cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_traces_state_id ON agent_traces(state_id)"
            )
            await cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_traces_agent_name ON agent_traces(agent_name)"
            )

            await self.connection.commit()

    async def disconnect(self):
        """Disconnect from SQLite database."""
        if not self._connected:
            return

        if self.connection:
            await self.connection.close()
            self._connected = False
            logger.info("Disconnected from SQLite")

    async def _ensure_connected(self):
        """Ensure database is connected."""
        if not self._connected:
            await self.connect()

    # ==================== TheoremForgeState Operations ====================

    async def create_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new TheoremForgeState.

        Args:
            state: Dictionary containing state data with keys:
                - id (optional, will generate if not provided)
                - informal_statement
                - formal_statement
                - formal_proof
                - subgoals (list, will be JSON serialized)
                - parent_id
                - success (bool)

        Returns:
            Created state dictionary
        """
        await self._ensure_connected()

        # Generate ID if not provided
        state_id = state.get("id") or str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        # Serialize subgoals list to JSON
        subgoals = state.get("subgoals")
        subgoals_json = json.dumps(subgoals) if subgoals is not None else None

        # Convert success bool to integer
        success = 1 if state.get("success", False) else 0

        sql = """
            INSERT INTO theorem_forge_states 
            (id, informal_statement, formal_statement, formal_proof, subgoals, parent_id, success, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        values = (
            state_id,
            state.get("informal_statement"),
            state.get("formal_statement"),
            state.get("formal_proof"),
            subgoals_json,
            state.get("parent_id"),
            success,
            now,
            now,
        )

        async with self.connection.cursor() as cursor:
            await cursor.execute(sql, values)
            await self.connection.commit()

        logger.debug(f"Created TheoremForgeState with id={state_id}")

        return {
            "id": state_id,
            "informal_statement": state.get("informal_statement"),
            "formal_statement": state.get("formal_statement"),
            "formal_proof": state.get("formal_proof"),
            "subgoals": subgoals,
            "parent_id": state.get("parent_id"),
            "success": state.get("success", False),
            "created_at": now,
            "updated_at": now,
        }

    async def get_state(self, state_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a TheoremForgeState by ID.

        Args:
            state_id: State ID

        Returns:
            State dictionary or None if not found
        """
        await self._ensure_connected()

        sql = "SELECT * FROM theorem_forge_states WHERE id = ?"

        async with self.connection.cursor() as cursor:
            await cursor.execute(sql, (state_id,))
            row = await cursor.fetchone()

            if row is None:
                return None

            columns = [desc[0] for desc in cursor.description]
            result = dict(zip(columns, row))

            # Deserialize subgoals JSON
            if result.get("subgoals"):
                try:
                    result["subgoals"] = json.loads(result["subgoals"])
                except json.JSONDecodeError:
                    result["subgoals"] = None

            # Convert success integer to bool
            result["success"] = bool(result.get("success", 0))

            return result

    async def update_state(self, state_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a TheoremForgeState.

        Args:
            state_id: State ID
            updates: Dictionary of fields to update

        Returns:
            True if updated, False if not found
        """
        await self._ensure_connected()

        # Build SET clause dynamically
        set_parts = []
        values = []

        for key, value in updates.items():
            if key == "id":
                continue  # Don't update ID
            if key == "subgoals":
                set_parts.append("subgoals = ?")
                values.append(json.dumps(value) if value is not None else None)
            elif key == "success":
                set_parts.append("success = ?")
                values.append(1 if value else 0)
            elif key in ("informal_statement", "formal_statement", "formal_proof", "parent_id"):
                set_parts.append(f"{key} = ?")
                values.append(value)

        if not set_parts:
            return False

        # Add updated_at
        set_parts.append("updated_at = ?")
        values.append(datetime.utcnow().isoformat())
        values.append(state_id)

        sql = f"UPDATE theorem_forge_states SET {', '.join(set_parts)} WHERE id = ?"

        async with self.connection.cursor() as cursor:
            await cursor.execute(sql, values)
            await self.connection.commit()
            return cursor.rowcount > 0

    async def delete_state(self, state_id: str) -> bool:
        """
        Delete a TheoremForgeState.

        Args:
            state_id: State ID

        Returns:
            True if deleted, False if not found
        """
        await self._ensure_connected()

        sql = "DELETE FROM theorem_forge_states WHERE id = ?"

        async with self.connection.cursor() as cursor:
            await cursor.execute(sql, (state_id,))
            # Delete the traces of the state
            await self.delete_traces_by_state(state_id)
            await self.connection.commit()
            return cursor.rowcount > 0

    async def find_states(
        self,
        filter: Optional[Dict[str, Any]] = None,
        limit: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Find TheoremForgeStates matching filter.

        Args:
            filter: Filter dictionary (e.g., {"success": True, "parent_id": "xxx"})
            limit: Maximum number of results (0 = no limit)

        Returns:
            List of state dictionaries
        """
        await self._ensure_connected()

        sql = "SELECT * FROM theorem_forge_states"
        values = []

        if filter:
            conditions = []
            for key, value in filter.items():
                if key == "success":
                    conditions.append("success = ?")
                    values.append(1 if value else 0)
                elif key in ("id", "informal_statement", "formal_statement", "formal_proof", "parent_id"):
                    conditions.append(f"{key} = ?")
                    values.append(value)
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)

        if limit > 0:
            sql += f" LIMIT {limit}"

        async with self.connection.cursor() as cursor:
            await cursor.execute(sql, values)
            rows = await cursor.fetchall()

            columns = [desc[0] for desc in cursor.description]
            results = []

            for row in rows:
                result = dict(zip(columns, row))
                # Deserialize subgoals JSON
                if result.get("subgoals"):
                    try:
                        result["subgoals"] = json.loads(result["subgoals"])
                    except json.JSONDecodeError:
                        result["subgoals"] = None
                # Convert success to bool
                result["success"] = bool(result.get("success", 0))
                results.append(result)

            return results

    # ==================== AgentTrace Operations ====================

    async def create_trace(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new AgentTrace.

        Args:
            trace: Dictionary containing trace data with keys:
                - id (optional, will generate if not provided)
                - state_id
                - prompt
                - response (list of strings)
                - input_token
                - output_token
                - agent_name

        Returns:
            Created trace dictionary
        """
        await self._ensure_connected()

        # Generate ID if not provided
        trace_id = trace.get("id") or str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        response_json = json.dumps(trace.get("response"), ensure_ascii=False)
        sql = """
            INSERT INTO agent_traces 
            (id, state_id, prompt, response, input_token, output_token, agent_name, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        values = (
            trace_id,
            trace.get("state_id"),
            trace.get("prompt"),
            response_json,
            trace.get("input_token", 0),
            trace.get("output_token", 0),
            trace.get("agent_name"),
            now,
            now,
        )

        async with self.connection.cursor() as cursor:
            await cursor.execute(sql, values)
            await self.connection.commit()

        logger.debug(f"Created AgentTrace with id={trace_id}")

        return {
            "id": trace_id,
            "state_id": trace.get("state_id"),
            "prompt": trace.get("prompt"),
            "response": response_json,
            "input_token": trace.get("input_token", 0),
            "output_token": trace.get("output_token", 0),
            "agent_name": trace.get("agent_name"),
            "created_at": now,
            "updated_at": now,
        }

    async def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an AgentTrace by ID.

        Args:
            trace_id: Trace ID

        Returns:
            Trace dictionary or None if not found
        """
        await self._ensure_connected()

        sql = "SELECT * FROM agent_traces WHERE id = ?"

        async with self.connection.cursor() as cursor:
            await cursor.execute(sql, (trace_id,))
            row = await cursor.fetchone()

            if row is None:
                return None

            columns = [desc[0] for desc in cursor.description]
            result = dict(zip(columns, row))
            result["response"] = json.loads(result["response"])
            return result

    async def find_traces(
        self,
        filter: Optional[Dict[str, Any]] = None,
        limit: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Find AgentTraces matching filter.

        Args:
            filter: Filter dictionary (e.g., {"state_id": "xxx", "agent_name": "yyy"})
            limit: Maximum number of results (0 = no limit)

        Returns:
            List of trace dictionaries
        """
        await self._ensure_connected()

        sql = "SELECT * FROM agent_traces"
        values = []

        if filter:
            conditions = []
            for key, value in filter.items():
                if key in ("id", "state_id", "agent_name"):
                    conditions.append(f"{key} = ?")
                    values.append(value)
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)

        if limit > 0:
            sql += f" LIMIT {limit}"

        async with self.connection.cursor() as cursor:
            await cursor.execute(sql, values)
            rows = await cursor.fetchall()

            columns = [desc[0] for desc in cursor.description]
            results = []
            for row in rows:
                result = dict(zip(columns, row))
                result["response"] = json.loads(result["response"])
                results.append(result)
            return results

    async def delete_trace(self, trace_id: str) -> bool:
        """
        Delete an AgentTrace.

        Args:
            trace_id: Trace ID

        Returns:
            True if deleted, False if not found
        """
        await self._ensure_connected()

        sql = "DELETE FROM agent_traces WHERE id = ?"

        async with self.connection.cursor() as cursor:
            await cursor.execute(sql, (trace_id,))
            await self.connection.commit()
            return cursor.rowcount > 0

    async def delete_traces_by_state(self, state_id: str) -> int:
        """
        Delete all AgentTraces for a given state.

        Args:
            state_id: State ID

        Returns:
            Number of deleted traces
        """
        await self._ensure_connected()

        sql = "DELETE FROM agent_traces WHERE state_id = ?"

        async with self.connection.cursor() as cursor:
            await cursor.execute(sql, (state_id,))
            await self.connection.commit()
            return cursor.rowcount

