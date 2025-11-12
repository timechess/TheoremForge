import aiohttp
from typing import List, Dict, Any


class Retriever:
    def __init__(self, url: str) -> None:
        self.url = url
        self.session = None

    async def __aenter__(self):
        await self._get_session()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.close()

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp client session."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session

    async def search_theorems(
        self, queries: List[str], topk: int
    ) -> List[Dict[str, Any]]:
        if self.session is None:
            await self._get_session()
        async with self.session.post(
            f"{self.url}/search",
            json={"queries": queries, "topk": topk, "collection_name": "theorem"},
        ) as response:
            if response.status != 200:
                raise Exception(f"Server error: {await response.text()}")
            return await response.json()

    async def search_definitions(
        self, queries: List[str], topk: int
    ) -> List[Dict[str, Any]]:
        if self.session is None:
            await self._get_session()
        async with self.session.post(
            f"{self.url}/search",
            json={"queries": queries, "topk": topk, "collection_name": "definition"},
        ) as response:
            if response.status != 200:
                raise Exception(f"Server error: {await response.text()}")
            return await response.json()

    async def close(self):
        """Close the HTTP client session."""
        if self.session is not None:
            await self.session.close()
            self.session = None
