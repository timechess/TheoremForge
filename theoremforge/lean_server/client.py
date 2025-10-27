import aiohttp

class RemoteVerifier:
    """
    A client for the Lean verification server.

    This class provides methods to verify Lean code by making HTTP requests
    to a running verification server instance.

    Attributes:
        url (str): Base URL of the verification server.
    """

    def __init__(self, url: str):
        """
        Initialize a new RemoteVerifier.

        Args:
            url (str): Base URL of the verification server (e.g., "http://localhost:8000").
        """
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

    async def verify(
        self, code: str, allow_sorry: bool = True
    ) -> tuple[bool, list[dict], str]:
        """
        Verify a single piece of Lean code by making a request to the verification server.

        Args:
            code (str): The Lean code to verify.
            allow_sorry (bool, optional): Whether to allow 'sorry' in the code. Defaults to True.

        Returns:
            tuple[bool, list[dict]]: A tuple containing:
                - bool: Whether the verification was successful.
                - list[dict]: Messages from the verification process.
        """
        session = await self._get_session()
        async with session.post(
            f"{self.url}/verify", json={"code": code, "allow_sorry": allow_sorry}
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Server error: {error_text}")
            result = await response.json()
            return result["valid"], result["messages"], result["error_str"]

    async def extract_subgoals(self, code: str) -> list[str]:
        """
        Extract subgoals from a piece of Lean code by making a request to the verification server.

        Args:
            code (str): The Lean code to extract subgoals from.

        Returns:
            list[str]: A list of subgoals.
        """
        session = await self._get_session()
        async with session.post(
            f"{self.url}/extract_subgoals", json={"code": code}
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Server error: {error_text}")
            result = await response.json()
            return result["subgoals"]

    async def close(self):
        """Close the HTTP client session."""
        if self.session is not None:
            await self.session.close()
            self.session = None

    def __del__(self):
        """Ensure the session is closed when the verifier is deleted."""
        if self.session is not None:
            import asyncio

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.close())
                else:
                    loop.run_until_complete(self.close())
            except Exception:
                pass
