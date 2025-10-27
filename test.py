from theoremforge.lean_server.client import RemoteVerifier
import asyncio
verifier = RemoteVerifier("http://localhost:8001")

statement = """import Mathlib

theorem foo : 1 + 1 = 2 := by
  exact
"""

async def main():
    valid, messages = await verifier.verify(statement)
    print(messages)

if __name__ == "__main__":
    asyncio.run(main())