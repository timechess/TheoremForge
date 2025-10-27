from theoremforge.agents.base_agent import BaseAgent
from theoremforge.state import TheoremForgeState, TheoremRetrieverTrace
from typing import Tuple
from openai import AsyncOpenAI
from jinja2 import FileSystemLoader, Environment
from pathlib import Path
import re
from typing import List
import aiohttp
from theoremforge.utils import payload_to_string

env = Environment(loader=FileSystemLoader(Path(__file__).parent / "prompts"))


class TheoremRetrieverAgent(BaseAgent):
    def __init__(
        self, base_url: str, api_key: str, model_name: str, retriever_url: str
    ):
        super().__init__(
            agent_name="theorem_retriever_agent",
            subagents=[],
        )
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.session = aiohttp.ClientSession()
        self.retriever_url = retriever_url

    def _extract_search_queries(self, text: str) -> List[str]:
        pattern = r"<search>(.*?)</search>"
        matches = re.findall(pattern, text, re.DOTALL)
        return [match.strip() for match in matches][:5]


    def _extract_theorems(self, text: str) -> List[str]:
        pattern = r"<theorem>(.*?)</theorem>"
        matches = re.findall(pattern, text, re.DOTALL)
        return [match.strip() for match in matches]

    async def run(
        self, state: TheoremForgeState, formal_statement: str, **kwargs
    ) -> Tuple[TheoremForgeState, TheoremRetrieverTrace]:
        query_generation_prompt = env.get_template("search_query_generation.j2").render(
            formal_statement=formal_statement
        )
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": query_generation_prompt},
            ],
            **kwargs,
        )
        query_generation_output = response.choices[0].message.content
        queries = self._extract_search_queries(query_generation_output)
        async with self.session.post(
            f"{self.retriever_url}/search", json={"queries": queries, "topk": 5}
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Server error: {error_text}")
            result = await response.json()
            query_results = sum(result["results"], [])
        theorems = [
            payload_to_string(result["payload"]) for result in query_results
        ]
        theorem_selection_prompt = env.get_template("theorem_selection.j2").render(
            formal_statement=formal_statement, theorems="\n".join(theorems)
        )
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": theorem_selection_prompt},
            ],
            **kwargs,
        )
        theorem_selection_output = response.choices[0].message.content
        theorem_names = self._extract_theorems(theorem_selection_output)
        selected_theorems = [
            theorem["payload"]
            for theorem in query_results
            if theorem["payload"]["name"] in theorem_names
        ]
        trace = TheoremRetrieverTrace(
            step="theorem_retrieval",
            agent_name="theorem_retriever_agent",
            query_generation_prompt=query_generation_prompt,
            query_generation_output=query_generation_output,
            query_results=query_results,
            theorem_selection_prompt=theorem_selection_prompt,
            theorem_selection_output=theorem_selection_output,
            theorem_selection_results=selected_theorems,
        )
        state.trace.append(trace)
        return state, trace
