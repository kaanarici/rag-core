from __future__ import annotations

import asyncio
import sys

from agents import Agent, Runner

from rag_core import RAGCore


async def main() -> None:
    question = " ".join(sys.argv[1:]) or "How can customers pay invoices?"
    async with RAGCore.from_env(index="company-docs") as rag:
        agent = Agent(
            name="Docs assistant",
            instructions=(
                "Use the retrieval tool for questions about company documents. "
                "Cite the sources returned by the tool."
            ),
            tools=[rag.tool()],
        )
        result = await Runner.run(agent, question)
        print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
