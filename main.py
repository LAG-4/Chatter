import os
from agno.agent import Agent, RunResponse
from agno.models.google import Gemini
from agno.tools.reasoning import ReasoningTools

from dotenv import load_dotenv

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
agent = Agent(
    model=Gemini(
        id="gemini-2.5-flash",
    ),
    instructions="You are a helpful assistant. You answer users questions funnily and end each answer with a deez nuts joke.",
    markdown=True,
    tools=[ReasoningTools(add_instructions=True)],
)

response= agent.run("What is the capital of France?",show_full_reasoning=True,
        stream_intermediate_steps=True,)
print(response)
