# test_ollama.py

#from langchain_ollama import ChatOllama
#llm = ChatOllama(model="gemma3:1b", base_url="http://127.0.0.1:11434")
#print(llm.invoke("Which tool am I talking to? Answer in 3 words."))


from crewai import Agent, Task, Crew, Process , LLM
from langchain_ollama import ChatOllama # Using the new, recommended import
from langchain_google_genai import ChatGoogleGenerativeAI
import os


# --- Configuration ---
# Use the exact settings that worked in test_ollama.py
local_llm = ChatGoogleGenerativeAI(model="gemini/gemini-1.5-flash", verbose=True, google_api_key=os.getenv("GEMINI_API_KEY"), temperature=0.5)

llm_model_string = "gemini/gemini-2.0-flash"

# --- Agent Definition ---
simple_agent = Agent(
    role='Simple Echo Agent',
    goal='Repeat the user\'s prompt and say hello.',
    backstory='A very simple agent designed to test LLM connectivity.',
    verbose=True, # Set to True to see the thought process
    llm=llm_model_string, # Explicitly use the working Ollama instance
    allow_delegation=False
)

print("LLM IN USE:", type(local_llm).__name__, getattr(local_llm, "model", None))


# --- Task Definition ---
simple_task = Task(
    description='Select any random country and tell what is the capital city? Answer concisely.',
    agent=simple_agent,
    expected_output='A one-word answer stating the capital city.'
)

# --- Crew Execution ---
test_crew = Crew(
    agents=[simple_agent],
    tasks=[simple_task],
    process=Process.sequential,
    memory=False,
    llm=local_llm,
    verbose=True # See high-level crew execution logs
)

print("Starting CrewAI test ...")
result = test_crew.kickoff()

print("\n\n✅ Crew Final Result:")
print(result)