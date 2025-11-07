import os
import time
from crewai import Crew, Process
from dotenv import load_dotenv

# 1. Import Agents, Tasks, and Tools
from agents import CodeCrewAgents
from tasks import CodeCrewTasks
from tools import CODE_CREW_TOOLS # The list of all your tools

# 2. Set Up Environment Variables (CRITICAL for LLM and Tools)
# You should have these defined in a .env file or directly exported in your terminal
# For Gemini/Google:
# os.environ["GEMINI_API_KEY"] = "YOUR_GEMINI_API_KEY" 
# os.environ["OPENAI_MODEL_NAME"] = "gemini-2.5-flash" 
# For Serper (search tool):
# os.environ["SERPER_API_KEY"] = "YOUR_SERPER_API_KEY"

# Load environment variables...
load_dotenv()
# Ensure the model name is set for your LLM (Gemini 1.5 Flash)
os.environ["OPENAI_MODEL_NAME"] = "gemini-1.5-flash"
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

### --- Main Execution ---

if __name__ == "__main__":
    # 3. Instantiate Components

    # Pass the list of tools to the Agents class
    agents = CodeCrewAgents(tools=CODE_CREW_TOOLS) 
    
    # Create the Agent instance
    coding_agent = agents.coding_agent()
    
    # Pass the agent to the Tasks class
    tasks = CodeCrewTasks(agent=coding_agent)
    
    # Create the Task instance
    feature_task = tasks.implement_feature_task()

    # 4. Form the Crew
    tech_crew = Crew(
        agents=[coding_agent],     # List of agents in the crew
        tasks=[feature_task],      # List of tasks to be executed
        process=Process.sequential, # Tasks are executed one after the other (simple for one agent)
        verbose=True                 # Detailed logging of agent thoughts and tool usage
    )

    # 5. Kick off the Crew Execution
    print("🚀 Starting the Code Generation Crew...")
    
    try:
        result = tech_crew.kickoff()
        print("\n#############################################")
        print("✅ Crew Execution Finished Successfully!")
        print("Final Output should be in: data_validator.py")
        print("#############################################\n")
        # The result here is usually the output of the final task
        print(result) 
        
    except Exception as e:
        # ⚠️ This is where you would handle the rate limit failure you saw previously
        print(f"\n❌ Crew Execution Failed: {e}")
        print("🚨 Waiting 10 seconds to recover from potential rate limiting...")
        time.sleep(10) # Using time.sleep() as discussed earlier
        # You would typically have a retry loop here in a production environment