from crewai import Agent
# Assuming you will define and import your tools from tools.py
# from .tools import file_io_tool, serper_search_tool 

llm_model_string = "gemini/gemini-2.0-flash"


class CodeCrewAgents:
    def __init__(self, tools):
        # Tools will be passed in from the main file
        self.tools = tools 

    def coding_agent(self):
        return Agent(
            role="Senior Python Developer",
            goal="Craft well-designed and thought-out code that adheres to best practices and solves complex problems.",
            backstory=(
                "You are a senior Python developer with extensive experience in software architecture, "
                "writing clean, tested, and efficient code. You ensure every line is justified."
            ),
            tools=self.tools,  # Assign the tools here
            allow_code_execution=False,
            # Set a low max_iter to prevent excessive tool usage (like your previous 9 searches)
            max_iter=5, 
            verbose=True,
            llm=llm_model_string
        )

# Add other agents here if you expand your crew (e.g., QA_Agent, Product_Owner_Agent)