import os
# Importing the confirmed correct tool names
from crewai_tools import SerperDevTool, FileReadTool, FileWriterTool 

# --- Initialization ---

# 1. Web Search Tool (for looking up documentation)
serper_search_tool = SerperDevTool()

# 2. File Read Tool
# IMPORTANT: Specify a directory (the current directory './') for the agent to read from
file_read_tool = FileReadTool(file_path='./') 

# 3. File Write Tool
# IMPORTANT: Specify a directory (the current directory './') for the agent to write to
file_write_tool = FileWriterTool(file_path='./') 

# --- List of Tools to be used in agents.py ---
CODE_CREW_TOOLS = [
    serper_search_tool, 
    file_read_tool,
    file_write_tool
]