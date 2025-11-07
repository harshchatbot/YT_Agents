from crewai import Task

class CodeCrewTasks:
    def __init__(self, agent):
        # Agent will be passed in from the main file
        self.agent = agent

    def implement_feature_task(self):
        return Task(
            description=(
                "**Task Goal:** Implement Core Feature\n"
                "Write a complete Python class named `DataValidator`. It must include two public methods:\n"
                "1. A method for **schema validation** (using the Pydantic library).\n"
                "2. A method for **basic data type checking** (e.g., check if a value is an integer or string).\n"
                "Ensure the final output file (`data_validator.py`) contains full docstrings for the class "
                "and all methods, and include a small execution block to demonstrate functionality."
            ),
            expected_output=(
                "A single, runnable Python file named `data_validator.py` containing the fully "
                "tested `DataValidator` class and its methods."
            ),
            agent=self.agent,
            output_file='data_validator.py' # Automatically writes the final output to this file
        )
    
    # Add other tasks here if needed (e.g., refactoring_task, debugging_task)