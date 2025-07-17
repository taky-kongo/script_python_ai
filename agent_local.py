# agent_local.py
import os
from dotenv import load_dotenv
from langchain.agents import tool
import datetime

load_dotenv() # Optional

@tool
def get_current_datetime(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Returns the current date and time, formatted according to the provided Python strftime format string.
    Use this tool whenever the user asks for the current date, time, or both.
    Example format strings: '%Y-%m-%d' for date, '%H:%M:%S' for time.
    If no format is specified, defaults to '%Y-%m-%d %H:%M:%S'.
    """
    try:
        return datetime.datetime.now().strftime(format)
    except Exception as e:
        return f"Error formatting date/time: {e}"

# List of tools the agent can use
tools = [get_current_datetime]
print("Custom tool defined.")

# agent_local.py (continued)
from langchain_ollama import ChatOllama

def get_agent_llm(model_name="qwen3:8b", temperature=0):
    """Initializes the ChatOllama model for the agent."""
    # Ensure Ollama server is running (ollama serve)
    llm = ChatOllama(
        model=model_name,
        temperature=temperature # Lower temperature for more predictable tool use
        # Consider increasing num_ctx if expecting long conversations or complex reasoning
        # num_ctx=8192
    )
    print(f"Initialized ChatOllama agent LLM with model: {model_name}")
    return llm

# agent_llm = get_agent_llm() # Call this later

# agent_local.py (continued)
from langchain import hub

def get_agent_prompt(prompt_hub_name="hwchase17/openai-tools-agent"):
    """Pulls the agent prompt template from LangChain Hub."""
    # This prompt is designed for OpenAI but often works well with other tool-calling models.
    # Alternatively, define a custom ChatPromptTemplate.
    prompt = hub.pull(prompt_hub_name)
    print(f"Pulled agent prompt from Hub: {prompt_hub_name}")
    # print("Prompt Structure:")
    # prompt.pretty_print() # Uncomment to see the prompt structure
    return prompt

# agent_prompt = get_agent_prompt() # Call this later

# agent_local.py (continued)
from langchain.agents import create_tool_calling_agent

def build_agent(llm, tools, prompt):
    """Builds the tool-calling agent runnable."""
    agent = create_tool_calling_agent(llm, tools, prompt)
    print("Agent runnable created.")
    return agent

# agent_runnable = build_agent(agent_llm, tools, agent_prompt) # Call this later

# agent_local.py (continued)
from langchain.agents import AgentExecutor

def create_agent_executor(agent, tools):
    """Creates the agent executor."""
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True # Set to True to see agent thoughts and tool calls
    )
    print("Agent executor created.")
    return agent_executor

# agent_executor = create_agent_executor(agent_runnable, tools) # Call this later

# agent_local.py (continued)

def run_agent(executor, user_input):
    """Runs the agent executor with the given input."""
    print("\nInvoking agent...")
    print(f"Input: {user_input}")
    response = executor.invoke({"input": user_input})
    print("\nAgent Response:")
    print(response['output'])

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Define Tools (already done above)

    # 2. Get Agent LLM
    agent_llm = get_agent_llm(model_name="qwen3:8b") # Use the chosen Qwen 3 model

    # 3. Get Agent Prompt
    agent_prompt = get_agent_prompt()

    # 4. Build Agent Runnable
    agent_runnable = build_agent(agent_llm, tools, agent_prompt)

    # 5. Create Agent Executor
    agent_executor = create_agent_executor(agent_runnable, tools)

    # 6. Run Agent
    run_agent(agent_executor, "What is the current date?")
    run_agent(agent_executor, "What time is it right now? Use HH:MM format.")
    run_agent(agent_executor, "Tell me a joke.") # Should not use the tool