from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain.schema import SystemMessage, HumanMessage
from typing import TypedDict

# Initialize separate LLM instances for each agent part
# Placeholder models - user can replace with specific models as needed
planner_llm = ChatOllama(model="thinking-model", temperature=0.1)
codegen_llm = ChatOllama(model="codellama:instruct", temperature=0.1)
toolgen_llm = ChatOllama(model="", temperature=0.1)
tester_llm = ChatOllama(model="tester-model", temperature=0.1)
result_llm = ChatOllama(model="result-model", temperature=0.1)


class AgentState(TypedDict):
    prompt: str
    plan: str
    code: str
    tools_code: str
    test_results: str
    final_result: str
    error: str
    status: str


# Initial state
def initial_state(task: str) -> AgentState:
    return {
        "prompt": task,
        "plan": None,
        "code": None,
        "tools_code": None,
        "test_results": None,
        "final_result": None,
        "error": None,
        "status": None
    }


# Planner Node with web search and GitHub referencing tool placeholders
def planner_node(state: AgentState) -> AgentState:
    prompt = state["prompt"]
    messages = [
        SystemMessage(content="You are a Planner agent with access to web search and GitHub repo referencing tools. Break down the user's task into a detailed plan and workflow."),
        HumanMessage(content=f"Task: {prompt}")
    ]
    plan = planner_llm.invoke(messages).content
    state["plan"] = plan
    return state


# Code Generator Node
def code_generator_node(state: AgentState) -> AgentState:
    plan = state["plan"]
    messages = [
        SystemMessage(content="""
You are a Code Generator agent. Based on the plan, generate the main code for the agent.
The code must be runnable and complete.
"""),
        HumanMessage(content=f"PLAN:\n{plan}")
    ]
    code = codegen_llm.invoke(messages).content
    state["code"] = code
    return state


# Tool Generator Node
def tool_generator_node(state: AgentState) -> AgentState:
    plan = state["plan"]
    messages = [
        SystemMessage(content="""
You are a Tool Generator agent. Based on the plan, generate any necessary tool code or integrations (e.g., web search, GitHub API).
The code must be runnable and complete.
"""),
        HumanMessage(content=f"PLAN:\n{plan}")
    ]
    tools_code = toolgen_llm.invoke(messages).content
    state["tools_code"] = tools_code
    return state


# Tester Node
def tester_node(state: AgentState) -> AgentState:
    try:
        # Combine main code and tools code for testing
        full_code = (state["code"] or "") + "\n" + (state["tools_code"] or "")
        exec(full_code, {})
        test_results = "Tests passed successfully."
        state["test_results"] = test_results
        state["status"] = "success"
    except Exception as e:
        state["test_results"] = str(e)
        state["status"] = "retry"
        state["error"] = str(e)
    return state


# Result Node
def result_node(state: AgentState) -> AgentState:
    if state["status"] == "success":
        messages = [
            SystemMessage(content="You are a Result agent. Compile the final results and summary of the agent creation process."),
            HumanMessage(content=f"Plan:\n{state['plan']}\nTest Results:\n{state['test_results']}")
        ]
        final_result = result_llm.invoke(messages).content
        state["final_result"] = final_result
    else:
        state["final_result"] = f"Agent creation failed with error: {state.get('error', 'Unknown error')}"
    return state


# Evaluation condition for LangGraph flow control
def eval_condition(state: AgentState):
    return state.get("status", "retry")

# Verification Node - Planner verifies all reports
def verification_node(state: AgentState) -> AgentState:
    messages = [
        SystemMessage(content="You are the Planner agent verifying all reports from other agents. Check if all tasks are completed correctly and provide feedback."),
        HumanMessage(content=f"Plan:\n{state.get('plan')}\nTest Results:\n{state.get('test_results')}\nFinal Result:\n{state.get('final_result')}")
    ]
    verification_feedback = planner_llm.invoke(messages).content
    state["verification_feedback"] = verification_feedback
    return state

