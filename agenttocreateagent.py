from langgraph.graph import StateGraph, END
from nodes import initial_state, planner_node, code_generator_node, tool_generator_node, tester_node, result_node, verification_node, AgentState, eval_condition


#format of the agent construction
#--------------------------------
#- Node 1: Receive question
#- Node 2: Planner creates plan
#- Node 3: Code Generator generates main code
#- Node 4: Tool Generator generates tool code
#- Node 5: Tester tests combined code
#- Node 6: Result compiles final results
#- Node 7: Verification node (Planner verifies all reports)
#- Using LangGraph to link nodes
# -------------------------------


# LangGraph Agent construction function
def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("planner", planner_node)
    builder.add_node("code_generator", code_generator_node)
    builder.add_node("tool_generator", tool_generator_node)
    builder.add_node("tester", tester_node)
    builder.add_node("result", result_node)
    builder.add_node("verification", verification_node)

    builder.set_entry_point("planner")
    builder.add_edge("planner", "code_generator")
    builder.add_edge("code_generator", "tool_generator")
    builder.add_edge("tool_generator", "tester")
    builder.add_edge("tester", "result")
    builder.add_edge("result", "verification")

    # Defining the evaluation condition for tester node
    builder.add_conditional_edges("tester", eval_condition)

    return builder.compile()




#Running the agent
if __name__ == "__main__":
    user_task = input("Enter the task for the sophisticated multi-agent system:\n> ")
    graph = build_graph()
    final_state = graph.invoke(initial_state(user_task))

    print("\n📋 PLAN:\n")
    print(final_state.get("plan", "No plan generated."))

    print("\n✅ GENERATED CODE:\n")
    print(final_state.get("code", "No code generated."))

    print("\n🛠️ TOOLS CODE:\n")
    print(final_state.get("tools_code", "No tools code generated."))

    print("\n🧪 TEST RESULTS:\n")
    print(final_state.get("test_results", "No test results."))

    print("\n📄 FINAL RESULT:\n")
    print(final_state.get("final_result", "No final result."))

    print("\n🔍 VERIFICATION FEEDBACK:\n")
    print(final_state.get("verification_feedback", "No verification feedback."))

