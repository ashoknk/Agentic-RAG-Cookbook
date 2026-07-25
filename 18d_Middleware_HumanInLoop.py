"""
================================================================================
CONCEPT 2: GUARDRAILS & HUMAN-IN-THE-LOOP (HITL)
================================================================================
This script explores safety architectures using `HumanInTheLoopMiddleware`. When 
autonomous agents are equipped with real-world operational capabilities (e.g., 
sending emails, processing payments, executing database modifications), letting 
them run completely unchecked poses severe operational risks.

This code demonstrates how to securely intercept agent actions, pause the runtime 
state machine, and prompt for human supervisor intervention through 3 behaviors:
1. APPROVAL: Pauses mid-execution to allow an operator to review the agent's 
   intended action and grant clearance to finalize it.
2. REJECTION: Allows an operator to deny tool execution, terminating or 
   redirecting the workflow if the agent acts erroneously.
3. INLINE EDITING: Allows a supervisor to inspect parameters (e.g., target email 
   addresses, subject text) and correct them mid-flight before letting it execute.

Significance: This showcases production-grade compliance guardrails, assuring that 
high-stakes actions are never completed without verified human consensus.
================================================================================
"""

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage
from langgraph.types import Command

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Tools Setup
def read_email_tool(email_id: str) -> str:
    """Mock function to read an email by its ID."""
    return f"Email content for ID: {email_id}"

def send_email_tool(recipient: str, subject: str, body: str) -> str:
    """Mock function to send an email."""
    return f"Email sent to {recipient} with subject '{subject}'"

GPT_MODEL_AGENT = "gpt-4o-mini"

# ------------------------------------------------------------------------------
# 1. APPROVAL DEMONSTRATION FLOW
# ------------------------------------------------------------------------------
print("\n=== Scenario 1: Approving an Action ===")

# https://reference.langchain.com/python/langchain/agents/middleware/human_in_the_loop/HumanInTheLoopMiddleware

# HumanInTheLoopMiddleware - The Human-in-the-Loop (HITL) middleware lets you add human oversight to agent tool calls. 
# When a model proposes an action that might require review—for example, writing to a file or 
# executing SQL—the middleware can pause execution and wait for a decision.

agent_hitl = create_agent(
    model=GPT_MODEL_AGENT,
    tools=[read_email_tool, send_email_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email_tool": {"allowed_decisions": ["approve", "edit", "reject"]},
                "read_email_tool": False,
            }
        )
    ]
)

config_approveid = {"configurable": {"thread_id": "test-approve"}}
result_approve = agent_hitl.invoke(
    {"messages": [HumanMessage(content="Send email to peterpan@neverland.com with subject 'Hello Pixie' and body 'All the world is made of faith, and trust, and pixie dust.'")]},
    config=config_approveid
)

# Before send_email_tool actually runs, the HumanInTheLoopMiddleware halts execution.
# Under the hood, LangGraph pauses state execution. When a workflow is paused by a Human-in-the-Loop (HITL) interrupt, 
# LangGraph populates the return state dictionary with a special key: "__interrupt__"

# Command - One or more commands to update the graph's state and send messages to nodes.
# https://reference.langchain.com/python/langgraph/types/Command
if "__interrupt__" in result_approve:
    print("⏸️ Action Intercepted! Human operator is approving...")
    result_approve = agent_hitl.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config=config_approveid
    )
    print(f"✅ Executed Result: {result_approve['messages'][-1].content}")



# ------------------------------------------------------------------------------
# 2. REJECTION DEMONSTRATION FLOW
# ------------------------------------------------------------------------------
print("\n=== Scenario 2: Rejecting an Action ===")

agent_reject = create_agent(
    model=GPT_MODEL_AGENT,
    tools=[read_email_tool, send_email_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email_tool": {"allowed_decisions": ["approve", "edit", "reject"]},
                "read_email_tool": False,
            }
        ),
    ],
)

config_rejectid = {"configurable": {"thread_id": "test-reject"}}
result_reject = agent_reject.invoke(
    {"messages": [HumanMessage(content="Send email to tinkerbell@neverland.com with subject 'Follows Peter Pan' and body 'My pixie dust helps the children to fly'")]},
    config=config_rejectid
)

if "__interrupt__" in result_reject:
    print("⏸️ Action Intercepted! Human operator is rejecting...")
    result_reject = agent_reject.invoke(
        Command(resume={"decisions": [{"type": "reject"}]}),
        config=config_rejectid
    )
    print(f"❌ Terminated Result: {result_reject['messages'][-1].content}")



# ------------------------------------------------------------------------------
# 3. EDITING DEMONSTRATION FLOW
# ------------------------------------------------------------------------------
print("\n=== Scenario 3: Editing Parameters Inline ===")

agent_edit = create_agent(
    model=GPT_MODEL_AGENT,
    tools=[read_email_tool, send_email_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email_tool": {"allowed_decisions": ["approve", "edit", "reject"]},
                "read_email_tool": False,
            }
        ),
    ],
)

config_editid = {"configurable": {"thread_id": "test-edit"}}
result_edit = agent_edit.invoke(
    {"messages": [HumanMessage(content="Send email to wrongmissy@idontknow.com with subject 'Test' and body 'Hello'")]},
    config=config_editid
)

if "__interrupt__" in result_edit:
    print("⏸️ Action Intercepted! Human operator is modifying arguments...")
    result_edit = agent_edit.invoke(
        Command(
            resume={
                "decisions": [
                    {
                        "type": "edit",
                        "edited_action": {
                            "name": "send_email_tool",
                            "args": {
                                "recipient": "correctwendy@neverland.com",
                                "subject": "Corrected Subject",
                                "body": "This was edited by a human before sending."
                            }
                        }
                    }
                ]
            }
        ),
        config=config_editid
    )
    print(f"✏️ Adjusted Result: {result_edit['messages'][-1].content}. Edited email sent to Wendy")