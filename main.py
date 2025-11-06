import asyncio
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()

mcp_client = MultiServerMCPClient(
    {
        "n8n-mcp": {
            "command": "npx",
            "args": ["n8n-mcp@2.20.6"],
            "env": {
                "MCP_MODE": "stdio",
                "LOG_LEVEL": "error",
                "DISABLE_CONSOLE_OUTPUT": "true",
                "N8N_MCP_TELEMETRY_DISABLED": "true",
            },
            "transport": "stdio",
        },
    }
)


chat = ChatOpenAI(
    model=os.environ["OPENAI_MODEL"],
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"],
    streaming=False,
    disable_streaming=True,
    temperature=0.0,
    max_tokens=4096,
    timeout=60,
)


async def get_tools() -> list[BaseTool]:
    tools = await mcp_client.get_tools()
    return tools

async def main():
    tools = await get_tools()
    
    agent = create_react_agent(
        model=chat,
        tools=tools,
        prompt="""\
        You are an expert in n8n automation software.
        Do not ask follow-up questions or request clarification.
        You must use provided tools to serve the user's request.
        **Start**: Call `tools_documentation()` for best practices.
        You can also call other tools if needed.
        - list_tasks({category: "category name"}) - Lists available task templates organized by category
        - `get_node_for_task('task_name')` - Get pre-configured node for tasks
        - `search_nodes({query: 'keyword'})` - Search nodes by functionality.
        - `list_nodes({category: 'trigger'})` - Browse nodes by category.""",
    )

    state = {"messages": [HumanMessage(content="Build a n8n workflow JSON that sends an email on manual trigger. Use tools provided to you.")]}
    
    response = await agent.ainvoke(state)

    print("Agent Response:", response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
