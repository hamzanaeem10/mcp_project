# client.py
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
import langchain
from dotenv import load_dotenv

langchain.debug = True
load_dotenv()

# ✅ Global client & agent cache so we don’t reinit every call
client = MultiServerMCPClient(
    {
        "shystem": {
            "command": "python",
            "args": ["server.py"],
            "transport": "stdio"
        }
    }
)
agent = None   # will hold the initialized agent

# ✅ Keywords for solar/electricity-related queries
SOLAR_KEYWORDS = ["solar panel", "electricity bill", "batteries", "battery", "power", "kw", "kilowatt","bill"]

def preprocess_prompt(user_prompt: str) -> str:
    """Prepend JSON instruction if user query is about solar/electricity."""
    if any(keyword.lower() in user_prompt.lower() for keyword in SOLAR_KEYWORDS):
        return "Read the appliances.json, products.json and tarrifs.json.\n\n" + user_prompt
    return user_prompt


async def init_agent():
    """Initialize agent once and reuse it with memory."""
    global agent
    if agent is not None:
        return agent

    # Load MCP tools + resources
    tools = await client.get_tools()                     # ✅ FIXED (no args here)
    resources = await client.get_resources("shystem")    # ✅ keep server_name here
    print("DEBUG: Tools available ->", [t.name for t in tools])

    # Define system rules
    system_message = """
    You are an AI assistant connected to a tool-augmented MCP server.

You have access to the following tools:

1. brave_search(query, max_results)
   • Retrieve real-time web search results using the Tavily API.
   • Always use for factual or up-to-date information not in general knowledge.

2. calculator(op, a, b, expression)
   • Perform basic arithmetic (+, -, ×, ÷) or evaluate a math expression.
   • Always use this tool for numeric calculations; never compute directly yourself.

3. calc(expr)
   • Safely evaluate algebraic/numeric expressions (supports Fractions, sqrt, etc.).
   • Use when an expression is more complex than calculator supports.

4. d_s_t(distance, speed, time, units)
   • Solve distance = speed × time when exactly two values are given.
   • Only for single uniform motion segments. Do not retry repeatedly with same inputs.
   • For multi-step problems with stops or changing speeds, compute each segment separately.

5. percent_of(percent, base)
   • Compute percentages of values.

6. nCr(n, r)
   • Compute combinations C(n, r).

7. average(values)
   • Compute the arithmetic mean of a list.

8. ratio_split(total, a, b)
   • Split a total into two parts given ratio a:b.
   • Use for sharing/dividing problems.

9. solve_linear(equation, var)
   • Solve a single-variable linear equation (e.g., "2x+3=11").
   • Returns solution, no-solution, or infinite-solution.

10. format_answer(value, style, decimals)
    • Format final numeric answers consistently: int, float, money, percent, or fraction.

11. time_math(op, time, delta, hours, minutes, seconds, wrap_24h)
    • Add or subtract durations from times.
    • Supports wall-clock "HH:MM[:SS]" and durations like "1h 30m", "-45m", "0:15:30".
    • Returns normalized result time and days_offset for cross-midnight cases.

12. time_diff(start, end, cross_midnight)
    • Compute duration between two times, optionally across midnight.

13. read_document(filename)
    • Read and return text of a PDF in the docs/ folder.
    • Must be used for all PDF-based Q&A.

14. documents()
    • List all available PDF documents in docs/.

15. read_json(filename)
    • Read structured data from a JSON file in docs/.
    • Always use for answering JSON-based data questions.

16. list_json_files()
    • List all available JSON files in docs/.

---

### Rules for using tools

1. **Math / Arithmetic**
   - Always use calculator, calc, d_s_t, nCr, percent_of, ratio_split, average, solve_linear, or time_math/time_diff for math questions.
   - Do NOT perform calculations by reasoning in your own text.
   - Only use for a single uniform motion calculation. Do not call this repeatedly for multi-segment word problems. Do algebra in text instead

2. **Web / Knowledge**
   - For real-time, factual, or current data (e.g., news, events, product info), always call brave_search.
   - Do not fabricate answers that depend on the current world.

3. **Documents**
   - For PDFs: first list with documents(), then call read_document(filename).
   - Do not answer from your own memory when a document is required.
   - For JSON: use list_json_files() and read_json(filename) to access data.

4. **Answer Formatting**
   - Use format_answer for presenting final numeric results in user-friendly style.

5. **Avoid Loops**
   - Do not call d_s_t multiple times with the same arguments.
   - If a tool returns `do_not_retry: true`, stop calling it again.
   - Prefer higher-level tools (solve_linear, ratio_split, time_math) if available.

6. **General**
   - Combine tools as needed (e.g., brave_search → calculator).
   - Always return final answers clearly after tool usage.
   - Never skip tool usage when required by these rules.

---


    """

    # LLM (OpenAI compatible model)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1).with_config(
        {"system_message": system_message}
    )

    # ✅ Add memory
    memory = MemorySaver()
    agent = create_react_agent(llm, tools, checkpointer=memory)
    return agent


async def run_agent(prompt: str, thread_id: str = "chat-1"):
    """Run the agent with preserved memory per thread_id."""
    agent = await init_agent()

    # ✅ Preprocess user prompt before sending
    final_prompt = preprocess_prompt(prompt)

    response = await agent.ainvoke(
    {"messages": [{"role": "user", "content": final_prompt}]},
    config={
        "recursion_limit": 100,  # ⬅️ bump from default 25
        "configurable": {"thread_id": thread_id}
    }
)
    return response["messages"][-1].content


if __name__ == "__main__":
    async def main():
        # Example normal query (unchanged)
        res1 = await run_agent(
            "Load the PDF 'uploads/09. Build & Evaluate a Tool-Calling AI Agent.pdf' and tell me the main objectives.",
            thread_id="demo-thread"
        )
        print("Agent:", res1)

        # ✅ Example solar/electricity query (prepends JSON instruction)
        res2 = await run_agent(
            "How many kW of solar panels do I need for a monthly electricity bill of 5000 PKR?",
            thread_id="demo-thread"
        )
        print("Agent:", res2)

    asyncio.run(main())
