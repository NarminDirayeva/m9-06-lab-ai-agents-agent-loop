import ollama
import json
from datetime import date


with open("orders.json") as f:
    ORDERS = json.load(f)

def lookup_order(order_id: str) -> str:
    order = ORDERS.get(order_id)
    if not order:
        return f"Order {order_id} not found."

    purchased    = date.fromisoformat(order["purchased"])
    today        = date.today()
    months_passed = (today.year - purchased.year) * 12 + (today.month - purchased.month)
    under_warranty = months_passed < order["warranty_months"]

    return (
        f"Order {order_id}: {order['item']}, "
        f"price=${order['price']}, "
        f"purchased={order['purchased']}, "
        f"warranty={order['warranty_months']} months, "
        f"months_passed={months_passed}, "
        f"still_under_warranty={under_warranty}"
    )


def calculate(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}})
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {e}"


TOOLS = {
    "lookup_order": lookup_order,
    "calculate":    calculate,
}


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": (
                "Look up an order by its ID. "
                "Returns item name, price, purchase date, warranty months, "
                "months passed since purchase, and whether it is still under warranty. "
                "If the order does not exist, returns 'not found'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID, e.g. A1001"
                    }
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a simple arithmetic expression like '2 * 1200'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A valid Python math expression"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]


SYSTEM_INSTRUCTION = """
You are a helpful customer orders assistant.

You have two tools:
- lookup_order(order_id): ALWAYS call this first when a user mentions an order ID.
  It returns item, price, purchase date, warranty period, and warranty status.
- calculate(expression): Use this for any arithmetic, e.g. '2 * 1200'.

Rules:
1. Always call lookup_order before answering any order question.
2. If the order is not found, tell the user clearly. Do NOT invent information.
3. For multi-step questions, use both tools as needed and explain your reasoning step by step.
4. Always include warranty status in your answer when relevant.
"""

# ─────────────────────────────────────────
# 5. Agent loop (əl ilə yazılmış)
# ─────────────────────────────────────────
def run_agent(goal: str, messages: list, max_steps: int = 6):
    print("\n" + "=" * 60)
    print(f"GOAL: {goal}")
    print("=" * 60)

    messages.append({"role": "user", "content": goal})

    for step in range(1, max_steps + 1):
        print(f"\n[Step {step}] Model is being called...")

        response = ollama.chat(
            model="llama3.1",
            messages=messages,
            tools=TOOL_DEFINITIONS,
        )

        msg = response["message"]
        messages.append(msg)

        if msg.get("tool_calls"):
            for tool_call in msg["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                tool_args = tool_call["function"]["arguments"]

                print(f"\n[TOOL CALL]   {tool_name}({tool_args})")

                result = TOOLS[tool_name](**tool_args)

                print(f"[TOOL RESULT] {result}")

                messages.append({
                    "role": "tool",
                    "content": result,
                })

        else:
            final = msg.get("content", "")
            print(f"\n[FINAL ANSWER]\n{final}")
            return final

    print("\n[STOP] Step limiti aşıldı — couldn't finish in time.")
    return "couldn't finish in time"


if __name__ == "__main__":

    messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]

    run_agent(
        goal=(
            "I'm thinking of buying two more of order A1001. "
            "What would those two cost, and is the original still under warranty?"
        ),
        messages=messages,
    )

    run_agent(
        goal="Can you look up order A9999 for me?",
        messages=messages,
    )