"""
TaskMate - An AI Agent built with Groq's free API (Llama 3.3)

This agent can reason about user requests and decide which tool to call:
- calculate: solve math expressions
- get_current_time: return current date & time
- get_weather: fetch live weather for any city (no API key needed)
- add_todo / list_todos / remove_todo: manage an in-memory to-do list

Run:
    pip install -r requirements.txt
    export GROQ_API_KEY="your_free_key_from_console.groq.com"
    python agent.py
"""

import os
import json
import ast
import operator
import requests
from datetime import datetime
from groq import Groq

# ---------- In-memory state ----------
TODOS = []

# ---------- Tool implementations ----------

def calculate(expression: str) -> str:
    """Safely evaluate a basic math expression without using eval()."""
    allowed_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            return allowed_ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            return allowed_ops[type(node.op)](_eval(node.operand))
        raise ValueError("Unsupported expression")

    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval(tree.body)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


def get_current_time() -> str:
    return datetime.now().strftime("%A, %d %B %Y, %I:%M %p")


def get_weather(city: str) -> str:
    """Fetch live weather using Open-Meteo (free, no API key required)."""
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=10,
        ).json()

        if "results" not in geo or not geo["results"]:
            return f"Could not find location: {city}"

        lat = geo["results"][0]["latitude"]
        lon = geo["results"][0]["longitude"]
        resolved_name = geo["results"][0]["name"]

        weather = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current_weather": True},
            timeout=10,
        ).json()

        cw = weather.get("current_weather", {})
        return (
            f"Weather in {resolved_name}: {cw.get('temperature')}°C, "
            f"windspeed {cw.get('windspeed')} km/h"
        )
    except Exception as e:
        return f"Error fetching weather: {e}"


def add_todo(task: str) -> str:
    TODOS.append(task)
    return f"Added to your to-do list: '{task}'"


def list_todos() -> str:
    if not TODOS:
        return "Your to-do list is empty."
    return "\n".join(f"{i+1}. {t}" for i, t in enumerate(TODOS))


def remove_todo(task: str) -> str:
    for t in TODOS:
        if t.lower() == task.lower():
            TODOS.remove(t)
            return f"Removed '{task}' from your to-do list."
    return f"Could not find '{task}' in your to-do list."


# ---------- Tool schema (told to the LLM) ----------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a basic math expression, e.g. '12*7+3'",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression to evaluate"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current live weather for a given city name",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_todo",
            "description": "Add a task to the user's to-do list",
            "parameters": {
                "type": "object",
                "properties": {"task": {"type": "string", "description": "Task description"}},
                "required": ["task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_todos",
            "description": "List all current tasks in the to-do list",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_todo",
            "description": "Remove a task from the to-do list by its text",
            "parameters": {
                "type": "object",
                "properties": {"task": {"type": "string", "description": "Task text to remove"}},
                "required": ["task"],
            },
        },
    },
]

FUNCTION_MAP = {
    "calculate": lambda args: calculate(args["expression"]),
    "get_current_time": lambda args: get_current_time(),
    "get_weather": lambda args: get_weather(args["city"]),
    "add_todo": lambda args: add_todo(args["task"]),
    "list_todos": lambda args: list_todos(),
    "remove_todo": lambda args: remove_todo(args["task"]),
}


# ---------- Agent loop ----------
def run_agent():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: Please set the GROQ_API_KEY environment variable.")
        print("Get a free key at: https://console.groq.com/keys")
        return

    client = Groq(api_key=api_key)

    messages = [
        {
            "role": "system",
            "content": (
                "You are TaskMate, a helpful personal assistant agent. "
                "Use the available tools whenever a user's request needs "
                "a calculation, the time, weather, or to-do list management. "
                "Otherwise, answer directly and concisely."
            ),
        }
    ]

    print("TaskMate Agent is ready! Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if tool_calls:
            messages.append(response_message)
            for tool_call in tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)
                result = FUNCTION_MAP[fn_name](fn_args)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": fn_name,
                        "content": result,
                    }
                )

            second_response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=messages,
            )
            final_reply = second_response.choices[0].message.content
            print(f"TaskMate: {final_reply}\n")
            messages.append({"role": "assistant", "content": final_reply})
        else:
            print(f"TaskMate: {response_message.content}\n")
            messages.append({"role": "assistant", "content": response_message.content})


if __name__ == "__main__":
    run_agent()
