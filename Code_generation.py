import asyncio
import aiohttp
import time
import textwrap

# =========================
# CONFIG
# =========================
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "mistral:7b"

# =========================
# ASYNC LLM CALL (DAG NODE)
# =========================
async def call_llm(prompt, session):
    start = time.time()
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 400
        }
    }

    async with session.post(OLLAMA_URL, json=payload, timeout=600) as r:
        data = await r.json()

    end = time.time()

    return {
        "output": data["message"]["content"].strip(),
        "prompt_tokens": data.get("prompt_eval_count", 0),
        "completion_tokens": data.get("eval_count", 0),
        "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
        "time_sec": round(end - start, 3)
    }

# =========================
# PROMPTS
# =========================
def direct_prompt(question, contexts):
    return textwrap.dedent(f"""
    Solve the following problem step by step:
    {question}

    Your code should be a python function with format:
    {contexts}

    Please extend your reasoning process as much as possible;
    the longer the chain of thought, the better.

    Last step, enclose your code within ```python ```
    """)

def decompose_prompt():
    return textwrap.dedent("""
    Decompose the previous reasoning trajectory into a series of
    sub-questions or thoughts.

    Instructions:
    1. Each sub-question should list the indexes it depends on (0-based).
    2. Dependencies must come only from previous sub-questions.
    """)

def contract_prompt(dag, test_cases):
    return textwrap.dedent(f"""
    Generate a simplified intermediate form of the original problem
    based on variable dependency analysis.

    You are given a DAG:
    {dag}

    Original test cases:
    {test_cases}

    Enclose the simplified problem within <question></question>
    """)

def judge_prompt(question, solutions):
    sol_text = "\n".join(
        [f"solution {i}: {s}" for i, s in enumerate(solutions)]
    )
    return textwrap.dedent(f"""
    Here is the original problem:
    {question}

    Here are some reference solutions:
    {sol_text}

    Give the index of the best solution.

    Enclose the answer within <answer></answer>
    """)

# =========================
# DAG EXECUTION
# =========================
async def run_experiment_dag(question, function_signature, test_cases):
    logs = []

    async with aiohttp.ClientSession() as session:

        # ---- Direct ----
        direct_res = await call_llm(direct_prompt(question, function_signature), session)
        print("\nDIRECT\n", direct_res["output"])
        logs.append(("Direct", direct_res))

        # ---- Decompose ----
        decompose_res = await call_llm(
            decompose_prompt() + "\n\n" + direct_res["output"],
            session
        )
        print("\nDECOMPOSE\n", decompose_res["output"])
        logs.append(("Decompose", decompose_res))

        # ---- Contract ----
        contract_res = await call_llm(
            contract_prompt(decompose_res["output"], test_cases),
            session
        )
        print("\nCONTRACT\n", contract_res["output"])
        logs.append(("Contract", contract_res))

        if "<question>" in contract_res["output"]:
            simplified_question = contract_res["output"].split("<question>")[1].split("</question>")[0]
        else:
            simplified_question = question

        # ---- DAG FAN-OUT (parallel AoT paths) ----
        aot_tasks = [
            call_llm(direct_prompt(simplified_question, function_signature), session),
            call_llm(direct_prompt(question, function_signature), session)
        ]

        aot_results = await asyncio.gather(*aot_tasks)

        for i, r in enumerate(aot_results):
            print(f"\nAoT PATH {i}\n", r["output"])

        logs.append(("AoT DAG", {"paths": len(aot_results)}))

        # ---- Judge (fan-in) ----
        judge_res = await call_llm(
            judge_prompt(
                question,
                [direct_res["output"]] + [r["output"] for r in aot_results]
            ),
            session
        )
        print("\nJUDGE\n", judge_res["output"])
        logs.append(("Judge", judge_res))

    # ---- Summary ----
    print("\n==============================")
    print("DAG TOKEN & TIME SUMMARY")
    print("==============================")
    for name, res in logs:
        if isinstance(res, dict) and "total_tokens" in res:
            print(f"{name:10s} | tokens={res['total_tokens']:4d} | time={res['time_sec']}s")
        else:
            print(f"{name:10s} | {res}")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    question = "Write a function that returns the nth Fibonacci number."
    function_signature = "def fib(n: int) -> int:"
    test_cases = "assert fib(0) == 0\nassert fib(1) == 1\nassert fib(5) == 5"

    asyncio.run(run_experiment_dag(question, function_signature, test_cases))
