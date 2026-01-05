import asyncio
import aiohttp
import textwrap
import time

# =========================
# CONFIG
# =========================
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "mistral:7b"

# =========================
# ASYNC DAG NODE
# =========================
async def call_llm(prompt, session):
    start = time.time()

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 128,
            "top_p": 0.9
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
def direct_prompt(question):
    return textwrap.dedent(f"""
    You are a precise math question solver. Solve the given math
    question step by step using a standard algebraic approach:

    QUESTION:
    {question}

    Enclose the final answer within <answer></answer> tags.
    """)

def decompose_prompt():
    return textwrap.dedent("""
    Decompose the previous reasoning trajectory into a series of sub-questions or thoughts.

    Instructions:
    1. Each sub-question should list the indexes of sub-questions it depends on (0-based).
    2. Dependencies must come only from previous sub-questions.
    """)

def contract_prompt():
    return textwrap.dedent("""
    Generate a simplified intermediate form of the original
    question based on the previous sub-questions or thoughts.

    Enclose the simplified question within <question></question>.
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

    Ensemble the best answer to the original problem.

    Enclose the final answer within <answer></answer> tags.
    """)

# =========================
# DAG EXPERIMENT
# =========================
async def run_experiment_dag(question):
    logs = []

    async with aiohttp.ClientSession() as session:

        # ---- DIRECT ----
        direct_res = await call_llm(direct_prompt(question), session)
        print("\nDIRECT\n", direct_res["output"])
        logs.append(("Direct", direct_res))

        # ---- DECOMPOSE ----
        decompose_res = await call_llm(
            decompose_prompt() + "\n\n" + direct_res["output"],
            session
        )
        print("\nDECOMPOSE\n", decompose_res["output"])
        logs.append(("Decompose", decompose_res))

        # ---- CONTRACT ----
        contract_res = await call_llm(
            contract_prompt() + "\n\n" + decompose_res["output"],
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
            call_llm(direct_prompt(simplified_question), session),
            call_llm(direct_prompt(question), session)
        ]

        aot_results = await asyncio.gather(*aot_tasks)

        for i, r in enumerate(aot_results):
            print(f"\nAoT PATH {i}\n", r["output"])

        logs.append(("AoT DAG", {"paths": len(aot_results)}))

        # ---- JUDGE (fan-in) ----
        judge_res = await call_llm(
            judge_prompt(
                question,
                [direct_res["output"]] + [r["output"] for r in aot_results]
            ),
            session
        )
        print("\nJUDGE\n", judge_res["output"])
        logs.append(("Judge", judge_res))

    # ---- SUMMARY ----
    print("\n==============================")
    print("DAG TOKEN & TIME SUMMARY")
    print("==============================")

    total_tokens = 0
    total_time = 0.0

    for name, res in logs:
        if isinstance(res, dict) and "total_tokens" in res:
            print(f"{name:10s} | tokens={res['total_tokens']:4d} "
                  f"(prompt={res['prompt_tokens']}, completion={res['completion_tokens']}) "
                  f"| time={res['time_sec']}s")
            total_tokens += res["total_tokens"]
            total_time += res["time_sec"]
        else:
            print(f"{name:10s} | {res}")

    print("\nTOTAL TOKENS:", total_tokens)
    print("TOTAL TIME (sec):", round(total_time, 3))

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    question = input("\nEnter a math question:\n").strip()
    asyncio.run(run_experiment_dag(question))
