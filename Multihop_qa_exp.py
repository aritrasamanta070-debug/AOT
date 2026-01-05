import asyncio
import aiohttp
import time
import textwrap
import json

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
            "num_predict": 350
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
# MULTI-LINE USER INPUT
# =========================
def read_multiline(prompt_text):
    print(prompt_text)
    print("(End input with an empty line)")
    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    return " ".join(lines)

# =========================
# PROMPTS
# =========================
def direct_prompt(question, contexts):
    return textwrap.dedent(f"""
    Solve the following multi-hop question step by step:
    {question}

    CONTEXTS:
    {contexts}

    Firstly, extract the relevant supporting sentences from the text,
    then cut out the continuous segments as the answer.

    Provide your response in this JSON format:
    {{
      "question": "{question}",
      "thought": "step-by-step reasoning",
      "supporting_sentences": [
        "all sentences needed to justify the answer"
      ],
      "answer": "precise answer or none"
    }}
    """)

def decompose_prompt(question, trajectory, answer):
    return textwrap.dedent(f"""
    You are tasked with breaking down a multiple choice question
    reasoning process into sub-questions.

    Original Question:
    {question}

    Complete Reasoning Process:
    {trajectory}

    Instructions:
    1. Break down the reasoning into sub-questions
    2. Each sub-question must be interrogative
    3. Each must list dependency indices (0-based)

    Format output as JSON:
    {{
      "thought": "how sub-questions lead to the answer",
      "sub-questions": [
        {{
          "description": "sub-question text",
          "answer": "answer",
          "depend": []
        }}
      ],
      "answer": "{answer}"
    }}
    """)

def contract_prompt(question, decompose_result, independent, dependent):
    return textwrap.dedent(f"""
    You are a multiple choice question solver optimizing reasoning.

    Original question:
    {question}

    Reasoning process:
    {decompose_result}

    Known conditions:
    {independent}

    Dependent reasoning steps:
    {dependent}

    The optimized question must:
    1. Be self-contained
    2. Be more efficient
    3. Retain original options

    Enclose the optimized question within <question></question> tags.
    """)

def judge_prompt(question, solutions):
    sol_text = "\n".join(
        [f"solution {i}: {s}" for i, s in enumerate(solutions)]
    )
    return textwrap.dedent(f"""
    You are a precise multiple choice question solver.

    QUESTION:
    {question}

    SOLUTIONS:
    {sol_text}

    Compare and synthesize the best answer.

    Enclose the final option within <answer></answer> tags.
    """)

# =========================
# DAG EXPERIMENT
# =========================
async def run_experiment_dag(question, contexts):
    logs = []

    async with aiohttp.ClientSession() as session:

        # ---- DIRECT ----
        direct_res = await call_llm(direct_prompt(question, contexts), session)
        print("\nDIRECT\n", direct_res["output"])
        logs.append(("Direct", direct_res))

        try:
            direct_json = json.loads(direct_res["output"])
            direct_answer = direct_json.get("answer", "")
        except Exception:
            direct_answer = ""

        # ---- DECOMPOSE ----
        decompose_res = await call_llm(
            decompose_prompt(question, direct_res["output"], direct_answer),
            session
        )
        print("\nDECOMPOSE\n", decompose_res["output"])
        logs.append(("Decompose", decompose_res))

        # ---- CONTRACT ----
        contract_res = await call_llm(
            contract_prompt(question, decompose_res["output"], "independent", "dependent"),
            session
        )
        print("\nCONTRACT\n", contract_res["output"])
        logs.append(("Contract", contract_res))

        if "<question>" in contract_res["output"]:
            optimized_question = contract_res["output"].split("<question>")[1].split("</question>")[0]
        else:
            optimized_question = question

        # ---- DAG FAN-OUT (parallel AoT paths) ----
        aot_tasks = [
            call_llm(direct_prompt(optimized_question, contexts), session),
            call_llm(direct_prompt(question, contexts), session)
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
    for name, res in logs:
        if isinstance(res, dict) and "total_tokens" in res:
            print(f"{name:10s} | tokens={res['total_tokens']:4d} | time={res['time_sec']}s")
        else:
            print(f"{name:10s} | {res}")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("\n=== Atom of Thoughts (DAG Execution) ===\n")

    question = input("Enter the question:\n> ").strip()
    contexts = read_multiline("\nEnter supporting contexts:")

    if not question or not contexts:
        print("Error: Question and contexts cannot be empty.")
    else:
        asyncio.run(run_experiment_dag(question, contexts))
