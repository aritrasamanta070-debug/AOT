import requests
import textwrap
import time

# =========================
# CONFIG
# =========================
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "mistral:7b"

# =========================
# TOKEN + TIME AWARE LLM CALL
# =========================
def call_llm(prompt):
    start = time.time()

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 128,   # ↓ reduced
            "top_p": 0.9
        }
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()

    end = time.time()

    return {
        "output": data["message"]["content"].strip(),
        "prompt_tokens": data.get("prompt_eval_count", 0),
        "completion_tokens": data.get("eval_count", 0),
        "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
        "time_sec": round(end - start, 3)
    }


# =========================
# APPENDIX A.1 — MATH PROMPTS
# =========================

def direct_prompt(question):
    return textwrap.dedent(f"""
    You are a precise math question solver . Solve the given math
    question step by step using a standard algebraic approach :

    QUESTION:
    {question}

    Enclose the final answer within <answer></answer> tags.
    """)

def decompose_prompt():
    return textwrap.dedent("""
    Decompose the previous reasoning trajectory into a series of sub-questions or thoughts.

    Instructions:
    1. Each sub-question should list the indexes of sub-questions it depends on (0-based).
    2. Dependencies are defined as information needed in sub -
    question or thought that :
    - Does NOT come directly from the original question
    - MUST come from previous sub - questions or thoughts

    """)

def contract_prompt():
    return textwrap.dedent("""
    Generate a simplified intermediate form of the original
    question based on the previous sub - questions or thoughts step by
    step .
    The previous sub - questions or thoughts with marked
    dependencies actually form a directed acyclic graph ( DAG), where
    nodes whose dependencies is empty list can be regarded as
    independent sub - questions or thoughts .
    The simplified question must be:
    1. self - contained : The simplified question ’s description must
    contain all information needed to solve itself , without requiring
    additional information from the original question or reasoning
    trajectory
    2. test - time reduced : The simplified question must require
    fewer reasoning steps compared to the original question ( these
    steps are reduced because these solved independent sub - problems or
    thoughts become known conditions in the simplified question or
    excluded as incorrect explorations )

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

    Ensemble the best answer to the original problem from the
    solutions step by step :

    Enclose the final answer within <answer></answer> tags.
    """)

# =========================
# EXPERIMENT
# =========================

def run_experiment(question):
    logs = []

    print("\n==============================")
    print("QUESTION")
    print("==============================")
    print(question)

    # ---- DIRECT ----
    print("\n==============================")
    print("DIRECT")
    print("==============================")
    direct_res = call_llm(direct_prompt(question))
    print(direct_res["output"])
    logs.append(("Direct", direct_res))

    # ---- DECOMPOSE ----
    print("\n==============================")
    print("DECOMPOSE")
    print("==============================")
    decompose_res = call_llm(decompose_prompt() + "\n\n" + direct_res["output"])
    print(decompose_res["output"])
    logs.append(("Decompose", decompose_res))

    # ---- CONTRACT ----
    print("\n==============================")
    print("CONTRACT")
    print("==============================")
    contract_res = call_llm(contract_prompt() + "\n\n" + decompose_res["output"])
    print(contract_res["output"])
    logs.append(("Contract", contract_res))

    # Extract simplified question
    if "<question>" in contract_res["output"]:
        simplified_question = contract_res["output"].split("<question>")[1].split("</question>")[0]
    else:
        simplified_question = question

    # ---- AoT FINAL ----
    print("\n==============================")
    print("AoT FINAL")
    print("==============================")
    aot_res = call_llm(direct_prompt(simplified_question))
    print(aot_res["output"])
    logs.append(("AoT Final", aot_res))

    # ---- JUDGE ----
    print("\n==============================")
    print("JUDGE")
    print("==============================")
    judge_res = call_llm(judge_prompt(question, [direct_res["output"], aot_res["output"]]))
    print(judge_res["output"])
    logs.append(("Judge", judge_res))

    # ---- SUMMARY ----
    print("\n==============================")
    print("TOKEN & TIME SUMMARY")
    print("==============================")

    total_tokens = 0
    total_time = 0.0

    for name, res in logs:
        print(f"{name:10s} | tokens={res['total_tokens']:4d} "
              f"(prompt={res['prompt_tokens']}, completion={res['completion_tokens']}) "
              f"| time={res['time_sec']}s")
        total_tokens += res["total_tokens"]
        total_time += res["time_sec"]

    print("\nTOTAL TOKENS:", total_tokens)
    print("TOTAL TIME (sec):", round(total_time, 3))

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    question = input("\nEnter a math question:\n").strip()
    run_experiment(question)
