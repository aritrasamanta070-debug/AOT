import requests
import time
import textwrap
import json

# =========================
# CONFIG
# =========================
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "mistral:7b"   # recommended for stability

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
            "num_predict": 350
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
# APPENDIX A.1.3 PROMPTS
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
# EXPERIMENT PIPELINE
# =========================
def run_experiment(question, contexts):
    logs = []

    print("\n==============================")
    print("DIRECT (Baseline)")
    print("==============================")
    direct_res = call_llm(direct_prompt(question, contexts))
    print(direct_res["output"])
    logs.append(("Direct", direct_res))

    try:
        direct_json = json.loads(direct_res["output"])
        direct_answer = direct_json.get("answer", "")
    except Exception:
        direct_answer = ""

    print("\n==============================")
    print("DECOMPOSE")
    print("==============================")
    decompose_res = call_llm(
        decompose_prompt(question, direct_res["output"], direct_answer)
    )
    print(decompose_res["output"])
    logs.append(("Decompose", decompose_res))

    print("\n==============================")
    print("CONTRACT")
    print("==============================")
    contract_res = call_llm(
        contract_prompt(question, decompose_res["output"], "independent", "dependent")
    )
    print(contract_res["output"])
    logs.append(("Contract", contract_res))

    if "<question>" in contract_res["output"]:
        optimized_question = contract_res["output"].split("<question>")[1].split("</question>")[0]
    else:
        optimized_question = question

    print("\n==============================")
    print("AoT FINAL (Optimized Question)")
    print("==============================")
    aot_res = call_llm(direct_prompt(optimized_question, contexts))
    print(aot_res["output"])
    logs.append(("AoT Final", aot_res))

    print("\n==============================")
    print("JUDGE")
    print("==============================")
    judge_res = call_llm(
        judge_prompt(question, [direct_res["output"], aot_res["output"]])
    )
    print(judge_res["output"])
    logs.append(("Judge", judge_res))

    print("\n==============================")
    print("TOKEN & TIME SUMMARY")
    print("==============================")
    for name, res in logs:
        print(f"{name:10s} | tokens={res['total_tokens']:4d} | time={res['time_sec']}s")

# =========================
# MAIN (USER INPUT)
# =========================
if __name__ == "__main__":
    print("\n=== Atom of Thoughts (AoT) Experiment ===\n")

    question = input("Enter the question:\n> ").strip()
    contexts = read_multiline("\nEnter supporting contexts:")

    if not question or not contexts:
        print("\nError: Question and contexts cannot be empty.")
    else:
        run_experiment(question, contexts)
