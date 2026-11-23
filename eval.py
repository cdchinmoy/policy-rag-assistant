"""Score the pipeline against your test questions.

questions.json format (a list):
[
  {"question": "What is the cancellation policy?",
   "expected_source": "cancellation-policy.md",
   "expected_answer": "24 hours notice required"},
  {"question": "Who won the football match?",
   "expected_source": null,
   "expected_answer": "should say it does not know"}
]

Usage: python eval.py
"""
import json

from rag import ask, retrieve


def main():
    with open("questions.json", encoding="utf-8") as f:
        cases = json.load(f)

    answerable = [c for c in cases if c["expected_source"]]
    hits = 0

    for n, case in enumerate(cases, 1):
        question = case["question"]
        retrieved = {m["source"] for _, m in retrieve(question)}
        if case["expected_source"]:
            found = case["expected_source"] in retrieved
            hits += found
            status = "RETRIEVED" if found else "MISSED"
        else:
            status = "(no source expected)"

        answer, _ = ask(question)
        print(f"\n--- {n}. {question}")
        print(f"Retrieval: {status}")
        print(f"Expected:  {case['expected_answer']}")
        print(f"Answer:    {answer}")

    if answerable:
        print(f"\nRetrieval hit rate: {hits}/{len(answerable)} = {hits / len(answerable):.0%}")
    print("Now read each answer and mark it correct / wrong / ungrounded by hand.")


if __name__ == "__main__":
    main()
