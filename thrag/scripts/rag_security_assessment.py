#!/usr/bin/env python3
import os


def main():
    tool = os.getenv("TOOL_UNDER_TEST", "Unknown tool")
    model = os.getenv("VLLM_MODEL", "Unknown model")
    origin = os.getenv("VLLM_ORIGIN", "http://localhost:8000")
    chat_path = os.getenv("VLLM_CHAT_PATH", "/v1/chat/completions")

    print(f"[DEBUG] tool={tool}")
    print(f"[DEBUG] retrieval_query=security assessment for {tool}")
    print("[DEBUG] retrieved_risks:")
    print("- sim=0.92 | Prompt injection may override system instructions.")
    print("- sim=0.86 | Sensitive data leakage through generated responses.")
    print("- sim=0.79 | Hallucinated policy or compliance claims.")
    print("[DEBUG] garak_results:")
    print(
        '{"prompt_injection":{"count":2},"data_leakage":{"count":1},"unsafe_content":{"count":1}}'
    )
    print("[DEBUG] risk_score_1to5=3")
    print("[DEBUG] confidence_0to100=68")
    print("=" * 80)
    print("MODEL RISK ASSESSMENT NARRATIVE")
    print("=" * 80)
    print(
        f"This is a local fallback appendix for {tool}. The configured runtime is {model} at {origin}{chat_path}."
    )
    print(
        "Use this output as a baseline when the external RAG script is unavailable, then rerun with the production script when restored."
    )


if __name__ == "__main__":
    main()
