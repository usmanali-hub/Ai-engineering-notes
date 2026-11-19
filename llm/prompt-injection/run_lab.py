"""Run prompt injection evaluation across naive vs defended prompts."""

from __future__ import annotations

from injection_lab import (
    ATTACK_CASES,
    build_defended_prompt,
    evaluate_attack,
    mock_model_generate,
)


def run_benchmark():
    print("=" * 70)
    print("PROMPT INJECTION BENCHMARK: NAIVE vs DEFENDED PROMPTS")
    print("=" * 70)

    naive_passed = 0
    defended_passed = 0

    for case in ATTACK_CASES:
        print(
            f"\n[Case {case.case_id}] {case.attack_type.value.upper()}: {case.description}"
        )
        print(f"Payload: {case.payload!r}")

        # 1. Naive run
        naive_prompt = f"System: Process this.\nUser: {case.payload}"
        naive_resp = mock_model_generate(naive_prompt, defend=False)
        naive_ok = evaluate_attack(case, naive_resp)
        if naive_ok:
            naive_passed += 1

        # 2. Defended run
        defended_prompt = build_defended_prompt(case.payload)
        defended_resp = mock_model_generate(defended_prompt, defend=True)
        defended_ok = evaluate_attack(case, defended_resp)
        if defended_ok:
            defended_passed += 1

        print(
            f"  Naive Result:    {'PASSED' if naive_ok else 'COMPROMISED (Defeated)'}"
        )
        print(f"  Defended Result: {'PASSED (Held)' if defended_ok else 'COMPROMISED'}")

    total = len(ATTACK_CASES)
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print(
        f"  Naive Defense Pass Rate:    {naive_passed}/{total} ({naive_passed / total:.1%})"
    )
    print(
        f"  Defended System Pass Rate: {defended_passed}/{total} ({defended_passed / total:.1%})"
    )
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
