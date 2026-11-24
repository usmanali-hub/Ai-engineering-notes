"""Run the LLM-as-judge pattern end to end with two scripted "judges" standing in for a real model: a
word-overlap heuristic scoring four candidate answers pointwise, and a deliberately first-slot-biased judge used to
show what `check_position_bias` is for and what it looks like when it fires.
"""

from __future__ import annotations

import json

from judge import build_pointwise_prompt, check_position_bias, judge_pointwise

QUESTION = "What does normalizing a vector to unit length do to cosine similarity?"
REFERENCE = "It turns cosine similarity into a plain dot product, since the magnitude terms become 1."

CANDIDATES = {
    "correct_and_concise": "It reduces cosine similarity to a dot product, because both vectors now have magnitude 1.",
    "correct_but_verbose": (
        "Great question! When you normalize a vector, you scale it so its length (its Euclidean norm) becomes "
        "exactly 1. Once both vectors you're comparing have unit length, the formula for cosine similarity — which "
        "normally divides the dot product by the product of the two magnitudes — simplifies, because those "
        "magnitudes are both just 1, so the whole expression collapses down to being a plain dot product."
    ),
    "wrong": "It makes cosine similarity equal to the Euclidean distance between the two vectors.",
    "off_topic": "Cosine similarity is used in recommendation systems to compare user preference vectors.",
}


def word_overlap_judge(prompt: str) -> str:
    """A scripted stand-in for a real judge model: scores purely by how many of the reference's content words
    appear in the candidate, with no 'understanding' of whether the surrounding sentence is right. Good enough to
    separate 'wrong' and 'off_topic' from the two correct answers, but blind to the differences a real judge would
    catch (such as 'distance' being outside the reference despite overlapping on other words)."""
    reference_line = next(
        line for line in prompt.splitlines() if line.startswith("Reference answer:")
    )
    candidate_line = next(
        line for line in prompt.splitlines() if line.startswith("Candidate answer:")
    )
    reference_words = set(reference_line.lower().split())
    candidate_words = set(candidate_line.lower().split())
    overlap = len(reference_words & candidate_words) / len(reference_words)
    score = max(1, min(5, round(1 + overlap * 4)))
    return json.dumps(
        {
            "score": score,
            "rationale": f"{overlap:.0%} of the reference's words appear in the candidate",
        }
    )


def biased_pairwise_judge(prompt: str) -> str:
    """A scripted judge that always prefers whichever answer is in slot A — a toy exaggeration of a real,
    documented LLM-as-judge failure mode (a judge model's measurable tendency to favor the first answer it reads,
    regardless of content), built deliberately broken so `check_position_bias` has something real to catch."""
    return json.dumps(
        {"winner": "A", "rationale": "answer A addresses the question directly"}
    )


if __name__ == "__main__":
    print("Pointwise scoring (word-overlap scripted judge):\n")
    for name, candidate in CANDIDATES.items():
        verdict = judge_pointwise(word_overlap_judge, QUESTION, REFERENCE, candidate)
        print(f"  {name}: score={verdict.score}  ({verdict.rationale})")

    print("\nPrompt sent to the judge for one case, for reference:\n")
    print(
        build_pointwise_prompt(QUESTION, REFERENCE, CANDIDATES["correct_and_concise"])
    )

    print(
        "\nPosition-bias check (deliberately broken judge, to show what a real one can slip into):"
    )
    consistent = check_position_bias(
        biased_pairwise_judge,
        QUESTION,
        CANDIDATES["correct_and_concise"],
        CANDIDATES["wrong"],
    )
    print(
        f"  consistent across slot order: {consistent}  (False means position, not content, decided the winner)"
    )
