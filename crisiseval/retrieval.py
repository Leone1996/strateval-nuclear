from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .schema import CorpusChunk, Scenario, Treatment


class TfidfRetriever:
    def __init__(self, chunks: list[CorpusChunk]) -> None:
        if not chunks:
            raise ValueError("Retriever requires at least one corpus chunk")
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(
            [f"{chunk.title}\n{chunk.text}" for chunk in chunks]
        )

    def query(self, text: str, top_k: int = 5, priority_terms: list[str] | None = None) -> list[CorpusChunk]:
        query_vector = self.vectorizer.transform([text])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        if priority_terms:
            normalized_terms = [term.lower() for term in priority_terms if term.strip()]
            for index, chunk in enumerate(self.chunks):
                haystack = f"{chunk.id} {chunk.source} {chunk.title} {chunk.text}".lower()
                scores[index] += sum(0.18 for term in normalized_terms if term in haystack)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        return [self.chunks[index] for index, _ in ranked[:top_k]]


def build_query(scenario: Scenario, actor: str, treatment: Treatment) -> str:
    posture = scenario.actors[actor]
    return " ".join(
        [
            scenario.name,
            scenario.family,
            scenario.region,
            scenario.summary,
            scenario.background,
            scenario.initial_state,
            actor,
            posture.political_objective,
            posture.military_objective,
            posture.military_position,
            posture.doctrine_tendency,
            posture.alliance_constraints,
            posture.termination_preference,
            treatment.name,
            treatment.modifier,
            " ".join(scenario.friction_points),
            " ".join(scenario.actors),
        ]
    )


def priority_terms_for(scenario: Scenario, actor: str) -> list[str]:
    terms = [actor, scenario.region, scenario.family]
    if "India" in scenario.actors or actor == "India":
        terms.extend(["india", "indian", "no-first-use", "massive retaliation"])
    if "Pakistan" in scenario.actors or actor == "Pakistan":
        terms.extend(["pakistan", "pakistani", "full-spectrum deterrence", "threshold"])
    if "China" in scenario.actors or actor == "China":
        terms.extend(["china", "chinese", "restraint", "no-first-use"])
    if "North Korea" in scenario.actors or "Korean" in scenario.region:
        terms.extend(["korea", "korean peninsula", "exercise signaling"])
    if scenario.family == "Ambiguous Dual-Use Escalation":
        terms.extend(["dual-use", "entanglement", "capability-intent", "verification"])
    return terms
