"""Event classification engine combining weighted keyword rules with zero-shot NLI."""

from __future__ import annotations

import re
import time
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from src.config import get_events_config, logger


class EventResult(BaseModel):
    """Output schema for event classification."""

    event_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    matched_keywords: List[str] = Field(default_factory=list)
    method: Literal["rules", "zero_shot", "combined", "fallback"]


class EventClassifier:
    """Classifies financial texts into 12 risk event categories using hybrid rules & zero-shot NLI."""

    _instance: Optional["EventClassifier"] = None
    _zero_shot_pipeline = None

    def __init__(
        self,
        rule_threshold: float = 0.65,
        confidence_floor: float = 0.35,
        use_zero_shot: bool = True,
    ) -> None:
        import os

        self.events_cfg = get_events_config()
        self.rule_threshold = rule_threshold
        self.confidence_floor = confidence_floor
        # Disable zero-shot if offline mode requested or explicitly disabled
        is_offline = os.getenv("RISK_MODE", "").upper() == "OFFLINE"
        self.use_zero_shot = use_zero_shot and not is_offline
        self._compiled_patterns: Dict[str, List[Tuple[re.Pattern, float, str]]] = {}
        self._descriptions: Dict[str, str] = {}
        self._build_matchers()

    def _build_matchers(self) -> None:
        """Pre-compiles word-boundary regexes for all event patterns."""
        for label, item in self.events_cfg.events.items():
            self._descriptions[label] = item.description
            compiled = []
            for kw, weight in item.patterns.items():
                pattern_str = rf"\b{re.escape(kw)}\b"
                regex = re.compile(pattern_str, re.IGNORECASE)
                compiled.append((regex, float(weight), kw))
            self._compiled_patterns[label] = compiled

    def _get_zero_shot_pipeline(self):
        """Lazy-loads valhalla/distilbart-mnli-12-1 zero-shot pipeline once (singleton)."""
        if not self.use_zero_shot:
            return None

        if EventClassifier._zero_shot_pipeline is None:
            start_t = time.perf_counter()
            logger.info("Loading Zero-Shot model ('valhalla/distilbart-mnli-12-1')...")
            try:
                import torch
                from transformers import pipeline

                device = 0 if torch.cuda.is_available() else -1
                pipe = pipeline(
                    task="zero-shot-classification",
                    model="valhalla/distilbart-mnli-12-1",
                    device=device,
                )
                EventClassifier._zero_shot_pipeline = pipe
                elapsed = time.perf_counter() - start_t
                logger.info(f"Zero-shot model loaded in {elapsed:.2f}s on device: {device}.")
            except Exception as exc:
                logger.warning(f"Zero-shot model failed to load ({exc}). Operating in rules-only mode.")
                return None
        return EventClassifier._zero_shot_pipeline

    def rule_scores(self, text: str) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
        """Calculates normalized rule scores [0, 1] and captures matched keywords for each category."""
        scores: Dict[str, float] = {}
        matched_kw: Dict[str, List[str]] = {}

        for label, patterns in self._compiled_patterns.items():
            raw_sum = 0.0
            hits: List[str] = []
            for regex, weight, kw_str in patterns:
                if regex.search(text):
                    raw_sum += weight
                    hits.append(kw_str)

            # Normalization: single 3.0 trigger -> 0.75, two triggers >= 4.0 -> 1.0
            normalized = min(1.0, raw_sum / 4.0)
            scores[label] = round(normalized, 4)
            matched_kw[label] = hits

        return scores, matched_kw

    def _run_zero_shot(self, text: str) -> Dict[str, float]:
        """Runs zero-shot NLI using natural language category descriptions."""
        pipe = self._get_zero_shot_pipeline()
        if pipe is None:
            return {}

        candidate_descriptions = list(self._descriptions.values())
        desc_to_label = {desc: label for label, desc in self._descriptions.items()}

        try:
            output = pipe(
                text,
                candidate_labels=candidate_descriptions,
                hypothesis_template="This text is about {}.",
                multi_label=False,
            )
            # Map description probabilities back to event labels
            scores: Dict[str, float] = {}
            for desc, prob in zip(output["labels"], output["scores"]):
                label = desc_to_label.get(desc, "Other")
                scores[label] = float(prob)
            return scores
        except Exception as exc:
            logger.warning(f"Zero-shot inference failed: {exc}")
            return {}

    def classify(self, text: str) -> EventResult:
        """Classifies a text into one of the 12 taxonomy categories."""
        if not text or not text.strip():
            return EventResult(
                event_type="Other",
                confidence=0.10,
                matched_keywords=[],
                method="rules",
            )

        rule_scores_dict, matched_keywords_dict = self.rule_scores(text)

        # Determine best rule match
        best_rule_label = max(rule_scores_dict, key=lambda k: rule_scores_dict[k])
        best_rule_score = rule_scores_dict[best_rule_label]

        # If rule score is decisive (>= rule_threshold), bypass zero-shot NLI
        if best_rule_score >= self.rule_threshold:
            final_type = best_rule_label
            confidence = best_rule_score
            method: Literal["rules", "zero_shot", "combined", "fallback"] = "rules"
            keywords = matched_keywords_dict.get(final_type, [])
        else:
            # Ambiguous or weak rule match -> run zero-shot
            zero_shot_scores = self._run_zero_shot(text)

            if zero_shot_scores:
                # Combine 50% rules + 50% zero-shot
                combined_scores: Dict[str, float] = {}
                for label in self.events_cfg.events:
                    r_score = rule_scores_dict.get(label, 0.0)
                    zs_score = zero_shot_scores.get(label, 0.0)
                    combined_scores[label] = 0.5 * r_score + 0.5 * zs_score

                final_type = max(combined_scores, key=lambda k: combined_scores[k])
                confidence = combined_scores[final_type]
                method = "combined"
                keywords = matched_keywords_dict.get(final_type, [])
            else:
                # Zero-shot unavailable -> use rules
                final_type = best_rule_label
                confidence = best_rule_score
                method = "fallback"
                keywords = matched_keywords_dict.get(final_type, [])

        # Apply confidence floor: if confidence < 0.35, classify as 'Other'
        if confidence < self.confidence_floor:
            final_type = "Other"
            keywords = matched_keywords_dict.get("Other", [])

        return EventResult(
            event_type=final_type,
            confidence=round(confidence, 4),
            matched_keywords=keywords,
            method=method,
        )

    def classify_batch(self, texts: List[str]) -> List[EventResult]:
        """Classifies a batch of texts."""
        return [self.classify(t) for t in texts]


# Module singleton
_CLASSIFIER: Optional[EventClassifier] = None


def get_event_classifier(use_zero_shot: bool = True) -> EventClassifier:
    """Returns singleton EventClassifier instance."""
    global _CLASSIFIER
    if _CLASSIFIER is None:
        _CLASSIFIER = EventClassifier(use_zero_shot=use_zero_shot)
    return _CLASSIFIER
