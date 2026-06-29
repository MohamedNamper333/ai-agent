"""Autoencoder-based anomaly detector — flags suspicious/malicious requests."""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# Known attack patterns for rule-based pre-filter
_INJECTION_PATTERNS = [
    r"(?i)(union\s+select|drop\s+table|insert\s+into|delete\s+from)",  # SQLi
    r"(?i)<script[^>]*>",                                                # XSS
    r"(?i)(\.\.\/){2,}",                                                 # Path traversal
    r"(?i)(exec|eval|system|passthru)\s*\(",                             # Code injection
    r"(?i)(curl|wget|nc\s+-|bash\s+-i)",                                # RCE
    r"(?i)(\x00|\x1a|%00|%1a)",                                          # Null byte
]
_COMPILED = [re.compile(p) for p in _INJECTION_PATTERNS]


class AnomalyDetector:
    """Detect anomalous or malicious user inputs using statistical + rule-based methods.

    Uses sklearn IsolationForest on TF-IDF features for statistical anomaly
    detection, combined with regex-based pattern matching for known attacks.
    """

    MODEL_PATH = Path("learning_data/anomaly_detector.pkl")
    THRESHOLD = -0.3  # anomaly score threshold (lower = more anomalous)

    def __init__(self):
        """Initialize the anomaly detector with lazy model loading."""
        self._model = None
        self._vectorizer = None
        self._trained = False
        self._detection_count = 0
        self._anomaly_count = 0
        self._init()

    def _init(self) -> None:
        """Load or train the anomaly detection model."""
        try:
            import pickle
            if self.MODEL_PATH.exists():
                with open(self.MODEL_PATH, "rb") as f:
                    data = pickle.load(f)
                self._vectorizer = data["vectorizer"]
                self._model = data["model"]
                self._trained = True
                logger.info("AnomalyDetector: model loaded")
                return
        except Exception as exc:
            logger.warning("AnomalyDetector: load failed: %s", exc)
        self._train()

    def _train(self) -> None:
        """Train IsolationForest on synthetic normal requests."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.ensemble import IsolationForest

            normal_samples = [
                "write a python function to sort a list",
                "analyze this csv file and show statistics",
                "what is the capital of France",
                "fix this bug in my code",
                "create a web scraper for news",
                "explain how neural networks work",
                "generate a dockerfile for fastapi",
                "search for latest AI papers",
                "read file data.csv and plot it",
                "refactor this class to use dataclasses",
                "اكتب دالة بايثون لحساب مجموع قائمة",
                "حلل هذا الملف وأعطني إحصاءات",
                "ما هو الذكاء الاصطناعي",
                "أصلح الخطأ في الكود",
                "create unit tests for this module",
            ] * 20

            self._vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
            X = self._vectorizer.fit_transform(normal_samples)

            self._model = IsolationForest(
                n_estimators=200,
                contamination=0.01,  # Only 1% of data is anomalous
                random_state=42,
                n_jobs=-1,
            )
            self._model.fit(X)
            self._trained = True
            self._save()
            logger.info("AnomalyDetector: trained on %d normal samples", len(normal_samples))
        except Exception as exc:
            logger.error("AnomalyDetector training failed: %s", exc)

    def _save(self) -> None:
        """Persist model to disk."""
        try:
            import pickle
            self.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(self.MODEL_PATH, "wb") as f:
                pickle.dump({"vectorizer": self._vectorizer, "model": self._model}, f)
        except Exception as exc:
            logger.warning("AnomalyDetector: save failed: %s", exc)

    def _rule_check(self, text: str) -> Optional[str]:
        """Check text against known attack patterns.

        Returns:
            Pattern name if attack detected, None otherwise.
        """
        for pattern in _COMPILED:
            if pattern.search(text):
                return pattern.pattern[:40]
        return None

    def score(self, text: str) -> dict:
        """Score a request for anomaly. Returns risk level and details.

        Returns:
            dict with: is_anomaly, risk_level, rule_match, ml_score, latency_ms
        """
        start = time.perf_counter()
        self._detection_count += 1

        # Rule-based check first (fast)
        rule_match = self._rule_check(text)
        if rule_match:
            self._anomaly_count += 1
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.warning("AnomalyDetector: rule match '%s' in input", rule_match[:30])
            return {
                "is_anomaly": True,
                "risk_level": "HIGH",
                "rule_match": rule_match,
                "ml_score": -1.0,
                "latency_ms": latency_ms,
            }

        # ML-based check
        ml_score = 0.0
        if self._trained and self._vectorizer and self._model:
            try:
                X = self._vectorizer.transform([text])
                ml_score = float(self._model.score_samples(X)[0])
                prediction = self._model.predict(X)[0]  # -1=anomaly, 1=normal
                # Only flag as anomaly when both score is low AND predict says -1
                is_anomaly = bool(prediction == -1) and (ml_score < self.THRESHOLD)
                if is_anomaly:
                    self._anomaly_count += 1
            except Exception as exc:
                logger.error("AnomalyDetector ML score error: %s", exc)
                is_anomaly = False
        else:
            is_anomaly = False

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return {
            "is_anomaly": is_anomaly,
            "risk_level": "MEDIUM" if is_anomaly else "LOW",
            "rule_match": None,
            "ml_score": round(ml_score, 4),
            "latency_ms": latency_ms,
        }

    def get_stats(self) -> dict:
        """Return detection statistics."""
        rate = self._anomaly_count / max(self._detection_count, 1) * 100
        return {
            "trained": self._trained,
            "total_checked": self._detection_count,
            "anomalies_detected": self._anomaly_count,
            "anomaly_rate_pct": round(rate, 2),
            "threshold": self.THRESHOLD,
        }
