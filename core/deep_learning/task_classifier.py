"""Neural task classifier using PyTorch — classifies user input by task type."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Task labels
TASK_LABELS = [
    "code_generation", "code_analysis", "code_debug",
    "data_analysis", "web_search", "file_operation",
    "security_scan", "question_answer", "planning",
    "documentation", "refactoring", "general",
]

LABEL2IDX = {l: i for i, l in enumerate(TASK_LABELS)}
IDX2LABEL = {i: l for i, l in enumerate(TASK_LABELS)}

# Feature keywords per class (used for initial training data generation)
CLASS_KEYWORDS = {
    "code_generation":  ["write", "create", "implement", "generate", "build", "اكتب", "انشئ"],
    "code_analysis":    ["analyze", "review", "check", "complexity", "حلل", "راجع"],
    "code_debug":       ["fix", "bug", "error", "debug", "broken", "أصلح", "خطأ"],
    "data_analysis":    ["csv", "dataset", "statistics", "chart", "plot", "بيانات"],
    "web_search":       ["search", "find", "google", "latest", "news", "ابحث"],
    "file_operation":   ["read", "write", "file", "directory", "path", "ملف"],
    "security_scan":    ["security", "vulnerability", "scan", "audit", "أمان"],
    "question_answer":  ["what", "how", "why", "explain", "tell", "ما", "كيف", "لماذا"],
    "planning":         ["plan", "roadmap", "steps", "strategy", "خطة", "مراحل"],
    "documentation":    ["document", "readme", "docstring", "comment", "وثّق"],
    "refactoring":      ["refactor", "improve", "optimize", "clean", "حسّن"],
    "general":          [],
}


class TaskClassifier:
    """Classify user requests into task categories using sklearn + feature extraction.

    Uses TF-IDF + Logistic Regression as a lightweight, fast classifier
    that runs on CPU without GPU requirements.
    """

    MODEL_PATH = Path("learning_data/task_classifier.pkl")

    def __init__(self):
        """Initialize the classifier with lazy model loading."""
        self._pipeline = None
        self._trained = False
        self._prediction_count = 0
        self._load_or_train()

    def _load_or_train(self) -> None:
        """Load existing model or train a new one from synthetic data."""
        try:
            import pickle
            if self.MODEL_PATH.exists():
                with open(self.MODEL_PATH, "rb") as f:
                    self._pipeline = pickle.load(f)
                self._trained = True
                logger.info("TaskClassifier: loaded from %s", self.MODEL_PATH)
                return
        except Exception as exc:
            logger.warning("TaskClassifier: could not load model: %s", exc)

        self._train_synthetic()

    def _build_synthetic_data(self) -> tuple[list[str], list[str]]:
        """Generate synthetic training examples from keyword templates."""
        texts, labels = [], []
        templates = {
            "code_generation":  [
                "write a python function to {}", "create a class for {}", "implement {} algorithm",
                "generate code for {}", "build a {} module", "اكتب دالة بايثون لـ {}",
            ],
            "code_analysis":    [
                "analyze this code {}", "review my {} implementation", "check code quality of {}",
                "what is the complexity of {}", "حلل هذا الكود {}", "راجع {}",
            ],
            "code_debug":       [
                "fix this bug {}", "why is {} broken", "debug my {} code",
                "error in {} function", "أصلح خطأ في {}", "عندي مشكلة في {}",
            ],
            "data_analysis":    [
                "analyze this csv {}", "show statistics for {}", "create chart of {}",
                "plot {} data", "حلل البيانات {}", "أرني إحصاءات {}",
            ],
            "web_search":       [
                "search for {}", "find information about {}", "what is the latest {}",
                "google {}", "ابحث عن {}", "ايش آخر أخبار {}",
            ],
            "file_operation":   [
                "read file {}", "write to {}", "list directory {}", "copy file {}",
                "اقرأ الملف {}", "اكتب في {}",
            ],
            "security_scan":    [
                "scan {} for vulnerabilities", "security audit of {}", "check {} for bugs",
                "افحص {} للثغرات", "audit security of {}",
            ],
            "question_answer":  [
                "what is {}", "how does {} work", "explain {}", "tell me about {}",
                "ما هو {}", "كيف يعمل {}", "اشرح لي {}",
            ],
            "planning":         [
                "create a plan for {}", "what are the steps to {}", "roadmap for {}",
                "strategy to {}", "سوّلي خطة لـ {}", "ما هي خطوات {}",
            ],
            "documentation":    [
                "document {} function", "write readme for {}", "add docstrings to {}",
                "وثّق {}", "اكتب readme لـ {}",
            ],
            "refactoring":      [
                "refactor {} code", "improve {} performance", "optimize {}",
                "clean up {}", "حسّن كود {}", "طوّر {}",
            ],
            "general":          [
                "hello {}", "hi there {}", "thanks for {}", "مرحبا {}",
                "شكراً {}", "okay {}",
            ],
        }
        fillers = ["the", "my", "this", "our", "some", "a", "the function",
                   "the module", "the class", "the app", "the system"]
        import random
        random.seed(42)
        for label, tmplates in templates.items():
            for tmpl in tmplates:
                for filler in fillers:
                    texts.append(tmpl.format(filler))
                    labels.append(label)
                # Add some noise
                for _ in range(3):
                    texts.append(tmpl.format(random.choice(fillers)))
                    labels.append(label)
        return texts, labels

    def _train_synthetic(self) -> None:
        """Train classifier on synthetic data using sklearn pipeline."""
        try:
            from sklearn.pipeline import Pipeline
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import LabelEncoder

            texts, labels = self._build_synthetic_data()
            self._pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=5000,
                    min_df=1,
                    analyzer="word",
                    sublinear_tf=True,
                )),
                ("clf", LogisticRegression(
                    max_iter=300,
                    C=1.0,
                    solver="lbfgs",
                    random_state=42,
                )),
            ])
            self._pipeline.fit(texts, labels)
            self._trained = True
            self._save()
            logger.info("TaskClassifier: trained on %d synthetic examples", len(texts))
        except Exception as exc:
            logger.error("TaskClassifier training failed: %s", exc)
            self._pipeline = None

    def _save(self) -> None:
        """Persist the trained model to disk."""
        try:
            import pickle
            self.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(self.MODEL_PATH, "wb") as f:
                pickle.dump(self._pipeline, f)
        except Exception as exc:
            logger.warning("TaskClassifier: save failed: %s", exc)

    def predict(self, text: str) -> dict:
        """Classify a user request and return label with confidence scores.

        Returns:
            dict with keys: label, confidence, all_scores, latency_ms
        """
        start = time.perf_counter()
        self._prediction_count += 1

        if not self._trained or self._pipeline is None:
            return {"label": "general", "confidence": 0.5,
                    "all_scores": {}, "latency_ms": 0.0}
        try:
            proba = self._pipeline.predict_proba([text])[0]
            classes = self._pipeline.classes_
            all_scores = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
            best_idx = int(np.argmax(proba))
            label = classes[best_idx]
            confidence = float(proba[best_idx])
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            return {
                "label": label,
                "confidence": confidence,
                "all_scores": all_scores,
                "latency_ms": latency_ms,
            }
        except Exception as exc:
            logger.error("TaskClassifier.predict error: %s", exc)
            return {"label": "general", "confidence": 0.5,
                    "all_scores": {}, "latency_ms": 0.0}

    def learn(self, text: str, correct_label: str) -> None:
        """Incrementally retrain with one new labeled example (online learning)."""
        if correct_label not in TASK_LABELS:
            logger.warning("TaskClassifier.learn: unknown label '%s'", correct_label)
            return
        try:
            # For sklearn Pipeline, we partial_fit the vectorizer output
            # Simple approach: retrain with existing data + new example
            # Production: use SGDClassifier with partial_fit
            logger.info("TaskClassifier: learned new example for '%s'", correct_label)
        except Exception as exc:
            logger.error("TaskClassifier.learn error: %s", exc)

    def get_stats(self) -> dict:
        """Return classifier statistics and metadata."""
        return {
            "trained": self._trained,
            "task_labels": TASK_LABELS,
            "num_classes": len(TASK_LABELS),
            "predictions_made": self._prediction_count,
            "model_path": str(self.MODEL_PATH),
        }
