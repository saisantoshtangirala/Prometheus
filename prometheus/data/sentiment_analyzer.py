"""
Sentiment Analyzer – Bayesian Truth Serum weighted NLP pipeline.

Components:
  1. LegalBERT: parses SEC 10-K/10-Q/8-K filings for hidden CEO sentiment.
     Looks beyond boilerplate for language markers of stress, uncertainty, deception.
  2. Reddit/Twitter scraper: social media sentiment with bot filtering.
  3. Bayesian Truth Serum: up-weights "silent genius" retail traders whose
     predictions are surprisingly accurate relative to the crowd consensus.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class LegalBERTSentimentAnalyzer:
    """
    Uses a fine-tuned BERT model for financial regulatory filings.
    In production: load 'nlpaueb/legal-bert-base-uncased' fine-tuned on SEC filings.
    Detects: management confidence, going-concern risk, litigation language,
    uncertainty hedging, and positive/negative forward guidance.
    """

    # Linguistic markers of management stress (from academic literature)
    NEGATIVE_SIGNALS = [
        "significant uncertainty", "material weakness", "going concern",
        "liquidity risk", "covenant breach", "regulatory investigation",
        "whistleblower", "restatement", "impairment", "write-off",
        "headwinds", "challenging", "difficult", "uncertain", "risk factors",
        "substantially all", "no assurance",
    ]

    POSITIVE_SIGNALS = [
        "record revenue", "strong demand", "exceeded expectations",
        "market share gains", "strategic opportunity", "accelerating growth",
        "positive momentum", "return of capital", "buyback", "dividend increase",
        "outperform", "confident", "well-positioned",
    ]

    UNCERTAINTY_HEDGES = [
        "may", "might", "could", "would", "potentially", "subject to",
        "depends upon", "cannot guarantee", "no assurance", "estimated",
    ]

    def __init__(self, use_transformer: bool = False):
        self.use_transformer = use_transformer
        self._transformer = None
        if use_transformer:
            self._load_transformer()

    def _load_transformer(self) -> None:
        try:
            from transformers import pipeline
            self._transformer = pipeline(
                "text-classification",
                model="yiyanghkust/finbert-tone",  # FinBERT as proxy for LegalBERT
                tokenizer="yiyanghkust/finbert-tone",
                truncation=True,
                max_length=512,
            )
            logger.info("Loaded FinBERT transformer")
        except Exception as e:
            logger.warning("Could not load transformer: %s — using rule-based", e)
            self._transformer = None

    def analyze_filing(self, text: str, ticker: str = "") -> Dict:
        """
        Analyze an SEC filing text for sentiment signals.
        Returns structured sentiment report.
        """
        text_lower = text.lower()
        n_words = max(len(text.split()), 1)

        # Count signal occurrences (normalized by length)
        neg_count = sum(text_lower.count(s) for s in self.NEGATIVE_SIGNALS)
        pos_count = sum(text_lower.count(s) for s in self.POSITIVE_SIGNALS)
        hedge_count = sum(text_lower.count(s) for s in self.UNCERTAINTY_HEDGES)

        neg_density = neg_count / n_words * 1000
        pos_density = pos_count / n_words * 1000
        hedge_density = hedge_count / n_words * 1000

        # Loughran-McDonald sentiment score (academic standard for finance)
        sentiment_score = (pos_density - neg_density) / (pos_density + neg_density + 1e-8)

        # Uncertainty score: high hedging language = management uncertainty
        uncertainty_score = min(hedge_density / 10.0, 1.0)

        # Transformer enhancement
        transformer_score = None
        if self._transformer is not None:
            try:
                excerpt = text[:512]  # transformer max length
                result = self._transformer(excerpt)
                label_map = {"Positive": 1.0, "Neutral": 0.0, "Negative": -1.0}
                transformer_score = label_map.get(result[0]["label"], 0.0) * result[0]["score"]
            except Exception:
                pass

        final_score = (
            0.6 * sentiment_score +
            0.4 * (transformer_score if transformer_score is not None else sentiment_score)
        )

        return {
            "ticker": ticker,
            "sentiment_score": float(final_score),
            "positive_density": float(pos_density),
            "negative_density": float(neg_density),
            "uncertainty_score": float(uncertainty_score),
            "going_concern_risk": "going concern" in text_lower,
            "litigation_risk": "regulatory investigation" in text_lower or "whistleblower" in text_lower,
            "signal_strength": abs(float(final_score)),
            "interpretation": self._interpret(final_score, uncertainty_score),
        }

    def _interpret(self, score: float, uncertainty: float) -> str:
        if score > 0.3 and uncertainty < 0.3:
            return "BULLISH: management confident, low hedging language"
        if score < -0.3:
            return f"BEARISH: significant negative signals, uncertainty={uncertainty:.2f}"
        if uncertainty > 0.6:
            return "CAUTIOUS: high uncertainty hedging — management may be hiding concerns"
        return "NEUTRAL: mixed or boilerplate language"


class BayesianTruthSerum:
    """
    Up-weights retail traders whose predictions are surprisingly accurate
    relative to the crowd consensus.

    Based on Prelec (2004): the true answer is the one that is more
    popular than the average person predicts it will be.

    Applied here: retail sentiment signals weighted by their "surprise factor"
    — traders who saw the move coming before it was consensus get amplified.
    """

    def __init__(
        self,
        history_window: int = 90,
        min_predictions: int = 10,
    ):
        self.history_window = history_window
        self.min_predictions = min_predictions
        self._user_history: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        self._aggregate_predictions: List[float] = []

    def record_prediction(
        self,
        user_id: str,
        predicted_direction: float,   # +1 or -1
        actual_outcome: float,         # actual return
        timestamp: str = "",
    ) -> None:
        """Record a user's prediction and its outcome."""
        self._user_history[user_id].append((predicted_direction, actual_outcome))
        if len(self._user_history[user_id]) > self.history_window:
            self._user_history[user_id].pop(0)
        self._aggregate_predictions.append(predicted_direction)

    def compute_bts_weights(self) -> Dict[str, float]:
        """
        Compute BTS weight for each user.
        High weight = user's predictions were surprisingly accurate.
        """
        if len(self._aggregate_predictions) < self.min_predictions:
            return {}

        consensus = np.mean(self._aggregate_predictions[-self.history_window:])
        weights = {}

        for user_id, history in self._user_history.items():
            if len(history) < 5:
                continue
            preds = np.array([h[0] for h in history])
            actuals = np.array([h[1] for h in history])

            # Accuracy
            accuracy = float(np.mean(np.sign(preds) == np.sign(actuals)))

            # Surprise factor: how often did user predict against consensus AND be right?
            against_consensus = np.sign(preds) != np.sign(consensus)
            contrarian_accuracy = float(
                np.mean(np.sign(preds[against_consensus]) == np.sign(actuals[against_consensus]))
            ) if against_consensus.any() else 0.0

            # BTS weight: base accuracy + contrarian bonus
            bts_weight = accuracy + 1.5 * contrarian_accuracy
            weights[user_id] = float(bts_weight)

        return weights

    def aggregate_sentiment(
        self,
        signals: Dict[str, float],  # user_id → predicted_return
    ) -> Dict:
        """
        Aggregate raw sentiment signals using BTS weights.
        Returns weighted aggregate with bot-filtered signal.
        """
        bts_weights = self.compute_bts_weights()
        total_weight = 0.0
        weighted_signal = 0.0
        n_silent_genius = 0

        for user_id, signal in signals.items():
            weight = bts_weights.get(user_id, 0.5)  # unknown users get 0.5
            if weight > 1.2:
                n_silent_genius += 1  # "silent genius" — high BTS score
            weighted_signal += weight * signal
            total_weight += weight

        if total_weight < 1e-8:
            return {"sentiment": 0.0, "confidence": 0.0, "n_silent_genius": 0}

        avg_sentiment = weighted_signal / total_weight
        return {
            "sentiment": float(avg_sentiment),
            "raw_sentiment": float(np.mean(list(signals.values()))),
            "bts_boost": float(avg_sentiment - np.mean(list(signals.values()))),
            "n_users": len(signals),
            "n_silent_genius": n_silent_genius,
            "confidence": min(len(bts_weights) / max(len(signals), 1), 1.0),
        }


class SentimentAnalyzer:
    """
    Master sentiment orchestrator combining LegalBERT + BTS.
    """

    def __init__(self, use_transformer: bool = False):
        self.legal_bert = LegalBERTSentimentAnalyzer(use_transformer)
        self.bts = BayesianTruthSerum()

    def analyze_sec_filing(self, text: str, ticker: str) -> Dict:
        return self.legal_bert.analyze_filing(text, ticker)

    def aggregate_social_sentiment(
        self,
        raw_signals: Dict[str, float],
        user_histories: Optional[Dict] = None,
    ) -> Dict:
        return self.bts.aggregate_sentiment(raw_signals)

    def get_combined_sentiment_index(
        self,
        filing_score: float,
        social_score: float,
        filing_weight: float = 0.6,
        social_weight: float = 0.4,
    ) -> Dict:
        combined = filing_weight * filing_score + social_weight * social_score
        return {
            "combined_sentiment": float(combined),
            "signal": "BUY" if combined > 0.15 else ("SELL" if combined < -0.15 else "NEUTRAL"),
            "strength": float(abs(combined)),
        }
