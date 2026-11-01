"""
TradeAgent Sentiment Analyzer.

Uses FinBERT to score news headlines as bullish, bearish, or neutral.
"""

import asyncio
from typing import List

try:
    import torch
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from config.settings import trading_params
from core.logger import get_logger
from core.models import SentimentLabel, SentimentResult

logger = get_logger("sentiment_analyzer")


class SentimentAnalyzer:
    def __init__(self):
        self.config = trading_params.sentiment_config
        self.pipeline = None
        self.is_loaded = False

    def load_model(self) -> bool:
        if not HAS_TRANSFORMERS:
            logger.warning("transformers/torch not installed. FinBERT disabled.")
            return False
            
        if not self.config.get("enabled", True):
            return False
            
        try:
            # Load FinBERT
            # device=0 for GPU, -1 for CPU
            device = 0 if torch.cuda.is_available() else -1
            self.pipeline = pipeline(
                "sentiment-analysis", 
                model="ProsusAI/finbert",
                device=device
            )
            self.is_loaded = True
            logger.info(f"FinBERT loaded successfully (GPU: {device == 0}).")
            return True
        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}")
            return False

    async def analyze_text(self, text: str) -> SentimentResult:
        """Analyze a single piece of text."""
        if not self.is_loaded or not self.pipeline:
            return SentimentResult(text=text, label=SentimentLabel.NEUTRAL, confidence=0.0)
            
        try:
            def _analyze():
                return self.pipeline(text)
                
            res = await asyncio.to_thread(_analyze)
            
            if res and len(res) > 0:
                result = res[0]
                label_str = result["label"].lower() # positive, negative, neutral
                score = result["score"]
                
                if label_str == "positive":
                    label = SentimentLabel.BULLISH
                elif label_str == "negative":
                    label = SentimentLabel.BEARISH
                else:
                    label = SentimentLabel.NEUTRAL
                    
                return SentimentResult(
                    text=text,
                    label=label,
                    confidence=score,
                    source="finbert"
                )
                
        except Exception as e:
            logger.error(f"FinBERT analysis error: {e}")
            
        return SentimentResult(text=text, label=SentimentLabel.NEUTRAL, confidence=0.0)

    async def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze a batch of texts."""
        if not texts: return []
        
        # Simple concurrent processing for demo, pipeline handles batching natively 
        # better in a real implementation
        tasks = [self.analyze_text(t) for t in texts]
        return await asyncio.gather(*tasks)

    def aggregate_score(self, results: List[SentimentResult]) -> float:
        """
        Aggregate multiple results into a single score (-1.0 to +1.0).
        Positive = Bullish, Negative = Bearish.
        """
        if not results: return 0.0
        
        score = 0.0
        total_weight = 0.0
        
        min_conf = self.config.get("min_confidence", 0.6)
        
        for res in results:
            if res.confidence < min_conf: continue
                
            weight = res.confidence
            if res.label == SentimentLabel.BULLISH:
                score += weight
            elif res.label == SentimentLabel.BEARISH:
                score -= weight
                
            total_weight += weight
            
        if total_weight > 0:
            return score / total_weight
        return 0.0
