"""
PyTorch Time-Series Transformer.
Analyzes sequential candlestick data using Multi-Head Attention.
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from core.logger import get_logger

logger = get_logger("transformer_predictor")

class TimeSeriesTransformer(nn.Module):
    def __init__(self, feature_dim=15, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_linear = nn.Linear(feature_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 3) # output: [Bearish, Neutral, Bullish]
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len, feature_dim)
        x = self.input_linear(x)
        out = self.transformer(x)
        
        # take the last time step
        last_step = out[:, -1, :]
        return self.classifier(last_step)


class TransformerPredictor:
    def __init__(self, seq_len: int = 30):
        self.seq_len = seq_len
        self.is_trained = False
        try:
            self.model = TimeSeriesTransformer()
            self.model.eval()
            self.is_ready = True
            logger.info("Time-Series Transformer initialized (Architecture Ready).")
        except Exception as e:
            logger.error(f"Failed to initialize PyTorch Transformer: {e}")
            self.is_ready = False
            
    def predict(self, df: pd.DataFrame) -> tuple[int, float]:
        """
        Returns (direction, confidence)
        direction: 1 (Buy), -1 (Sell), 0 (Neutral)
        """
        if not self.is_ready or df is None or len(df) < self.seq_len:
            return 0, 0.0
            
        try:
            # Extract basic features
            features = ['close']
            if 'volume' in df.columns: features.append('volume')
            for col in ['RSI', 'MACD_HIST', 'ATR', 'ADX']:
                if col in df.columns: features.append(col)
                
            seq = df[features].iloc[-self.seq_len:].values
            # pad to feature_dim = 15
            if seq.shape[1] < 15:
                padding = np.zeros((self.seq_len, 15 - seq.shape[1]))
                seq = np.hstack((seq, padding))
                
            tensor_seq = torch.FloatTensor(seq).unsqueeze(0) # (1, seq_len, 15)
            
            with torch.no_grad():
                logits = self.model(tensor_seq)
                probs = torch.softmax(logits, dim=-1).squeeze()
                
            # For safety, since it's untrained, we return 0 for now to not ruin ensemble
            if not self.is_trained:
                return 0, 0.0
                
            pred_class = torch.argmax(probs).item()
            conf = probs[pred_class].item()
            
            if pred_class == 2: return 1, conf
            elif pred_class == 0: return -1, conf
            return 0, conf
            
        except Exception as e:
            logger.error(f"Transformer prediction error: {e}")
            return 0, 0.0
