# TradeAgent: Project Summary

TradeAgent is a fully autonomous, multi-market AI trading agent designed to execute trades across Forex, Crypto, and Equities. It supports seamless integration with various brokers including MetaTrader 5 (MT5), OANDA, and CCXT, as well as a safe Paper Trading mode. It acts as a comprehensive trading ecosystem that combines classical Technical Analysis (TA), Machine Learning (ML), Sentiment NLP, and strict Institutional-grade Risk Management.

## Multi-Timeframe Architecture

The core orchestrator (`core/agent.py`) operates on a multi-timeframe analysis loop.
- **Tick Stream & Heartbeat**: The agent continually ingests tick streams and historical candles.
- **Timeframe Processing**: During each analysis cycle, the agent fetches historical data for all active timeframes defined in the configuration (e.g., 1m, 5m, 15m, 1h, 4h).
- **Strategy Alignment**: Different strategies are mapped to specific time horizons. For example, Mean Reversion strategies are strictly executed on short-term timeframes (1m-30m), whereas Trend Following and Volatility Breakout strategies are restricted to long-term timeframes (1h+).
- **Walk-Forward Optimization**: The agent runs an autonomous background process (`_optimization_loop`) every 7 days. This invokes the `StrategyDiscovererAgent` to backtest and refine parameters dynamically, ensuring the agent adapts to evolving market conditions.

## ML and TA Agents Collaboration

TradeAgent uses a hybrid approach, merging dynamic technical analysis with machine learning:
1. **Dynamic TA Signals**: `TASignalGenerator` evaluates market data against three distinct strategies (Trend, Mean Reversion, Breakout). Instead of static indicator parameters, it dynamically injects optimized parameters (the "Brain") produced by the `StrategyDiscovererAgent` (via Optuna and vectorbt).
2. **Context Agent**: The `ContextAgent` identifies the overarching market regime (e.g., Strong Bullish, Ranging). If a generated TA signal opposes the prevailing macroeconomic or technical trend, the Context Agent penalizes the signal's confidence and position size (e.g., 50% size reduction for counter-trend trades).
3. **ML Ensemble Veto**: An ML `SignalEnsemble` (utilizing LightGBM and Reinforcement Learning models) acts as a secondary layer of validation. It evaluates the TA signal's direction and confidence. The ensemble can override the TA direction if it strongly disagrees, or it can veto the signal entirely (resulting in a hold).

## Risk Management Systems

The `RiskManager` acts as the final gatekeeper before any order execution, implementing institutional safeguards:
- **Fractional Kelly Sizing**: Position sizes are calculated using the Fractional Kelly Criterion (`W - ((1 - W) / R)`). The calculated fraction is capped (max 0.5 multiplier) to ensure safety. 
- **Volatility Regime Filter**: If the Average True Range (ATR) indicates a chaotic market (e.g., `ATR / Close > 0.4%`), the Kelly fraction is slashed by 50%.
- **ATR Dynamic Stops (SL/TP)**: The `StopLossManager` calculates Stop Loss and Take Profit distances dynamically using the ATR. The ATR multiplier is scaled based on the timeframe (e.g., 0.75x for short-term, 1.5x for long-term). Dynamic trailing stops are also supported.
- **Drawdown Circuit Breakers**: The agent tracks equity high-water marks and applies strict limits: Maximum Daily Drawdown (e.g., 3%), Weekly Drawdown (5%), and an All-Time Rolling Peak Drawdown (10%). Breaching any limit immediately pauses trading.
- **Correlation Guards**: The system analyzes open positions' base and quote currencies to prevent over-exposure to a single currency (e.g., limiting concurrent EUR-long positions).
- **Fundamental Kill-Switch**: Integration with economic calendars forces the liquidation of positions and pauses trading during high-impact news events.

## How to Run the Bot

1. **Environment Setup**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # (or `venv\Scripts\activate` on Windows)
   pip install -r requirements.txt
   ```
2. **Configuration**:
   Copy `config/.env.example` to `config/.env` and add any necessary API keys (OANDA, MT5, Finnhub, OpenAI). Adjust `config/trading_params.yaml` to set your desired timeframes, risk limits, and asset watchlists.
3. **Execution**:
   Run the bot in paper trading mode to simulate trades safely:
   ```bash
   python main.py --mode paper
   ```
   To run in live execution mode:
   ```bash
   python main.py --mode live
   ```
4. **Remote Control**: The bot supports a Telegram UI, allowing you to remotely monitor status, pause (`/pause`), resume (`/resume`), or trigger a complete liquidation kill-switch (`/kill`).
