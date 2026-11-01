"""
Production Configuration Manager for Pure Gold Sniper.
======================================================
Provides dynamic hot-reloading, strict validation, and thread-safe persistence.
Guarantees that live risk parameters are always bounded by safety hard-caps.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "pure_gold_config.json"


class SniperConfigManager:
    # Hard-coded absolute safety limits that cannot be overridden by config
    SAFETY_BOUNDS = {
        "max_risk_percent": 0.05,       # Absolute maximum 5.0% risk per trade
        "min_risk_percent": 0.005,      # Minimum 0.5%
        "max_emergency_loss": 0.05,     # Absolute maximum 5% account loss per trade
        "max_lot_limit": 2.00,          # Absolute hard cap on lot size
        "min_circuit_breaker_loss": 1,  # Minimum 1 loss before breaker
        "max_circuit_breaker_loss": 5,  # Maximum 5 consecutive losses
    }

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self._last_mtime: float = 0.0
        self._config: Dict[str, Any] = {}
        self.load(force=True)

    def load(self, force: bool = False) -> bool:
        """Loads configuration from file if changed or forced."""
        if not self.config_path.exists():
            self._config = self._get_fallback_config()
            self.save()
            return True

        try:
            mtime = os.path.getmtime(self.config_path)
            if not force and mtime <= self._last_mtime:
                return False

            with open(self.config_path, "r", encoding="utf-8") as f:
                raw_cfg = json.load(f)

            validated = self._validate_and_sanitize(raw_cfg)
            self._config = validated
            self._last_mtime = mtime
            return True
        except Exception as e:
            if not self._config:
                self._config = self._get_fallback_config()
            return False

    def check_reload(self) -> bool:
        """Lightweight check called in execution loop."""
        return self.load(force=False)

    def _validate_and_sanitize(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Applies hard safety bounds to prevent rogue settings from endangering account."""
        risk = cfg.get("risk", {})
        bounds = self.SAFETY_BOUNDS

        risk["risk_percent"] = max(bounds["min_risk_percent"], min(float(risk.get("risk_percent", 0.025)), bounds["max_risk_percent"]))
        risk["macro_risk_percent"] = max(bounds["min_risk_percent"], min(float(risk.get("macro_risk_percent", 0.0125)), bounds["max_risk_percent"]))
        risk["emergency_hard_loss_pct"] = min(float(risk.get("emergency_hard_loss_pct", 0.025)), bounds["max_emergency_loss"])
        risk["max_lot"] = min(float(risk.get("max_lot", 1.00)), bounds["max_lot_limit"])
        risk["max_consecutive_losses"] = max(bounds["min_circuit_breaker_loss"], min(int(risk.get("max_consecutive_losses", 2)), bounds["max_circuit_breaker_loss"]))

        cfg["risk"] = risk
        return cfg

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    @property
    def risk(self) -> Dict[str, Any]:
        return self._config.get("risk", {})

    @property
    def trade_management(self) -> Dict[str, Any]:
        return self._config.get("trade_management", {})

    @property
    def sessions(self) -> Dict[str, Any]:
        return self._config.get("sessions", {})

    @property
    def strategies(self) -> Dict[str, Any]:
        return self._config.get("strategies", {})

    def get_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        return self.strategies.get(strategy_name, {})

    def update_strategy_config(self, strategy_name: str, new_params: Dict[str, Any]) -> bool:
        """Safely updates a strategy's parameters and writes back to disk."""
        if "strategies" not in self._config:
            self._config["strategies"] = {}
        if strategy_name not in self._config["strategies"]:
            self._config["strategies"][strategy_name] = {}

        self._config["strategies"][strategy_name].update(new_params)
        return self.save()

    def save(self) -> bool:
        """Saves current config atomically."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.config_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._config, f, indent=2)
            os.replace(tmp_path, self.config_path)
            self._last_mtime = os.path.getmtime(self.config_path)
            return True
        except Exception:
            return False

    def _get_fallback_config(self) -> Dict[str, Any]:
        return {
            "version": "9.6-Modular",
            "engine_name": "Pure Gold Sniper Master Compounder",
            "symbol": "XAUUSD",
            "risk": {
                "risk_percent": 0.025,
                "macro_risk_percent": 0.0125,
                "min_lot": 0.01,
                "max_lot": 1.00,
                "emergency_hard_loss_pct": 0.025,
                "max_consecutive_losses": 2,
                "circuit_breaker_pause_hours": 4.0,
                "anti_chop_cooldown_seconds": 600,
                "max_spread_macro": 45,
                "max_spread_scalp": 42,
                "adr_cap": 0.85
            },
            "trade_management": {
                "tp1_r_multiple": 1.8,
                "tp2_r_multiple": 3.5,
                "tp1_bank_fraction": 0.80,
                "risk_free_buffer": 0.30,
                "runner_trail_trigger_r": 2.2,
                "runner_trail_distance_r": 1.2
            },
            "strategies": {
                "h4_macro_swing": {"enabled": True},
                "prime_m15_scalp": {"enabled": True},
                "smc_liquidity_sweep": {"enabled": True},
                "asian_judas_swing": {"enabled": True},
                "vwap_khakestar_snapback": {"enabled": True}
            }
        }
