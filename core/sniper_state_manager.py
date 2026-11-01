"""
Production State Persistence & Crash Recovery for Pure Gold Sniper.
===================================================================
Guarantees that active tracked positions, circuit breakers, and trailing state
survive process restarts, machine reboots, and network disconnects.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

DEFAULT_STATE_PATH = Path(__file__).resolve().parent.parent / "data_store" / "sniper_state.json"


class SniperStateManager:
    def __init__(self, state_path: Optional[Path] = None):
        self.state_path = state_path or DEFAULT_STATE_PATH
        self.active_tracked_tickets: Dict[int, Dict[str, Any]] = {}
        self.consecutive_losses: int = 0
        self.cooldown_until_time: float = 0.0
        self.last_micro_exit_time: float = 0.0
        self.closed_history: List[Dict[str, Any]] = []
        self.load()

    def load(self) -> bool:
        """Loads state from JSON file."""
        if not self.state_path.exists():
            return False

        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Convert string keys to int for tickets
            raw_tickets = data.get("active_tracked_tickets", {})
            self.active_tracked_tickets = {int(k): v for k, v in raw_tickets.items()}
            self.consecutive_losses = int(data.get("consecutive_losses", 0))
            self.cooldown_until_time = float(data.get("cooldown_until_time", 0.0))
            self.last_micro_exit_time = float(data.get("last_micro_exit_time", 0.0))
            self.closed_history = data.get("closed_history", [])[-50:]  # Keep last 50
            return True
        except Exception as e:
            return False

    def save(self) -> bool:
        """Atomically saves state to disk."""
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "active_tracked_tickets": {str(k): v for k, v in self.active_tracked_tickets.items()},
                "consecutive_losses": self.consecutive_losses,
                "cooldown_until_time": self.cooldown_until_time,
                "last_micro_exit_time": self.last_micro_exit_time,
                "closed_history": self.closed_history[-50:],
                "last_updated": time.time(),
            }
            tmp_path = self.state_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, self.state_path)
            return True
        except Exception:
            return False

    def register_open_trade(self, ticket: int, setup_meta: Dict[str, Any]) -> None:
        """Registers a newly opened trade."""
        self.active_tracked_tickets[ticket] = {
            "lane": setup_meta.get("lane", "SCALP"),
            "strategy": setup_meta.get("strategy", ""),
            "target_tp1": setup_meta.get("tp1", 0.0),
            "target_tp2": setup_meta.get("tp2", 0.0),
            "half_closed": False,
            "risk_dist": setup_meta.get("risk_dist", 2.50),
            "entry": setup_meta.get("entry", 0.0),
            "tp1_lot": setup_meta.get("tp1_lot", 0.0),
            "runner_lot": setup_meta.get("runner_lot", 0.0),
            "opened_at": time.time(),
        }
        self.save()

    def record_partial_close(self, ticket: int) -> None:
        """Marks TP1 as banked and resets consecutive losses."""
        if ticket in self.active_tracked_tickets:
            self.active_tracked_tickets[ticket]["half_closed"] = True
            self.consecutive_losses = 0
            self.save()

    def record_loss_circuit_breaker(self, pause_hours: float = 4.0) -> None:
        """Increments consecutive losses and triggers pause if limit reached."""
        self.consecutive_losses += 1
        if self.consecutive_losses >= 2:
            self.cooldown_until_time = time.time() + (pause_hours * 3600)
        self.save()

    def reconcile_with_live_positions(self, current_open_tickets: List[int]) -> List[int]:
        """
        Reconciles in-memory state with live broker open positions.
        Returns list of tickets that closed since last check.
        """
        open_set = set(current_open_tickets)
        closed_tickets = []

        for old_ticket in list(self.active_tracked_tickets.keys()):
            if old_ticket not in open_set:
                meta = self.active_tracked_tickets.pop(old_ticket)
                closed_tickets.append(old_ticket)
                lane = meta.get("lane", "")
                if "SCALP" in lane or "SMC" in lane or "JUDAS" in lane or "VWAP" in lane:
                    self.last_micro_exit_time = time.time()

                self.closed_history.append({
                    "ticket": old_ticket,
                    "lane": lane,
                    "closed_at": time.time(),
                    "half_closed": meta.get("half_closed", False),
                })

        if closed_tickets:
            self.save()

        return closed_tickets
