"""
TradeAgent Telegram Bot.

Provides remote control, alerting, and monitoring via Telegram.
"""

import asyncio
from typing import Callable, Optional

try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, ContextTypes
    HAS_TELEGRAM = True
except ImportError:
    HAS_TELEGRAM = False

from config.settings import settings
from core.logger import get_logger
from core.models import AccountInfo
from core.event_bus import event_bus, EventType, Event

logger = get_logger("telegram_bot")


class TelegramBot:
    def __init__(self):
        self.token = settings.telegram_bot_token
        self.chat_id = settings.telegram_chat_id
        self.app = None
        
        # Callbacks injected by Agent
        self.on_pause_req: Optional[Callable] = None
        self.on_resume_req: Optional[Callable] = None
        self.on_kill_req: Optional[Callable] = None
        self.get_account_info: Optional[Callable] = None
        self.get_positions: Optional[Callable] = None

    async def start(self) -> bool:
        if not HAS_TELEGRAM or not self.token or not self.chat_id:
            logger.warning("Telegram Bot disabled (missing token or library).")
            return False
            
        try:
            self.app = Application.builder().token(self.token).build()
            
            # Register handlers
            self.app.add_handler(CommandHandler("start", self._cmd_start))
            self.app.add_handler(CommandHandler("status", self._cmd_status))
            self.app.add_handler(CommandHandler("positions", self._cmd_positions))
            self.app.add_handler(CommandHandler("pause", self._cmd_pause))
            self.app.add_handler(CommandHandler("resume", self._cmd_resume))
            self.app.add_handler(CommandHandler("kill", self._cmd_kill))
            
            # Subscribe to event bus for alerts
            event_bus.subscribe(EventType.ORDER_FILLED, self._on_trade_event)
            event_bus.subscribe(EventType.POSITION_CLOSED, self._on_trade_event)
            event_bus.subscribe(EventType.RISK_ALERT, self._on_risk_alert)
            
            await self.app.initialize()
            await self.app.start()
            # In a real app we'd use updater.start_polling(), but since we are embedded in an asyncio loop:
            await self.app.updater.start_polling()
            
            logger.info("Telegram Bot started.")
            await self.send_message("🤖 *TradeAgent Started* 🚀\nMode: " + settings.trade_mode.upper())
            return True
        except Exception as e:
            logger.error(f"Telegram start error: {e}")
            return False

    async def stop(self):
        if self.app:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
            logger.info("Telegram Bot stopped.")

    async def send_message(self, text: str):
        if not self.app or not self.chat_id: return
        try:
            await self.app.bot.send_message(
                chat_id=self.chat_id, 
                text=text,
                parse_mode="Markdown"
            )
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")

    # ─── Event Handlers ─────────────────────────────────────────

    async def _on_trade_event(self, event: Event):
        trade = event.data
        if event.event_type == EventType.ORDER_FILLED:
            msg = (
                f"🟢 *TRADE OPENED*\n"
                f"Pair: {trade.symbol}\n"
                f"Side: {trade.side.value.upper()}\n"
                f"Entry: {trade.filled_price}\n"
                f"SL: {trade.stop_loss}\n"
                f"TP: {trade.take_profit}"
            )
        elif event.event_type == EventType.POSITION_CLOSED:
            pnl_icon = "💰" if trade.realized_pnl > 0 else "🩸"
            msg = (
                f"{pnl_icon} *TRADE CLOSED*\n"
                f"Pair: {trade.symbol}\n"
                f"Side: {trade.side.value.upper()}\n"
                f"Exit: {trade.close_price}\n"
                f"PnL: *${trade.realized_pnl:.2f}*"
            )
        await self.send_message(msg)
        
    async def _on_risk_alert(self, event: Event):
        msg = f"⚠️ *RISK ALERT*\n{event.data}"
        await self.send_message(msg)

    # ─── Command Handlers ───────────────────────────────────────

    async def _check_auth(self, update: Update) -> bool:
        if str(update.message.chat_id) != str(self.chat_id):
            logger.warning(f"Unauthorized access attempt from {update.message.chat_id}")
            return False
        return True

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update): return
        msg = "🤖 TradeAgent is running.\nCommands: /status, /positions, /pause, /resume, /kill"
        await update.message.reply_text(msg)

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update): return
        
        if self.get_account_info:
            acc: AccountInfo = await self.get_account_info()
            msg = (
                f"📊 *Account Status*\n"
                f"Balance: ${acc.balance:.2f}\n"
                f"Equity: ${acc.equity:.2f}\n"
                f"Margin Free: ${acc.margin_available:.2f}\n"
                f"Open Positions: {acc.open_positions}"
            )
        else:
            msg = "Account info provider not linked."
            
        await update.message.reply_markdown(msg)

    async def _cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update): return
        
        if self.get_positions:
            pos_list = await self.get_positions()
            if not pos_list:
                msg = "No open positions."
            else:
                msg = "📈 *Open Positions*\n"
                for p in pos_list:
                    icon = "🟢" if p.unrealized_pnl > 0 else "🔴"
                    msg += f"{icon} {p.symbol} {p.side.value.upper()} | PnL: ${p.unrealized_pnl:.2f}\n"
        else:
            msg = "Position provider not linked."
            
        await update.message.reply_markdown(msg)

    async def _cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update): return
        if self.on_pause_req:
            await self.on_pause_req()
            await update.message.reply_text("⏸️ Trading paused. Existing positions remain open.")

    async def _cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update): return
        if self.on_resume_req:
            await self.on_resume_req()
            await update.message.reply_text("▶️ Trading resumed.")

    async def _cmd_kill(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update): return
        if self.on_kill_req:
            await update.message.reply_text("☠️ KILL SWITCH ACTIVATED. Closing all positions and halting.")
            await self.on_kill_req()
