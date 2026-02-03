# 🚀 Quick Start: Telegram Alerts for Paper Trading

## TL;DR (30 seconds)

**Guided Setup (Recommended):**
```bash
python -m jsf.cli.setup_telegram
```
Follow the wizard → Done in 2 minutes! ✅

**Manual Setup:**
1. **Create bot**: Message `@BotFather` → `/newbot` → Get token
2. **Get chat_id**: Message `@userinfobot` → Copy your ID
3. **Start bot**: Search for your bot → Click "Start"
4. **Configure `.env`**:
   ```bash
   cp .env.example .env
   # Edit .env and paste:
   TELEGRAM_BOT_TOKEN=123456789:ABC...
   TELEGRAM_CHAT_IDS=987654321
   ```
5. **Test**:
   ```bash
   python examples/setup_telegram.py
   ```

✅ **Done!** You'll receive paper trading alerts via Telegram.

---

## 📊 What Are Paper Trading Alerts?

**Paper Trading** = Simulated trading with fake money
- ✅ Perfect for learning and testing strategies
- ✅ No real money at risk
- ✅ Same alerts as live trading (for practice)
- ✅ Safe to share with learning groups

**Alerts Include:**
- 📈 Trade signals and executions
- ⚠️  Risk warnings and limits
- 📊 Strategy performance updates
- 🔔 System status changes

All alerts show `[PAPER]` tag so you know it's simulation.

---

## Usage in Your Code

### Recommended (Production)
```python
from jsf.alerts import create_alert_manager_from_config, Alert, AlertSeverity

# Auto-loads from .env - zero configuration needed!
manager = create_alert_manager_from_config()

# Send paper trading alerts (mode tag appears automatically)
manager.send(Alert(
    title="🎯 Signal Detected",
    message="Strong buy signal for AAPL (confidence: 87%)",
    severity=AlertSeverity.WARNING,
    metadata={'mode': 'paper'}  # Shows [PAPER] in Telegram
))
```

**Alert appears as:**
```
⚠️ Signal Detected [PAPER]

Strong buy signal for AAPL (confidence: 87%)
```

### That's it! 
- Config loads automatically from `.env`
- All team members receive the alert
- Console logging happens simultaneously
- Secrets never touch git

---

## File Structure

```
JBAC-Strategy-Foundry/
├── .env.example          ← Template (safe to commit)
├── .env                  ← Your secrets (NEVER commit - in .gitignore)
├── docs/
│   └── TELEGRAM_SETUP.md ← Full guide with troubleshooting
├── examples/
│   └── setup_telegram.py ← Test your configuration
## Security Checklist

- ✅ `.env` is in `.gitignore` (already done)
- ✅ Use `.env.example` as template
- ✅ Keep bot token SECRET (don't share publicly)
- ✅ **Paper trading only** - use demo credentials
- ✅ Never hardcode tokens in code
- ❌ Don't commit `.env` file
- ❌ Don't screenshot tokens
- ❌ Don't use live broker credentials for testing

---

## ⚠️  Before Live Trading

**IMPORTANT:** Paper trading alerts are for learning. Before going live:

1. ✅ Test extensively with paper trading (weeks/months)
2. ✅ Verify strategy performance and risk management  
3. ✅ Consider separate bot for live alerts (different token)
4. ✅ Use live broker's paper trading mode first
5. ✅ Start with small position sizes
6. ⚠️  Live trading alerts will show `[LIVE]` tag

--- `.env` is in `.gitignore` (already done)
- ✅ Use `.env.example` as template
- ✅ Share bot token via **secure channel** (Signal, 1Password)
- ✅ Never hardcode tokens in code
- ❌ Don't commit `.env` file
- ❌ Don't screenshot tokens
- ❌ Don't share in plain text (Slack/email)

---

## Next Steps

1. **Read full guide**: `docs/TELEGRAM_SETUP.md` (5 min read)
2. **Test setup**: `python examples/setup_telegram.py`
3. **Integrate in strategies**: Use `create_alert_manager_from_config()`
4. **Add team members**: Update `TELEGRAM_CHAT_IDS` in `.env`

**Questions?** Check `docs/TELEGRAM_SETUP.md` troubleshooting section.

---

**Made by JBAC Team** | Version 0.6.0-dev | Feb 2026
