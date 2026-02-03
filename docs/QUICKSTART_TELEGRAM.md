# 🚀 Quick Start: Telegram Alerts for Your Team

## TL;DR (30 seconds)

1. **One person creates bot**: Message `@BotFather` → `/newbot` → Get token
2. **Everyone gets chat_id**: Message `@userinfobot` → Copy your ID
3. **Everyone starts bot**: Search for your team bot → Click "Start"
4. **Configure `.env`**:
   ```bash
   cp .env.example .env
   # Edit .env and paste:
   TELEGRAM_BOT_TOKEN=123456789:ABC...
   TELEGRAM_CHAT_IDS=987654321,876543210,765432109
   ```
5. **Install & test**:
   ```bash
   pip install python-telegram-bot
   python examples/setup_telegram.py
   ```

✅ **Done!** Everyone receives alerts via Telegram.

---

## Usage in Your Code

### Recommended (Production)
```python
from jsf.alerts import create_alert_manager_from_config, Alert, AlertSeverity

# Auto-loads from .env - zero configuration needed!
manager = create_alert_manager_from_config()

# Send alerts
manager.send(Alert(
    title="🎯 Signal Detected",
    message="Strong buy signal for AAPL (confidence: 87%)",
    severity=AlertSeverity.WARNING
))
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
└── src/jsf/
    ├── config.py         ← Configuration management
    └── alerts/
        └── factory.py    ← Easy setup functions
```

---

## Security Checklist

- ✅ `.env` is in `.gitignore` (already done)
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
