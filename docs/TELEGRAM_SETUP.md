# 🔔 Centralized Telegram Alert Setup Guide

This guide will help your team set up a single shared Telegram bot for trading alerts.

## ⚡ Quick Start (5 minutes)

### Step 1: Create Team Bot (One person does this)

1. **Open Telegram** and search for `@BotFather`
2. **Start a chat** and send: `/newbot`
3. **Choose a name**: `JBAC Strategy Alerts` (or similar)
4. **Choose username**: Must end in "bot", e.g., `jbac_strategy_bot`
5. **Copy the bot token** (looks like: `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`)

✅ **Save this token** - you'll add it to `.env` file

### Step 2: Get Your Personal Chat ID (Everyone does this)

1. **Open Telegram** and search for `@userinfobot`
2. **Start a chat** - it will reply with your info
3. **Copy your chat_id** (e.g., `987654321`)

### Step 3: Start the Team Bot (Everyone does this)

1. **Search for your team bot** in Telegram (e.g., `@jbac_strategy_bot`)
2. **Click "Start"** button

⚠️ **Important**: If you don't start the bot, it can't message you!

### Step 4: Configure Environment (One person does this)

1. **Copy the example file**:
   ```powershell
   Copy-Item .env.example .env
   ```

2. **Edit `.env` file** and add:
   ```bash
   # Paste the bot token from Step 1
   TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
   
   # Add everyone's chat IDs (comma-separated)
   TELEGRAM_CHAT_IDS=987654321,876543210,765432109
   
   # Optional: Create a Telegram channel for team-wide alerts
   TELEGRAM_CHANNEL_ID=
   ```

3. **Commit `.env.example`** (template) but **NEVER commit `.env`** (has secrets)

### Step 5: Install and Test

1. **Install Telegram package**:
   ```powershell
   pip install python-telegram-bot
   ```

2. **Test the setup**:
   ```python
   from jsf.alerts import create_alert_manager_from_config
   from jsf.alerts import Alert, AlertSeverity
   from jsf.settings import get_config, validate_config
   
   # Check configuration
   warnings = validate_config()
   if warnings:
       for warning in warnings:
           print(f"⚠️ {warning}")
   
   # Create manager (auto-loads from .env)
   manager = create_alert_manager_from_config()
   
   # Send test alert
   alert = Alert(
       title="🚀 Setup Complete",
       message="Telegram alerts are working! You'll receive notifications here.",
       severity=AlertSeverity.INFO
   )
   manager.send(alert)
   ```

## 📋 Configuration Options

### Basic Setup (.env file)
```bash
# Required for Telegram alerts
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_IDS=123456789,987654321  # Comma-separated

# Optional
ENABLE_TELEGRAM_ALERTS=true
ENABLE_CONSOLE_ALERTS=true
MIN_ALERT_SEVERITY=INFO  # INFO, WARNING, ERROR, CRITICAL
```

### Usage in Code

**Recommended (Production):**
```python
from jsf.alerts import create_alert_manager_from_config, Alert, AlertSeverity

# Auto-loads configuration from .env
manager = create_alert_manager_from_config()

# Send alerts
manager.send(Alert(
    title="Trade Executed",
    message="Bought 100 AAPL @ $150.00",
    severity=AlertSeverity.INFO
))
```

**Alternative (Quick Testing):**
```python
from jsf.alerts import create_simple_alert_manager

# Direct configuration (bypasses .env)
manager = create_simple_alert_manager(
    telegram_token="123456:ABC...",
    telegram_chat_ids=["123456789"],
    include_console=True
)
```

## 🔒 Security Best Practices

### ✅ DO:
- ✅ Keep `.env` in `.gitignore` (already configured)
- ✅ Share bot token through **secure channels** (encrypted chat, password manager)
- ✅ Use `.env.example` as template in repo
- ✅ Rotate bot token if accidentally exposed (via @BotFather)
- ✅ Use environment variables in production (CI/CD secrets)

### ❌ DON'T:
- ❌ Commit `.env` file to git
- ❌ Share tokens in plain text (Slack, email)
- ❌ Hardcode tokens in source code
- ❌ Screenshot tokens with visible values

### If Token is Leaked:
1. Open Telegram and message `@BotFather`
2. Send `/mybots`
3. Select your bot → **API Token** → **Revoke current token**
4. Update `.env` with new token
5. Notify team members

## 🎯 Advanced: Team Channel Setup

For team-wide announcements (only important alerts):

1. **Create a Telegram channel**:
   - Open Telegram → New Channel
   - Name: "JBAC Trading Alerts"
   - Make it private, add team members

2. **Add your bot as admin**:
   - Channel Settings → Administrators → Add Administrator
   - Search for your bot, add it

3. **Get channel ID**:
   ```python
   # Send a message to your channel, then run:
   from telegram import Bot
   
   bot = Bot(token="YOUR_BOT_TOKEN")
   updates = bot.get_updates()
   
   # Look for channel_post with your message
   for update in updates:
       if hasattr(update, 'channel_post'):
           print(f"Channel ID: {update.channel_post.chat.id}")
   ```

4. **Add to `.env`**:
   ```bash
   TELEGRAM_CHANNEL_ID=-1001234567890  # Negative number for channels
   ```

5. **Channel receives only WARNING+ alerts** (configured in factory.py)

## 🚨 Troubleshooting

### "Bot can't message user"
- **Solution**: User must click "Start" on the bot first

### "Unauthorized" error
- **Solution**: Check token is correct in `.env`
- Regenerate token if needed via @BotFather

### "Chat not found"
- **Solution**: Verify chat_id is correct (from @userinfobot)
- Ensure user started the bot

### "python-telegram-bot not installed"
- **Solution**: `pip install python-telegram-bot`

### Alerts not sending
- **Solution**: 
  ```python
  from jsf.settings import validate_config
  warnings = validate_config()
  print(warnings)
  ```

## 📊 Alert Severity Levels

- **INFO**: Regular events (trades, signals)
- **WARNING**: Important events (risk limits approaching)
- **ERROR**: Problems (strategy errors, data issues)
- **CRITICAL**: Urgent issues (system failures, major losses)

Configure minimum level in `.env`:
```bash
MIN_ALERT_SEVERITY=WARNING  # Only send WARNING and above
```

## 🔄 Updating Configuration

To add new team members:

1. **New member gets chat_id** (message @userinfobot)
2. **New member starts bot** (search bot, click Start)
3. **Update `.env`**:
   ```bash
   # Add new chat_id to the list
   TELEGRAM_CHAT_IDS=old_id1,old_id2,new_member_id
   ```
4. **Restart application** (config is cached)

## 📚 Example: Production Strategy

```python
from jsf.alerts import create_alert_manager_from_config, Alert, AlertSeverity
from jsf.strategy import Strategy

# Initialize alerts (auto-configured from .env)
alert_manager = create_alert_manager_from_config()

class MyStrategy(Strategy):
    def on_start(self):
        alert_manager.send(Alert(
            title="Strategy Started",
            message=f"Running {self.name} in {self.mode} mode",
            severity=AlertSeverity.INFO
        ))
    
    def on_signal(self, signal):
        if signal.strength > 0.8:
            alert_manager.send(Alert(
                title=f"Strong Signal: {signal.symbol}",
                message=f"Signal strength: {signal.strength:.2%}",
                severity=AlertSeverity.WARNING
            ))
    
    def on_error(self, error):
        alert_manager.send(Alert(
            title="Strategy Error",
            message=str(error),
            severity=AlertSeverity.ERROR,
            metadata={"traceback": error.__traceback__}
        ))
```

## 🎓 Team Workflow

1. **One person** creates bot and shares token securely
2. **Everyone** gets their chat_id and starts the bot
3. **One person** configures `.env` with all chat_ids
4. **Everyone** pulls latest `.env.example` and creates local `.env`
5. **All** receive alerts when strategies run!

---

**Questions?** Check logs or run `validate_config()` to diagnose issues.
