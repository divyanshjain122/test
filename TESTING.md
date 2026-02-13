# Testing Guide for JSF-Core Dashboard

This guide explains how to test the JSF-Core Streamlit dashboard with real or mock data.

## Quick Start - Mock Data (No API Keys Required)

The dashboard includes a mock data collector that generates synthetic trading data.

### 1. Start the Dashboard

```powershell
# Make sure you're in the project root
cd C:\Users\Jai Ansh Bindra\JBAC-Strategy-Foundry

# Run with Python 3.11
& "C:/Users/Jai Ansh Bindra/AppData/Local/Programs/Python/Python311/python.exe" -m streamlit run src\jsf\dashboard\app.py
```

The dashboard will launch at **http://localhost:8503** (or next available port)

### 2. Navigate the Dashboard

The dashboard has several pages:
- **P&L**: Equity curve, returns, and performance metrics
- **Portfolio**: Current positions and allocation
- **Trades**: Trade history and execution stats
- **Risk**: Drawdown, VaR, and risk metrics

All pages work with mock data by default!

## Testing with Real Market Data

### Option 1: Alpaca Paper Trading (Recommended)

Free paper trading account with real market data.

#### Setup:

1. **Sign up for Alpaca**:
   - Visit: https://alpaca.markets/
   - Create a free paper trading account
   - Get your API keys from the dashboard

2. **Configure Environment Variables**:
   ```powershell
   # Create .env file
   Copy-Item .env.example .env
   
   # Edit .env and add your keys:
   # ALPACA_API_KEY=PKxxxxxxxxx
   # ALPACA_SECRET_KEY=yyyyyyyy
   # ALPACA_BASE_URL=https://paper-api.alpaca.markets
   ```

3. **Install Alpaca SDK** (if not already installed):
   ```powershell
   & "C:/Users/Jai Ansh Bindra/AppData/Local/Programs/Python/Python311/python.exe" -m pip install alpaca-trade-api
   ```

4. **Test Connection**:
   ```powershell
   & "C:/Users/Jai Ansh Bindra/AppData/Local/Programs/Python/Python311/python.exe" -c "from alpaca_trade_api import REST; import os; api = REST(os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_SECRET_KEY'), 'https://paper-api.alpaca.markets'); print('Connected:', api.get_account().status)"
   ```

### Option 2: Yahoo Finance (Free, No Registration)

Uses historical data only (no live trading).

#### Setup:

1. **Install yfinance**:
   ```powershell
   & "C:/Users/Jai Ansh Bindra/AppData/Local/Programs/Python/Python311/python.exe" -m pip install yfinance
   ```

2. **Run a Demo**:
   ```powershell
   & "C:/Users/Jai Ansh Bindra/AppData/Local/Programs/Python/Python311/python.exe" demos\demo_complete.py
   ```

### Option 3: Alpha Vantage API

Free tier: 25 API calls per day.

#### Setup:

1. **Get API Key**:
   - Visit: https://www.alphavantage.co/support/#api-key
   - Sign up for free API key

2. **Add to .env**:
   ```
   ALPHA_VANTAGE_API_KEY=your_key_here
   ```

## Testing Telegram Alerts

1. **Create Telegram Bot**:
   - Message @BotFather on Telegram
   - Send `/newbot` and follow instructions
   - Copy the bot token

2. **Get Your Chat ID**:
   - Message @userinfobot
   - Copy your chat ID

3. **Configure .env**:
   ```
   TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
   TELEGRAM_CHAT_ID=123456789
   ```

4. **Test Alerts**:
   ```powershell
   & "C:/Users/Jai Ansh Bindra/AppData/Local/Programs/Python/Python311/python.exe" examples\alerts_example.py
   ```

## Running Example Strategies

### Basic Backtest (No API Keys)
```powershell
& "C:/Users/Jai Ansh Bindra/AppData/Local/Programs/Python/Python311/python.exe" examples\quickstart.py
```

### Complete Backtest with Metrics
```powershell
& "C:/Users/Jai Ansh Bindra/AppData/Local/Programs/Python/Python311/python.exe" examples\complete_backtest_example.py
```

### ML Strategy Example
```powershell
& "C:/Users/Jai Ansh Bindra/AppData/Local/Programs/Python/Python311/python.exe" examples\ml_example.py
```

### Paper Trading (requires Alpaca keys)
```powershell
& "C:/Users/Jai Ansh Bindra/AppData/Local/Programs/Python/Python311/python.exe" examples\paper_trading_alerts.py
```

## Troubleshooting

### Dashboard won't start
```powershell
# Check if streamlit is installed
& "C:/Users/Jai Ansh Bindra/AppData/Local/Programs/Python/Python311/python.exe" -m pip list | Select-String streamlit

# Reinstall if needed
& "C:/Users/Jai Ansh Bindra/AppData/Local/Programs/Python/Python311/python.exe" -m pip install streamlit plotly
```

### Import errors
```powershell
# Install JSF in editable mode
& "C:/Users/Jai Ansh Bindra/AppData/Local/Programs/Python/Python311/python.exe" -m pip install -e . --no-deps
```

### Port already in use
```powershell
# Streamlit will automatically try next available port
# Or specify a port manually:
& "C:/Users/Jai Ansh Bindra/AppData/Local/Programs/Python/Python311/python.exe" -m streamlit run src\jsf\dashboard\app.py --server.port=8504
```

## Security Reminders

1. **Never commit .env file** - It's already in .gitignore
2. **Use paper trading first** - Test with fake money before real trading
3. **Keep API keys secure** - Don't share them or post in public
4. **Rotate keys regularly** - Especially if you suspect they've been compromised
5. **Start small** - Even in paper trading, test with realistic (small) amounts

## Next Steps

1. Review the examples in `examples/` directory
2. Read the demo scripts in `demos/` for more advanced usage
3. Check CONTRIBUTING.md for development guidelines
4. See docs/ for detailed API documentation
