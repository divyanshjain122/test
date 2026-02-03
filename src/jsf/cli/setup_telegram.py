"""
Interactive setup wizard for Telegram bot configuration.

Guides users through creating a bot and configuring alerts in under 2 minutes.
"""

import sys
import os
from pathlib import Path


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_step(number: int, text: str):
    """Print a step number."""
    print(f"\n📍 Step {number}: {text}")
    print("-" * 70)


def print_success(text: str):
    """Print success message."""
    print(f"✅ {text}")


def print_warning(text: str):
    """Print warning message."""
    print(f"⚠️  {text}")


def print_info(text: str):
    """Print info message."""
    print(f"💡 {text}")


def get_input(prompt: str, default: str = "") -> str:
    """Get user input with optional default."""
    if default:
        response = input(f"{prompt} [{default}]: ").strip()
        return response if response else default
    return input(f"{prompt}: ").strip()


def confirm(prompt: str) -> bool:
    """Ask yes/no question."""
    while True:
        response = input(f"{prompt} (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        print("Please enter 'y' or 'n'")


def setup_telegram_bot():
    """Interactive Telegram bot setup wizard."""
    
    print_header("🤖 JBAC Strategy Foundry - Telegram Bot Setup Wizard")
    
    print("""
    This wizard will help you set up Telegram alerts for your trading strategies.
    
    ⏱️  Time required: ~2 minutes
    📱 What you'll need: Telegram app on your phone
    🎯 Result: Real-time paper trading alerts on Telegram
    """)
    
    if not confirm("Ready to start?"):
        print("\nSetup cancelled. Run 'python -m jsf.cli setup-telegram' anytime.")
        return
    
    # Step 1: Create bot
    print_step(1, "Create Your Telegram Bot")
    print("""
    1. Open Telegram and search for: @BotFather
    2. Start a chat and send: /newbot
    3. Choose a name (e.g., "My Trading Alerts")
    4. Choose a username (must end with 'bot', e.g., "mytrading_bot")
    5. BotFather will send you a TOKEN (looks like: 123456789:ABCdef...)
    """)
    
    print_warning("Keep your token SECRET - don't share it publicly!")
    
    input("\nPress Enter when you have your bot token...")
    
    bot_token = get_input("\n🔑 Paste your bot token here").strip()
    
    if not bot_token or len(bot_token) < 20:
        print_warning("Invalid token. Please restart setup with valid token.")
        return
    
    # Validate token format (basic check)
    if ':' not in bot_token:
        print_warning("Token doesn't look valid. Format should be: 123456789:ABCdef...")
        if not confirm("Continue anyway?"):
            return
    
    print_success("Bot token saved!")
    
    # Step 2: Get chat ID
    print_step(2, "Get Your Chat ID")
    print("""
    1. Open Telegram and search for: @userinfobot
    2. Start a chat with it
    3. It will reply with your user info
    4. Copy the 'Id' number (e.g., 987654321)
    """)
    
    input("\nPress Enter when you have your chat ID...")
    
    chat_id = get_input("\n💬 Paste your chat ID here").strip()
    
    if not chat_id or not chat_id.replace('-', '').isdigit():
        print_warning("Invalid chat ID. Should be a number like: 987654321")
        if not confirm("Continue anyway?"):
            return
    
    print_success("Chat ID saved!")
    
    # Step 3: Start the bot
    print_step(3, "Start Your Bot")
    print(f"""
    1. Open Telegram and search for your bot
    2. Click the "START" button
    
    ⚠️  IMPORTANT: If you don't start the bot, it can't send you messages!
    """)
    
    input("\nPress Enter after you've started your bot...")
    
    # Step 4: Configure .env file
    print_step(4, "Saving Configuration")
    
    project_root = Path(__file__).parent.parent.parent.parent
    env_file = project_root / ".env"
    env_example = project_root / ".env.example"
    
    # Check if .env exists
    if env_file.exists():
        print_info(f".env file found at: {env_file}")
        if not confirm("Update existing .env file?"):
            print("\nSetup cancelled. Your configuration was not saved.")
            return
    else:
        # Copy from .env.example if it exists
        if env_example.exists():
            print_info("Creating .env from .env.example template...")
            with open(env_example) as f:
                template = f.read()
            with open(env_file, 'w') as f:
                f.write(template)
        else:
            # Create minimal .env
            print_info("Creating new .env file...")
            with open(env_file, 'w') as f:
                f.write("# JBAC Strategy Foundry Configuration\n\n")
    
    # Update .env file
    with open(env_file, 'r') as f:
        content = f.read()
    
    # Replace or add Telegram settings
    if 'TELEGRAM_BOT_TOKEN=' in content:
        # Update existing
        lines = content.split('\n')
        new_lines = []
        for line in lines:
            if line.startswith('TELEGRAM_BOT_TOKEN='):
                new_lines.append(f'TELEGRAM_BOT_TOKEN={bot_token}')
            elif line.startswith('TELEGRAM_CHAT_IDS='):
                new_lines.append(f'TELEGRAM_CHAT_IDS={chat_id}')
            else:
                new_lines.append(line)
        content = '\n'.join(new_lines)
    else:
        # Add new
        content += f"\n# Telegram Configuration\nTELEGRAM_BOT_TOKEN={bot_token}\n"
        content += f"TELEGRAM_CHAT_IDS={chat_id}\n"
        content += "ENABLE_TELEGRAM_ALERTS=true\n"
    
    with open(env_file, 'w') as f:
        f.write(content)
    
    print_success(f"Configuration saved to: {env_file}")
    
    # Step 5: Test connection
    print_step(5, "Testing Connection")
    
    if confirm("Send a test alert now?"):
        try:
            from jsf.alerts import create_alert_manager_from_config, Alert, AlertSeverity
            
            print("\nConnecting to Telegram...")
            manager = create_alert_manager_from_config()
            
            print("Sending test alert...")
            manager.send(Alert(
                title="🎉 Setup Complete!",
                message=(
                    "Your Telegram alerts are configured and working!\n\n"
                    "📊 Mode: Paper Trading (Safe for learning)\n"
                    "🔔 You'll receive alerts for:\n"
                    "  • Trade signals\n"
                    "  • Order executions\n"
                    "  • Risk warnings\n"
                    "  • Strategy events\n\n"
                    "Happy trading! 🚀"
                ),
                severity=AlertSeverity.INFO
            ))
            
            print_success("Test alert sent! Check your Telegram.")
            
        except Exception as e:
            print_warning(f"Test failed: {e}")
            print("\nDon't worry - your config is saved. You can test later with:")
            print("  python examples/setup_telegram.py")
    
    # Final instructions
    print_header("✅ Setup Complete!")
    
    print("""
    🎓 What's Next?
    
    1. Check your Telegram for the test message
    2. Run your first paper trading strategy
    3. Receive real-time alerts on your phone!
    
    📚 Learn More:
    - Quick Start: docs/QUICKSTART_TELEGRAM.md
    - Full Guide: docs/TELEGRAM_SETUP.md
    - Examples: examples/alerts_example.py
    
    🔒 Security Reminder:
    - Your .env file contains secrets - keep it safe!
    - Never commit .env to git (it's already in .gitignore)
    - This is for PAPER TRADING - use demo/test credentials only
    
    ⚠️  Before Live Trading:
    - Test thoroughly with paper trading first
    - Review risk management settings
    - Consider separate bot for live alerts
    
    Happy trading! 🚀
    """)


if __name__ == "__main__":
    try:
        setup_telegram_bot()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nFor help, see: docs/TELEGRAM_SETUP.md")
        sys.exit(1)
