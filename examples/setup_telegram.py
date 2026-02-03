"""
Example: Using centralized Telegram configuration for team alerts.

This demonstrates the recommended way to set up alerts in production.
"""

from jsf.alerts import (
    create_alert_manager_from_config,
    Alert,
    AlertSeverity,
)
from jsf.settings import get_config, validate_config


def main():
    """Run configuration validation and test alerts."""
    
    print("=" * 70)
    print("JBAC Strategy Foundry - Centralized Alert Setup")
    print("=" * 70)
    print()
    
    # Step 1: Validate configuration
    print("Step 1: Validating configuration from .env file...")
    print("-" * 70)
    
    warnings = validate_config()
    if warnings:
        print("⚠️  Configuration warnings:")
        for warning in warnings:
            print(f"  - {warning}")
        print()
    else:
        print("✅ Configuration looks good!")
        print()
    
    # Step 2: Load configuration
    print("Step 2: Loading configuration...")
    print("-" * 70)
    
    config = get_config()
    print(f"Environment: {config.env}")
    print(f"Log Level: {config.log_level}")
    print(f"Min Alert Severity: {config.min_alert_severity}")
    print()
    
    # Telegram status
    print("Telegram Configuration:")
    if config.telegram.is_configured:
        print(f"  ✅ Enabled")
        print(f"  📱 Recipients: {len(config.telegram.chat_ids)}")
        if config.telegram.channel_id:
            print(f"  📢 Channel: Configured")
    else:
        print(f"  ❌ Not configured")
        print(f"     Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_IDS to .env")
    print()
    
    # Step 3: Create alert manager
    print("Step 3: Creating alert manager...")
    print("-" * 70)
    
    manager = create_alert_manager_from_config()
    print(f"✅ Alert manager created with {len(manager.alerters)} alerter(s)")
    
    for alerter in manager.alerters:
        print(f"  - {alerter.name}: {'✅ connected' if alerter.is_connected else '❌ not connected'}")
    print()
    
    # Step 4: Send test alerts
    if manager.alerters:
        print("Step 4: Sending test alerts...")
        print("-" * 70)
        
        # Test INFO alert
        manager.send(Alert(
            title="🚀 Setup Complete",
            message="Telegram alerts are configured and working! Team members will receive notifications.",
            severity=AlertSeverity.INFO,
            metadata={"test": True}
        ))
        print("✅ Sent INFO alert")
        
        # Test WARNING alert
        manager.send(Alert(
            title="⚠️ Test Warning",
            message="This is a test warning alert. In production, you'll receive important notifications here.",
            severity=AlertSeverity.WARNING,
        ))
        print("✅ Sent WARNING alert")
        
        print()
        print("=" * 70)
        print("Check your Telegram for the test messages!")
        print("=" * 70)
    else:
        print("⚠️  No alerters configured. Update your .env file.")
        print()
        print("Quick setup:")
        print("1. Copy .env.example to .env")
        print("2. Follow docs/TELEGRAM_SETUP.md")
        print("3. Run this script again")
    
    print()


if __name__ == "__main__":
    main()
