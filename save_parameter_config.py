#!/usr/bin/env python3
"""
Parameter Configuration Saver

This script shows how to save and load parameter configurations
from the enhanced GUI for consistent analysis reproduction.
"""

import json
import os
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox

def save_current_parameters(gui_instance):
    """Save current parameter configuration from GUI."""
    config = {
        "timestamp": datetime.now().isoformat(),
        "description": "Monte Carlo GUI Parameter Configuration",

        # Data & Strategy Parameters
        "strategy": {
            "preset": gui_instance.strategy_preset_var.get(),
            "algorithm": gui_instance.algorithm_var.get(),
            "risk_management": gui_instance.risk_mgmt_var.get(),
            "stop_loss": gui_instance.stop_loss_var.get(),
            "take_profit": gui_instance.take_profit_var.get(),
            "initial_capital": gui_instance.capital_var.get(),
            "ticker": gui_instance.ticker_var.get(),
            "period": gui_instance.period_var.get(),
            "interval": gui_instance.interval_var.get()
        },

        # Monte Carlo Parameters
        "monte_carlo": {
            "preset": gui_instance.mc_preset_var.get(),
            "simulations": gui_instance.num_sims_var.get(),
            "method": gui_instance.sim_method_var.get(),
            "confidence_level": gui_instance.confidence_var.get(),
            "risk_free_rate": gui_instance.risk_free_var.get()
        },

        # Scenario Parameters
        "scenarios": {
            "preset": gui_instance.scenario_preset_var.get(),
            "num_scenarios": gui_instance.num_scenarios_var.get(),
            "scenario_length": gui_instance.scenario_length_var.get(),
            "volatility_scale": gui_instance.volatility_scale_var.get(),
            "trend_strength": gui_instance.trend_strength_var.get()
        },

        # Portfolio Parameters
        "portfolio": {
            "preset": gui_instance.portfolio_preset_var.get(),
            "assets": gui_instance.assets_var.get(),
            "method": gui_instance.opt_method_var.get(),
            "target_risk": gui_instance.target_risk_var.get(),
            "target_return": gui_instance.target_return_var.get()
        },

        # Liquidity Parameters (if available)
        "liquidity": {
            "available": hasattr(gui_instance, 'liquidity_analyzer'),
            "ticker": getattr(gui_instance, 'liq_ticker_var', tk.StringVar()).get() if hasattr(gui_instance, 'liq_ticker_var') else "",
            "period": getattr(gui_instance, 'liq_period_var', tk.StringVar()).get() if hasattr(gui_instance, 'liq_period_var') else "",
            "interval": getattr(gui_instance, 'liq_interval_var', tk.StringVar()).get() if hasattr(gui_instance, 'liq_interval_var') else "",
            "preset": getattr(gui_instance, 'preset_var', tk.StringVar()).get() if hasattr(gui_instance, 'preset_var') else "",
            "swing_sensitivity": getattr(gui_instance, 'swing_sens_var', tk.IntVar()).get() if hasattr(gui_instance, 'swing_sens_var') else 3,
            "zone_sensitivity": getattr(gui_instance, 'zone_sens_var', tk.DoubleVar()).get() if hasattr(gui_instance, 'zone_sens_var') else 1.5,
            "volume_threshold": getattr(gui_instance, 'volume_threshold_var', tk.DoubleVar()).get() if hasattr(gui_instance, 'volume_threshold_var') else 1.2,
            "scoring_method": getattr(gui_instance, 'scoring_method_var', tk.StringVar()).get() if hasattr(gui_instance, 'scoring_method_var') else "Sophisticated"
        }
    }

    return config

def save_config_to_file(gui_instance, filename=None):
    """Save parameter configuration to JSON file."""
    if filename is None:
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save Parameter Configuration"
        )

    if filename:
        try:
            config = save_current_parameters(gui_instance)

            with open(filename, 'w') as f:
                json.dump(config, f, indent=2)

            messagebox.showinfo("Configuration Saved",
                              f"Parameter configuration saved to:\n{filename}")

            return True

        except Exception as e:
            messagebox.showerror("Save Error",
                               f"Failed to save configuration:\n{str(e)}")
            return False

    return False

def load_config_from_file(gui_instance):
    """Load parameter configuration from JSON file."""
    filename = filedialog.askopenfilename(
        defaultextension=".json",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        title="Load Parameter Configuration"
    )

    if filename:
        try:
            with open(filename, 'r') as f:
                config = json.load(f)

            # Apply loaded configuration
            apply_config_to_gui(gui_instance, config)

            messagebox.showinfo("Configuration Loaded",
                              f"Parameter configuration loaded from:\n{filename}")

            return True

        except Exception as e:
            messagebox.showerror("Load Error",
                               f"Failed to load configuration:\n{str(e)}")
            return False

    return False

def apply_config_to_gui(gui_instance, config):
    """Apply loaded configuration to GUI parameters."""
    try:
        # Strategy parameters
        if 'strategy' in config:
            strat = config['strategy']
            gui_instance.strategy_preset_var.set(strat.get('preset', 'Balanced'))
            gui_instance.algorithm_var.set(strat.get('algorithm', ''))
            gui_instance.risk_mgmt_var.set(strat.get('risk_management', 0.02))
            gui_instance.stop_loss_var.set(strat.get('stop_loss', 0.05))
            gui_instance.take_profit_var.set(strat.get('take_profit', 0.10))
            gui_instance.capital_var.set(strat.get('initial_capital', '10000'))
            gui_instance.ticker_var.set(strat.get('ticker', 'SPY'))
            gui_instance.period_var.set(strat.get('period', '1y'))
            gui_instance.interval_var.set(strat.get('interval', '1d'))

            # Trigger parameter updates
            gui_instance.on_strategy_preset_change()

        # Monte Carlo parameters
        if 'monte_carlo' in config:
            mc = config['monte_carlo']
            gui_instance.mc_preset_var.set(mc.get('preset', 'Balanced'))
            gui_instance.num_sims_var.set(mc.get('simulations', '1000'))
            gui_instance.sim_method_var.set(mc.get('method', 'synthetic_returns'))
            gui_instance.confidence_var.set(mc.get('confidence_level', 0.95))
            gui_instance.risk_free_var.set(mc.get('risk_free_rate', 0.045))

            # Trigger parameter updates
            gui_instance.on_mc_preset_change()

        # Scenario parameters
        if 'scenarios' in config:
            scen = config['scenarios']
            gui_instance.scenario_preset_var.set(scen.get('preset', 'Balanced'))
            gui_instance.num_scenarios_var.set(scen.get('num_scenarios', '100'))
            gui_instance.scenario_length_var.set(scen.get('scenario_length', '126'))
            gui_instance.volatility_scale_var.set(scen.get('volatility_scale', 1.0))
            gui_instance.trend_strength_var.set(scen.get('trend_strength', 0.5))

            # Trigger parameter updates
            gui_instance.on_scenario_preset_change()

        # Portfolio parameters
        if 'portfolio' in config:
            port = config['portfolio']
            gui_instance.portfolio_preset_var.set(port.get('preset', 'Balanced'))
            gui_instance.assets_var.set(port.get('assets', 'AAPL,MSFT,GOOGL,TSLA'))
            gui_instance.opt_method_var.set(port.get('method', 'synthetic_prices'))
            gui_instance.target_risk_var.set(port.get('target_risk', 0.15))
            gui_instance.target_return_var.set(port.get('target_return', 0.12))

            # Trigger parameter updates
            gui_instance.on_portfolio_preset_change()

        # Liquidity parameters
        if 'liquidity' in config and hasattr(gui_instance, 'liq_ticker_var'):
            liq = config['liquidity']
            if hasattr(gui_instance, 'liq_ticker_var'):
                gui_instance.liq_ticker_var.set(liq.get('ticker', 'AAPL'))
            if hasattr(gui_instance, 'liq_period_var'):
                gui_instance.liq_period_var.set(liq.get('period', '3mo'))
            if hasattr(gui_instance, 'liq_interval_var'):
                gui_instance.liq_interval_var.set(liq.get('interval', '1d'))
            if hasattr(gui_instance, 'preset_var'):
                gui_instance.preset_var.set(liq.get('preset', 'Balanced'))
            if hasattr(gui_instance, 'swing_sens_var'):
                gui_instance.swing_sens_var.set(liq.get('swing_sensitivity', 3))
            if hasattr(gui_instance, 'zone_sens_var'):
                gui_instance.zone_sens_var.set(liq.get('zone_sensitivity', 1.5))
            if hasattr(gui_instance, 'volume_threshold_var'):
                gui_instance.volume_threshold_var.set(liq.get('volume_threshold', 1.2))
            if hasattr(gui_instance, 'scoring_method_var'):
                gui_instance.scoring_method_var.set(liq.get('scoring_method', 'Sophisticated'))

            # Trigger parameter updates
            if hasattr(gui_instance, 'on_preset_change'):
                gui_instance.on_preset_change()

        print("✅ Configuration applied successfully")

    except Exception as e:
        print(f"❌ Error applying configuration: {e}")
        raise

def create_config_template():
    """Create a template configuration file."""
    template = {
        "template": True,
        "description": "Parameter Configuration Template",
        "created": datetime.now().isoformat(),
        "strategy": {
            "preset": "Balanced",
            "algorithm": "MovingAverageCrossover",
            "risk_management": 0.02,
            "stop_loss": 0.05,
            "take_profit": 0.10,
            "initial_capital": "10000",
            "ticker": "SPY",
            "period": "1y",
            "interval": "1d"
        },
        "monte_carlo": {
            "preset": "Balanced",
            "simulations": "1000",
            "method": "synthetic_returns",
            "confidence_level": 0.95,
            "risk_free_rate": 0.045
        },
        "scenarios": {
            "preset": "Balanced",
            "num_scenarios": "100",
            "scenario_length": "126",
            "volatility_scale": 1.0,
            "trend_strength": 0.5
        },
        "portfolio": {
            "preset": "Balanced",
            "assets": "AAPL,MSFT,GOOGL,TSLA",
            "method": "synthetic_prices",
            "target_risk": 0.15,
            "target_return": 0.12
        },
        "liquidity": {
            "ticker": "AAPL",
            "period": "3mo",
            "interval": "1d",
            "preset": "Balanced",
            "swing_sensitivity": 3,
            "zone_sensitivity": 1.5,
            "volume_threshold": 1.2,
            "scoring_method": "Sophisticated"
        }
    }

    filename = f"parameter_template_{datetime.now().strftime('%Y%m%d')}.json"

    with open(filename, 'w') as f:
        json.dump(template, f, indent=2)

    print(f"✅ Template created: {filename}")
    return filename

def demo_parameter_saving():
    """Demonstrate parameter saving functionality."""
    print("💾 PARAMETER CONFIGURATION SAVING DEMO")
    print("=" * 60)

    print("📋 This demo shows how to save and load parameter configurations:")
    print()

    print("🎯 SAVE CONFIGURATION:")
    print("1. Set up your preferred parameters in the GUI")
    print("2. Run this script: save_config_to_file(gui_instance)")
    print("3. Choose save location and filename")
    print("4. JSON file created with all current parameters")
    print()

    print("🎯 LOAD CONFIGURATION:")
    print("1. Run this script: load_config_from_file(gui_instance)")
    print("2. Select previously saved JSON file")
    print("3. All parameters automatically restored")
    print("4. Ready to run analysis with saved settings")
    print()

    print("🎯 TEMPLATE CREATION:")
    print("1. Run: create_config_template()")
    print("2. Template file created for customization")
    print("3. Edit template with your preferred settings")
    print("4. Load template into GUI when needed")
    print()

    # Create template as example
    template_file = create_config_template()

    print("📄 TEMPLATE FILE CREATED:")
    print(f"   File: {template_file}")
    print("   Edit this file to create your custom parameter presets")
    print()

    print("💡 PRO TIP: Save configurations after finding optimal settings")
    print("   This allows you to reproduce successful analysis setups!")

if __name__ == "__main__":
    demo_parameter_saving()


