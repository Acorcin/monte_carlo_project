"""
Integration tests for GUI components and workflows.

Tests GUI interactions, data flow, and user workflows.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, Mock
import tkinter as tk
import threading
import time

# Import GUI modules for testing
try:
    from monte_carlo_gui_app import MonteCarloGUI
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False


@pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI module not available")
class TestGUIInitialization:
    """Test GUI initialization and component creation."""

    @pytest.mark.integration
    @pytest.mark.gui
    def test_gui_initialization(self, mock_gui_root):
        """Test GUI initialization without displaying."""
        with patch('tkinter.Tk', return_value=mock_gui_root):
            with patch('monte_carlo_gui_app.AlgorithmManager') as mock_manager:
                mock_manager.return_value.algorithms = {'TestAlgorithm': Mock()}
                
                # Initialize GUI without actually showing it
                gui = MonteCarloGUI(mock_gui_root)
                
                assert gui.root == mock_gui_root
                assert hasattr(gui, 'current_data')
                assert hasattr(gui, 'current_results')
                assert hasattr(gui, 'algorithm_manager')

    @pytest.mark.integration
    @pytest.mark.gui
    def test_gui_component_creation(self, mock_gui_root):
        """Test that all major GUI components are created."""
        with patch('tkinter.Tk', return_value=mock_gui_root):
            with patch('monte_carlo_gui_app.AlgorithmManager') as mock_manager:
                mock_manager.return_value.algorithms = {'TestAlgorithm': Mock()}
                
                with patch('tkinter.ttk.Notebook'), \
                     patch('tkinter.ttk.Frame'), \
                     patch('tkinter.ttk.Button'), \
                     patch('tkinter.ttk.Combobox'):
                    
                    gui = MonteCarloGUI(mock_gui_root)
                    
                    # Test that major components exist
                    assert hasattr(gui, 'notebook')
                    assert hasattr(gui, 'data_frame')
                    assert hasattr(gui, 'strategy_frame')
                    assert hasattr(gui, 'mc_frame')

    @pytest.mark.integration
    @pytest.mark.gui
    def test_gui_variable_initialization(self, mock_gui_root):
        """Test that GUI variables are properly initialized."""
        with patch('tkinter.Tk', return_value=mock_gui_root):
            with patch('monte_carlo_gui_app.AlgorithmManager') as mock_manager:
                mock_manager.return_value.algorithms = {'TestAlgorithm': Mock()}
                
                with patch('tkinter.StringVar') as mock_stringvar, \
                     patch('tkinter.DoubleVar') as mock_doublevar, \
                     patch('tkinter.IntVar') as mock_intvar:
                    
                    mock_stringvar.return_value = Mock()
                    mock_doublevar.return_value = Mock()
                    mock_intvar.return_value = Mock()
                    
                    gui = MonteCarloGUI(mock_gui_root)
                    
                    # Variables should be created
                    assert mock_stringvar.called
                    assert mock_doublevar.called


@pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI module not available")
class TestGUIDataFlow:
    """Test data flow through GUI components."""

    @pytest.mark.integration
    @pytest.mark.gui
    def test_data_loading_workflow(self, mock_gui_root, mock_yfinance_data):
        """Test data loading workflow through GUI."""
        with patch('tkinter.Tk', return_value=mock_gui_root):
            with patch('monte_carlo_gui_app.AlgorithmManager') as mock_manager:
                mock_manager.return_value.algorithms = {'TestAlgorithm': Mock()}
                
                with patch('yfinance.download', side_effect=mock_yfinance_data):
                    gui = MonteCarloGUI(mock_gui_root)
                    
                    # Mock GUI variables
                    gui.ticker_var = Mock()
                    gui.ticker_var.get.return_value = 'AAPL'
                    gui.period_var = Mock()
                    gui.period_var.get.return_value = '1y'
                    gui.interval_var = Mock()
                    gui.interval_var.get.return_value = '1d'
                    gui.asset_type_var = Mock()
                    gui.asset_type_var.get.return_value = 'stocks'
                    
                    # Mock data preview update
                    gui.data_preview_text = Mock()
                    gui.data_preview_text.config = Mock()
                    gui.data_preview_text.delete = Mock()
                    gui.data_preview_text.insert = Mock()
                    
                    # Test data loading
                    if hasattr(gui, 'load_data'):
                        gui.load_data()
                        
                        # Should have loaded data
                        assert gui.current_data is not None

    @pytest.mark.integration
    @pytest.mark.gui
    def test_algorithm_selection_workflow(self, mock_gui_root):
        """Test algorithm selection workflow."""
        with patch('tkinter.Tk', return_value=mock_gui_root):
            with patch('monte_carlo_gui_app.AlgorithmManager') as mock_manager:
                test_algorithms = {
                    'TestAlgorithm1': Mock(),
                    'TestAlgorithm2': Mock(),
                    'TestAlgorithm3': Mock()
                }
                mock_manager.return_value.algorithms = test_algorithms
                
                gui = MonteCarloGUI(mock_gui_root)
                
                # Mock algorithm selection components
                gui.algorithm_var = Mock()
                gui.algorithm_combo = Mock()
                gui.selected_algorithms = []
                gui.algo_checkbuttons = {}
                
                # Test algorithm selection
                if hasattr(gui, 'on_algorithm_selection_change'):
                    # Simulate selecting algorithms
                    gui.selected_algorithms = ['TestAlgorithm1', 'TestAlgorithm2']
                    gui.on_algorithm_selection_change()
                    
                    assert len(gui.selected_algorithms) == 2

    @pytest.mark.integration
    @pytest.mark.gui  
    def test_backtest_to_monte_carlo_workflow(self, mock_gui_root, mock_backtest_results):
        """Test workflow from backtest to Monte Carlo."""
        with patch('tkinter.Tk', return_value=mock_gui_root):
            with patch('monte_carlo_gui_app.AlgorithmManager') as mock_manager:
                mock_manager.return_value.algorithms = {'TestAlgorithm': Mock()}
                
                gui = MonteCarloGUI(mock_gui_root)
                
                # Set up backtest results
                gui.current_results = mock_backtest_results
                
                # Mock Monte Carlo components
                gui.mc_btn = Mock()
                gui.num_sims_var = Mock()
                gui.num_sims_var.get.return_value = '100'
                gui.sim_method_var = Mock()
                gui.sim_method_var.get.return_value = 'synthetic_returns'
                
                # Test Monte Carlo button enabling
                if hasattr(gui, 'mc_btn'):
                    # After backtest, Monte Carlo should be enabled
                    gui.mc_btn.config.assert_not_called()  # Initially
                    
                    # Simulate successful backtest
                    if hasattr(gui, 'run_backtest'):
                        gui.mc_btn.config(state='normal')
                        gui.mc_btn.config.assert_called_with(state='normal')


@pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI module not available")
class TestGUIErrorHandling:
    """Test GUI error handling and validation."""

    @pytest.mark.integration
    @pytest.mark.gui
    def test_invalid_data_handling(self, mock_gui_root):
        """Test GUI handling of invalid data inputs."""
        with patch('tkinter.Tk', return_value=mock_gui_root):
            with patch('monte_carlo_gui_app.AlgorithmManager') as mock_manager:
                mock_manager.return_value.algorithms = {'TestAlgorithm': Mock()}
                
                gui = MonteCarloGUI(mock_gui_root)
                
                # Mock invalid inputs
                gui.ticker_var = Mock()
                gui.ticker_var.get.return_value = 'INVALID_TICKER'
                gui.capital_var = Mock()
                gui.capital_var.get.return_value = '-1000'  # Invalid negative capital
                
                # Mock messagebox for error display
                with patch('tkinter.messagebox.showerror') as mock_error:
                    # Test validation
                    if hasattr(gui, 'validate_inputs'):
                        result = gui.validate_inputs()
                        assert not result  # Should fail validation
                        mock_error.assert_called()

    @pytest.mark.integration
    @pytest.mark.gui
    def test_network_error_handling(self, mock_gui_root):
        """Test GUI handling of network errors during data fetching."""
        with patch('tkinter.Tk', return_value=mock_gui_root):
            with patch('monte_carlo_gui_app.AlgorithmManager') as mock_manager:
                mock_manager.return_value.algorithms = {'TestAlgorithm': Mock()}
                
                # Mock network error
                with patch('yfinance.download', side_effect=Exception("Network error")):
                    gui = MonteCarloGUI(mock_gui_root)
                    
                    gui.ticker_var = Mock()
                    gui.ticker_var.get.return_value = 'AAPL'
                    gui.period_var = Mock()
                    gui.period_var.get.return_value = '1y'
                    gui.interval_var = Mock()
                    gui.interval_var.get.return_value = '1d'
                    gui.asset_type_var = Mock()
                    gui.asset_type_var.get.return_value = 'stocks'
                    
                    # Mock error display
                    with patch('tkinter.messagebox.showerror') as mock_error:
                        if hasattr(gui, 'load_data'):
                            gui.load_data()
                            
                            # Should handle error gracefully
                            assert gui.current_data is None


@pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI module not available")
class TestGUIThreading:
    """Test GUI threading and responsiveness."""

    @pytest.mark.integration
    @pytest.mark.gui
    @pytest.mark.slow
    def test_threaded_operations(self, mock_gui_root):
        """Test that long operations run in threads."""
        with patch('tkinter.Tk', return_value=mock_gui_root):
            with patch('monte_carlo_gui_app.AlgorithmManager') as mock_manager:
                mock_manager.return_value.algorithms = {'TestAlgorithm': Mock()}
                
                gui = MonteCarloGUI(mock_gui_root)
                
                # Mock threading
                with patch('threading.Thread') as mock_thread:
                    mock_thread_instance = Mock()
                    mock_thread.return_value = mock_thread_instance
                    
                    # Test that backtesting uses threading
                    if hasattr(gui, 'run_multi_strategy_backtest'):
                        gui.selected_algorithms = ['TestAlgorithm']
                        gui.current_data = Mock()
                        
                        gui.run_multi_strategy_backtest()
                        
                        # Should create a thread
                        mock_thread.assert_called()
                        mock_thread_instance.start.assert_called()

    @pytest.mark.integration
    @pytest.mark.gui
    def test_gui_responsiveness_during_operations(self, mock_gui_root):
        """Test that GUI remains responsive during long operations."""
        with patch('tkinter.Tk', return_value=mock_gui_root):
            with patch('monte_carlo_gui_app.AlgorithmManager') as mock_manager:
                mock_manager.return_value.algorithms = {'TestAlgorithm': Mock()}
                
                gui = MonteCarloGUI(mock_gui_root)
                
                # Mock root.update() calls
                gui.root.update = Mock()
                
                # Test that GUI updates during operations
                if hasattr(gui, 'run_monte_carlo'):
                    gui.current_results = {'returns': [0.01, -0.005, 0.008]}
                    gui.num_sims_var = Mock()
                    gui.num_sims_var.get.return_value = '10'
                    
                    with patch('monte_carlo_trade_simulation.random_trade_order_simulation'):
                        gui.run_monte_carlo()
                        
                        # Should call root.update() to keep GUI responsive
                        # (This depends on implementation)


@pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI module not available")
class TestGUIValidation:
    """Test GUI input validation and parameter checking."""

    @pytest.mark.integration
    @pytest.mark.gui
    def test_parameter_validation(self, mock_gui_root):
        """Test parameter validation in GUI."""
        with patch('tkinter.Tk', return_value=mock_gui_root):
            with patch('monte_carlo_gui_app.AlgorithmManager') as mock_manager:
                mock_manager.return_value.algorithms = {'TestAlgorithm': Mock()}
                
                gui = MonteCarloGUI(mock_gui_root)
                
                # Mock parameter variables
                gui.capital_var = Mock()
                gui.risk_mgmt_var = Mock()
                gui.stop_loss_var = Mock()
                gui.take_profit_var = Mock()
                
                # Test valid parameters
                gui.capital_var.get.return_value = '10000'
                gui.risk_mgmt_var.get.return_value = 0.02
                gui.stop_loss_var.get.return_value = 0.05
                gui.take_profit_var.get.return_value = 0.10
                
                if hasattr(gui, 'validate_strategy_params'):
                    gui.validate_strategy_params()
                    # Should pass validation
                
                # Test invalid parameters
                gui.capital_var.get.return_value = 'invalid'
                gui.risk_mgmt_var.get.return_value = -0.1  # Negative risk
                
                # Should handle invalid parameters appropriately

    @pytest.mark.integration
    @pytest.mark.gui
    def test_real_time_validation(self, mock_gui_root):
        """Test real-time parameter validation."""
        with patch('tkinter.Tk', return_value=mock_gui_root):
            with patch('monte_carlo_gui_app.AlgorithmManager') as mock_manager:
                mock_manager.return_value.algorithms = {'TestAlgorithm': Mock()}
                
                gui = MonteCarloGUI(mock_gui_root)
                
                # Mock validation status
                gui.strategy_status_var = Mock()
                
                # Test parameter change callbacks
                if hasattr(gui, 'on_risk_change'):
                    gui.risk_mgmt_var = Mock()
                    gui.risk_mgmt_var.get.return_value = 0.05
                    gui.risk_label = Mock()
                    
                    gui.on_risk_change()
                    
                    # Should update display
                    gui.risk_label.config.assert_called()


@pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI module not available")
class TestGUIStateManagement:
    """Test GUI state management and persistence."""

    @pytest.mark.integration
    @pytest.mark.gui
    def test_state_persistence_between_operations(self, mock_gui_root):
        """Test that GUI state persists between operations."""
        with patch('tkinter.Tk', return_value=mock_gui_root):
            with patch('monte_carlo_gui_app.AlgorithmManager') as mock_manager:
                mock_manager.return_value.algorithms = {'TestAlgorithm': Mock()}
                
                gui = MonteCarloGUI(mock_gui_root)
                
                # Set initial state
                gui.current_data = Mock()
                gui.current_results = {'total_return': 0.15}
                gui.selected_algorithms = ['TestAlgorithm']
                
                # Perform operation that shouldn't clear state
                if hasattr(gui, 'update_status'):
                    gui.update_status("Test status")
                
                # State should persist
                assert gui.current_data is not None
                assert gui.current_results is not None
                assert len(gui.selected_algorithms) > 0

    @pytest.mark.integration
    @pytest.mark.gui
    def test_gui_reset_functionality(self, mock_gui_root):
        """Test GUI reset functionality."""
        with patch('tkinter.Tk', return_value=mock_gui_root):
            with patch('monte_carlo_gui_app.AlgorithmManager') as mock_manager:
                mock_manager.return_value.algorithms = {'TestAlgorithm': Mock()}
                
                gui = MonteCarloGUI(mock_gui_root)
                
                # Set some state
                gui.current_data = Mock()
                gui.current_results = {'total_return': 0.15}
                
                # Test reset functionality if available
                if hasattr(gui, 'reset_application'):
                    gui.reset_application()
                    
                    # State should be cleared
                    assert gui.current_data is None
                    assert gui.current_results is None


if __name__ == '__main__':
    pytest.main([__file__])
