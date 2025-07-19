import pytest
from utilsforecast.data import generate_series

from timecopilot.agent import TimeCopilot


def test_agent_max_length_parameter():
    """Test that TimeCopilot agent accepts and uses max_length parameter."""
    # Create a TimeCopilot agent with max_length
    agent = TimeCopilot(llm="test", max_length=50)
    assert agent.max_length == 50
    
    # Test default (None)
    agent_default = TimeCopilot(llm="test")
    assert agent_default.max_length is None


def test_agent_forecast_method_max_length():
    """Test that forecast method accepts max_length parameter."""
    # Create test data
    df = generate_series(n_series=1, freq="D", min_length=100, max_length=100)
    
    # This is a unit test, so we just verify the parameter is passed through
    # We can't test the full forecast without LLM access
    agent = TimeCopilot(llm="test")
    
    # Test that the method accepts the parameter
    try:
        # This will fail during actual execution due to LLM, but we can verify
        # the parameter is accepted
        agent.forecast(df, max_length=30)
    except Exception as e:
        # We expect this to fail due to LLM issues, but the parameter should be accepted
        assert "max_length" not in str(e)  # Parameter error would mention max_length


def test_agent_max_length_override():
    """Test that forecast method can override instance max_length."""
    # Create agent with default max_length
    agent = TimeCopilot(llm="test", max_length=100)
    assert agent.max_length == 100
    
    # Generate test data
    df = generate_series(n_series=1, freq="D", min_length=50, max_length=50)
    
    # Test that calling forecast with max_length parameter overrides the instance setting
    original_max_length = agent.max_length
    
    try:
        # This will fail during execution, but we can verify the override works
        agent.forecast(df, max_length=30)
    except Exception:
        # The forecast will fail due to LLM, but the max_length should be updated
        assert agent.max_length == 30  # Should be overridden
    
    # Test that None value doesn't override if instance has a value
    agent.max_length = original_max_length  # Reset
    try:
        agent.forecast(df, max_length=None)
    except Exception:
        # max_length should remain at original value since None was passed
        assert agent.max_length == original_max_length