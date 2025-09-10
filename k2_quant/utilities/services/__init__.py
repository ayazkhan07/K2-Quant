"""K2 Quant Services"""

from .polygon_client import polygon_client
from .stock_data_service import stock_service

# Import new services conditionally to avoid errors if dependencies missing
try:
    from .model_loader_service import model_loader_service
except ImportError as e:
    print(f"Warning: Could not import model_loader_service: {e}")
    model_loader_service = None

try:
    from .technical_analysis_service import ta_service
except ImportError as e:
    print(f"Warning: Could not import ta_service: {e}")
    print("Install TA-Lib: pip install TA-Lib")
    ta_service = None

try:
    from .ai_chat_service import ai_chat_service
except ImportError as e:
    print(f"Warning: Could not import ai_chat_service: {e}")
    print("Install AI dependencies: pip install openai anthropic")
    ai_chat_service = None

try:
    from .dynamic_python_engine import dpe_service
except ImportError as e:
    print(f"Warning: Could not import dpe_service: {e}")
    dpe_service = None

try:
    from .strategy_service import strategy_service
except ImportError as e:
    print(f"Warning: Could not import strategy_service: {e}")
    strategy_service = None

__all__ = [
    'polygon_client',
    'stock_service',
    'model_loader_service',
    'ta_service',
    'ai_chat_service',
    'dpe_service',
    'strategy_service'
]

# Guarded export for Ollama integration
try:
    from .ollama_service import OllamaService, OllamaConfig
except Exception as e:
    print(f"Warning: Could not import OllamaService: {e}")
    OllamaService = None  # type: ignore
    OllamaConfig = None   # type: ignore
