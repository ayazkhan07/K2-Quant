"""
K2 Quant API Configuration Management

Secure handling of API keys and configuration settings.
"""

import os
from typing import Dict, Optional
from pathlib import Path


class APIConfig:
    """Secure API configuration management"""
    
    def __init__(self):
        self.load_environment()
        
    def load_environment(self):
        """Load environment variables from .env file if it exists"""
        env_path = Path('.env')
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
    
    @property
    def polygon_api_key(self) -> Optional[str]:
        """Get Polygon.io API key"""
        return os.getenv('POLYGON_API_KEY')
    
    @property
    def alpha_vantage_api_key(self) -> Optional[str]:
        """Get Alpha Vantage API key (backup)"""
        return os.getenv('ALPHA_VANTAGE_API_KEY')
    
    @property
    def openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key"""
        return os.getenv('OPENAI_API_KEY')
    
    @property
    def anthropic_api_key(self) -> Optional[str]:
        """Get Anthropic API key"""
        return os.getenv('ANTHROPIC_API_KEY')
    
    @property
    def fred_api_key(self) -> Optional[str]:
        """Get FRED API key"""
        return os.getenv('FRED_API_KEY')
    
    @property
    def grok_api_key(self) -> Optional[str]:
        """Get Grok API key"""
        return os.getenv('GROK_API_KEY')
    
    @property
    def github_token(self) -> Optional[str]:
        """Get GitHub token"""
        return os.getenv('GITHUB_TOKEN')
    
    def validate_keys(self) -> Dict[str, bool]:
        """Validate that required API keys are present"""
        return {
            'polygon': bool(self.polygon_api_key),
            'alpha_vantage': bool(self.alpha_vantage_api_key),
            'openai': bool(self.openai_api_key),
            'anthropic': bool(self.anthropic_api_key),
            'fred': bool(self.fred_api_key),
            'grok': bool(self.grok_api_key),
            'github': bool(self.github_token)
        }
    
    def get_polygon_config(self) -> Dict[str, str]:
        """Get Polygon.io specific configuration"""
        return {
            'api_key': self.polygon_api_key,
            'base_url': 'https://api.polygon.io',
            'version': 'v2'
        }


# Global configuration instance
api_config = APIConfig()