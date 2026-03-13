"""
K2 Quant API Configuration Management (self-contained)
"""

import os
from typing import Dict, Optional
from pathlib import Path


class APIConfig:
    """Secure API configuration management"""

    def __init__(self):
        self.load_environment()

    def load_environment(self):
        """
        Load environment variables from an env file if it exists.

        Notes:
        - We support a few common locations to avoid relying on the current working directory.
        - Secrets should live in a local, non-committed file (see `.env.example`).
        """
        # Allow explicit override for enterprise deployments
        override = os.getenv("K2_ENV_PATH")
        candidates = []
        if override:
            candidates.append(Path(override))

        # Current working directory (common when running from repo root)
        candidates.append(Path.cwd() / ".env")

        # Repo root (robust when app is launched from another working directory)
        try:
            repo_root = Path(__file__).resolve().parents[3]
            candidates.append(repo_root / ".env")
        except Exception:
            pass

        # User home (optional)
        try:
            candidates.append(Path.home() / ".k2_quant.env")
        except Exception:
            pass

        env_path: Optional[Path] = None
        for p in candidates:
            if not p:
                continue
            try:
                # Normal case: `.env` is a file
                if p.is_file():
                    env_path = p
                    break
                # Some users accidentally create a `.env` directory with a `.env` file inside.
                # Support that layout as well: `.env/.env`.
                if p.is_dir():
                    nested = p / ".env"
                    if nested.is_file():
                        env_path = nested
                        break
            except Exception:
                continue

        if not env_path:
            return

        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        os.environ[key.strip()] = value.strip()
        except Exception:
            # Never crash the app due to env file parsing/permission issues.
            return

    @property
    def polygon_api_key(self) -> Optional[str]:
        return os.getenv('POLYGON_API_KEY')

    @property
    def alpha_vantage_api_key(self) -> Optional[str]:
        return os.getenv('ALPHA_VANTAGE_API_KEY')

    @property
    def openai_api_key(self) -> Optional[str]:
        return os.getenv('OPENAI_API_KEY')

    @property
    def anthropic_api_key(self) -> Optional[str]:
        return os.getenv('ANTHROPIC_API_KEY')

    @property
    def fred_api_key(self) -> Optional[str]:
        return os.getenv('FRED_API_KEY')

    @property
    def grok_api_key(self) -> Optional[str]:
        return os.getenv('GROK_API_KEY')

    @property
    def github_token(self) -> Optional[str]:
        return os.getenv('GITHUB_TOKEN')

    @property
    def tavily_api_key(self) -> Optional[str]:
        return os.getenv('TAVILY_API_KEY')

    def validate_keys(self) -> Dict[str, bool]:
        return {
            'polygon': bool(self.polygon_api_key),
            'alpha_vantage': bool(self.alpha_vantage_api_key),
            'openai': bool(self.openai_api_key),
            'anthropic': bool(self.anthropic_api_key),
            'fred': bool(self.fred_api_key),
            'grok': bool(self.grok_api_key),
            'github': bool(self.github_token),
            'tavily': bool(self.tavily_api_key),
        }

    def get_polygon_config(self) -> Dict[str, str]:
        return {
            'api_key': self.polygon_api_key,
            'base_url': 'https://api.polygon.io',
            'version': 'v2',
        }


api_config = APIConfig()



