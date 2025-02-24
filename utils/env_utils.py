import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, Union

from .logger import logger

def load_env_variables(env_path: Optional[Union[str, Path]] = None) -> None:
    """
    Load environment variables from .env file.
    
    Args:
        env_path: Path to .env file. If None, looks for .env in project root
    """
    if env_path is None:
        env_path = Path(__file__).parent.parent / '.env'
    else:
        env_path = Path(env_path)
    
    if not env_path.exists():
        logger.warning(f".env file not found at {env_path}")
        return
        
    load_dotenv(env_path)
    logger.debug("Loaded environment variables from .env file")

def get_required_env(key: str) -> str:
    """
    Get a required environment variable.
    
    Args:
        key: Environment variable key
        
    Returns:
        str: Environment variable value
        
    Raises:
        ValueError: If environment variable is not set
    """
    value = os.getenv(key)
    if value is None:
        raise ValueError(
            f"Required environment variable '{key}' is not set. "
            f"Please set it in your .env file or environment."
        )
    return value

def get_optional_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get an optional environment variable.
    
    Args:
        key: Environment variable key
        default: Default value if not set
        
    Returns:
        Optional[str]: Environment variable value or default
    """
    return os.getenv(key, default) 