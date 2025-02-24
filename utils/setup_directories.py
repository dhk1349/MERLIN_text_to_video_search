import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass

# Add the project root to Python path when running as script
if __name__ == "__main__":
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.append(project_root)

from utils.logger import logger

@dataclass
class ProjectStructure:
    """Defines the project directory structure."""
    data: Dict[str, Dict[str, Dict[str, dict]]]
    outputs: Dict[str, Dict[str, dict]]
    configs: Dict[str, dict]

    @classmethod
    def get_default_structure(cls) -> 'ProjectStructure':
        """Returns the default project structure."""
        dataset_structure = {
            "videos": {},
            "gpt4v_captions": {},
            "video_embeddings": {},
            "text_embeddings": {}
        }
        
        return cls(
            data={
                "ActivityNet": dataset_structure.copy(),
                "MSRVTT-1k": dataset_structure.copy(),
                "MSVD": dataset_structure.copy()
            },
            outputs={
                "embeddings": {},
                "logs": {},
                "results": {}
            },
            configs={}
        )

def create_directory(path: Path, debug: bool = False) -> None:
    """
    Creates a directory if it doesn't exist.
    
    Args:
        path: Path to create
        debug: Whether to log debug information
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        if debug:
            logger.debug(f"Created directory: {path}")
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {str(e)}")
        raise

def create_directory_structure(
    base_path: Optional[Union[str, Path]] = None,
    structure: Optional[ProjectStructure] = None,
    debug: bool = False
) -> None:
    """
    Creates the directory structure for the MERLIN project.
    
    Args:
        base_path: Base directory path where all data will be stored
        structure: Custom project structure (uses default if None)
        debug: Whether to log debug information
        
    Raises:
        ValueError: If the structure is invalid
        OSError: If directory creation fails
    """
    if structure is None:
        structure = ProjectStructure.get_default_structure()
    
    if base_path is None:
        base_path = Path.cwd()
    else:
        base_path = Path(base_path)
    
    def create_recursive(struct: Dict[str, Any], current_path: Path) -> None:
        """Recursively creates directory structure."""
        if not isinstance(struct, dict):
            return
            
        for key, value in struct.items():
            path = current_path / key
            create_directory(path, debug)
            create_recursive(value, path)
    
    try:
        # Create main structure
        create_recursive(structure.data, base_path / "data")
        create_recursive(structure.outputs, base_path / "outputs")
        create_recursive(structure.configs, base_path / "configs")
        
        # Create necessary files
        config_path = base_path / "configs" / "default_config.yaml"
        if not config_path.exists():
            config_path.touch()
            if debug:
                logger.debug(f"Created config file: {config_path}")
        
        logger.info("Successfully created directory structure")
        
    except Exception as e:
        logger.error(f"Error creating directory structure: {str(e)}")
        raise

def verify_structure(base_path: Optional[Union[str, Path]] = None) -> bool:
    """
    Verifies that the required directory structure exists.
    
    Args:
        base_path: Base directory path to verify
        
    Returns:
        bool: True if structure is valid, False otherwise
    """
    if base_path is None:
        base_path = Path.cwd()
    else:
        base_path = Path(base_path)
    
    structure = ProjectStructure.get_default_structure()
    
    def verify_recursive(struct: Dict[str, Any], current_path: Path) -> bool:
        if not isinstance(struct, dict):
            return True
            
        if not current_path.exists():
            logger.warning(f"Missing directory: {current_path}")
            return False
            
        result = True
        for key, value in struct.items():
            path = current_path / key
            if not verify_recursive(value, path):
                result = False
                
        return result
    
    data_valid = verify_recursive(structure.data, base_path / "data")
    outputs_valid = verify_recursive(structure.outputs, base_path / "outputs")
    configs_valid = verify_recursive(structure.configs, base_path / "configs")
    
    return all([data_valid, outputs_valid, configs_valid])

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create MERLIN project directory structure")
    parser.add_argument("--path", type=str, help="Base path for project structure")
    parser.add_argument("--verify", action="store_true", help="Verify existing structure")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    if args.verify:
        if verify_structure(args.path):
            logger.info("Directory structure is valid")
            sys.exit(0)
        else:
            logger.error("Directory structure is invalid")
            sys.exit(1)
    else:
        create_directory_structure(args.path, debug=args.debug) 