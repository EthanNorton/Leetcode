"""
Utility functions for the Substack Analyzer
"""

import logging
from pathlib import Path
import json
from typing import Dict, Any


def setup_logging(output_dir: Path) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        output_dir: Directory for log files
        
    Returns:
        Configured logger
    """
    log_file = output_dir / 'analyzer.log'
    
    # Create logger
    logger = logging.getLogger('SubstackAnalyzer')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def load_config(config_path: str = 'config.json') -> Dict[str, Any]:
    """
    Load configuration from JSON file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Return default config
        return {
            'max_articles': 50,
            'num_predictions': 5,
            'num_books': 10,
            'output_dir': './output'
        }


def save_config(config: Dict[str, Any], config_path: str = 'config.json'):
    """
    Save configuration to JSON file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save config file
    """
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
