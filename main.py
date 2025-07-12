import os
import argparse
from config import Config
from trainer import main as run_training
from utils import setup_logging

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for configuration."""
    parser = argparse.ArgumentParser(description='NeuroPathNet Training Script')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--log_dir', type=str, default='./logs/', help='Directory for logging')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load configuration
    config = Config().get_config()  # Assuming Config class handles loading from YAML or similar
    
    # Set up logging
    logger = setup_logging(args.log_dir)
    logger.info("Starting NeuroPathNet training...")
    
    # Run training
    run_training(config)

if __name__ == "__main__":
    main()

