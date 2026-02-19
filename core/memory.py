import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
import json


class FileReader:
    """Optimized file reader for Markdown files."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize FileReader with config path."""
        # Resolve config path relative to this script's directory
        if not Path(config_path).is_absolute():
            config_path = Path(__file__).parent / config_path
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.target_dir = Path(self.config.get("SAVE_DIRECTORY", ""))

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, "r") as file:
                return yaml.safe_load(file) or {}
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML config: {e}")

    def list_files(self) -> list[Path]:
        """List all .md files in the target directory."""
        if not self.target_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.target_dir}")

        if not self.target_dir.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {self.target_dir}")

        # Get all .md files sorted by name
        files = sorted(
            [
                f
                for f in self.target_dir.iterdir()
                if f.is_file() and f.suffix.lower() == ".md"
            ]
        )
        return files

    def display_files(self) -> Optional[Path]:
        """Display files and get user selection."""
        files = self.list_files()

        if not files:
            print(f"No files found in {self.target_dir}")
            return None

        print(f"\nMarkdown files in {self.target_dir}:\n")
        for idx, file in enumerate(files, 1):
            file_size = file.stat().st_size
            print(f"  {idx}. {file.name} ({file_size} bytes)")

        while True:
            try:
                choice = input(f"\nSelect file (1-{len(files)}): ").strip()
                idx = int(choice) - 1

                if 0 <= idx < len(files):
                    return files[idx]
                else:
                    print(
                        f"Invalid selection. Please choose between 1 and {len(files)}"
                    )
            except ValueError:
                print("Invalid input. Please enter a number.")

    def read_file(self, file_path: Path) -> str:
        """Read file content with error handling."""
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        except UnicodeDecodeError:
            # Fallback to binary read if text read fails
            with open(file_path, "rb") as file:
                return file.read().decode("utf-8", errors="replace")

    def parse_data(self, content: str) -> str:
        """Return markdown content as-is."""
        return content

    def run(self) -> Optional[Any]:
        """Main execution flow: list files, select, read, and parse."""
        print(f"Reading from directory: {self.target_dir}\n")

        selected_file = self.display_files()
        if not selected_file:
            return None

        print(f"\nReading {selected_file.name}...\n")

        try:
            content = self.read_file(selected_file)
            data = self.parse_data(content)
            return data
        except Exception as e:
            print(f"Error reading file: {e}")
            return None


# Main execution
if __name__ == "__main__":
    reader = FileReader(config_path="config.yaml")
    data = reader.run()

    if data:
        print("\nData retrieved successfully!\n")
        print(data)
