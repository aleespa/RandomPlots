import toml

from tools.technology import create_directory, clear_folder


class Settings:
    filename: str

    def __init__(self):
        config = toml.load('config.toml')
        self.filename = config['file_to_run']
        create_directory(f"outputs/{self.filename}")
        clear_folder(f"outputs/{self.filename}")
