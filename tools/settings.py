import toml


class Settings:
    file_to_run: str

    def __init__(self):
        config = toml.load('config.toml')
        self.file_to_run = config['file_to_run']
