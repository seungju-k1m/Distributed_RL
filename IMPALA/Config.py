from baseline.utils import jsonParser


class IMPALAConfig:
    def __init__(self, path):
        parser = jsonParser(path)
        self.data = parser.loadParser()

        for key, value in self.data.items():
            setattr(self, key, value)
