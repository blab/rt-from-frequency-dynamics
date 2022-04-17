import json


class PosteriorHandler:
    def __init__(self, dataset=None, data=None, name=None):
        self.dataset = dataset
        self.data = data
        self.name = name

    def save_posterior(self, filepath):
        if self.dataset is not None:
            with open(filepath, "w") as file:
                json.dump(self.dataset, file)

    def load_posterior(self, filepath):
        with open(filepath, "w") as file:
            self.dataset = json.load(file)

    def unpack_posterior(self):
        return self.dataset, self.data


class MultiPosterior:
    def __init__(self, posteriors=None, posterior=None):
        self.locator = dict()
        if posterior is not None:
            self.add_posterior(posterior)
        if posteriors is not None:
            self.add_posteriors(posteriors)

    def add_posterior(self, posterior):
        self.locator[posterior.name] = posterior

    def add_posteriors(self, posteriors):
        for p in posteriors:
            self.add_posterior(p)

    def get(self, name):
        return self.locator[name]
