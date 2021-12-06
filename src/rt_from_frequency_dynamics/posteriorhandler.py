import arviz

class PosteriorHandler():
    def __init__(self, dataset=None, LD=None, name=None):
        self.dataset = dataset
        self.LD = LD
        self.name = name

    def save_posterior(self, filename):
        if self.dataset is not None:
            self.dataset.to_json(filename)

    def load_posterior(self, filename):
        self.dataset = arviz.from_json(filename)

    def unpack_posterior(self):
        return self.dataset, self.LD

class MultiPosterior():
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
