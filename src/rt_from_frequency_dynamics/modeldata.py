from .datahelpers import prep_cases, prep_sequence_counts, prep_vaccination
from .modelhelpers import make_breakpoint_splines
# Abstract class ModelData


# These might get turned into data classes
class AggregateData():
    def __init__(self, raw_cases):
        self.dates, self.raw_cases = prep_cases(raw_cases)


class LineageData():
    def __init__(self, raw_cases, raw_seqs):
        self.seq_names, dates_s, seq_counts = prep_sequence_counts(raw_seqs)
        dates_c, cases = prep_cases(raw_cases)

        idx_retain_s = [i for i,d in enumerate(dates_s) if d in dates_c] 
        idx_retain_c = [i for i,d in enumerate(dates_c) if d in dates_s]

        self.dates = [dates_s[i] for i in idx_retain_s]
        self.seq_counts = seq_counts[idx_retain_s, :]
        self.cases = cases[idx_retain_c]

    def make_numpyro_input(self, k=20): # Eventually want to seperate this k out into r_model
        data = dict()
        data["cases"] = self.cases
        data["seq_counts"] = self.seq_counts
        data["N"] = self.seq_counts.sum(axis=1)
        data["X"] = make_breakpoint_splines(self.cases, k)
        return data

class LineageVaccinationData():
    def __init__(self, raw_cases, raw_seqs, raw_vacc):
        self.seq_names, dates_s, seq_counts = prep_sequence_counts(raw_seqs)
        dates_c, cases = prep_cases(raw_cases)
        dates_v, vacc = prep_vaccination(raw_vacc)

        idx_retain_s = [i for i,d in enumerate(dates_s) if d in dates_c] 
        idx_retain_c = [i for i,d in enumerate(dates_c) if d in dates_s]

        self.dates = [dates_s[i] for i in idx_retain_s]

        self.seq_counts = seq_counts[idx_retain_s, :]
        self.cases = cases[idx_retain_c]
        
        idx_retain_v = [i for i,d in enumerate(dates_v) if d in self.dates]
        self.vacc = vacc[idx_retain_v]

    def make_numpyro_input(self, k=20):
        data = dict()
        data["cases"] = self.cases
        data["seq_counts"] = self.seq_counts
        data["N"] = self.seq_counts.sum(axis=1)
        data["vacc"] = self.vacc
        data["X"] = make_breakpoint_splines(self.cases, k)
        return data
