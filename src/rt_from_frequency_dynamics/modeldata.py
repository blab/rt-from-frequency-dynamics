import pandas as pd
from .datahelpers import prep_dates, prep_cases, prep_sequence_counts
# Abstract class ModelData


# These might get turned into data classes
class CaseData():
    def __init__(self, raw_cases):
        self.dates, self.raw_cases = prep_cases(raw_cases)


class VariantData():
    def __init__(self, raw_cases, raw_seqs):
        self.dates, date_to_index = prep_dates(pd.concat((raw_cases.date, raw_seqs.date)))
        self.cases = prep_cases(raw_cases, date_to_index=date_to_index)
        self.seq_names, self.seq_counts = prep_sequence_counts(raw_seqs, date_to_index=date_to_index)

    def make_numpyro_input(self):
        data = dict()
        data["cases"] = self.cases
        data["seq_counts"] = self.seq_counts
        data["N"] = self.seq_counts.sum(axis=1)
        return data

# class LineageVaccinationData():
#     def __init__(self, raw_cases, raw_seqs, raw_vacc):
#         self.seq_names, dates_s, seq_counts = prep_sequence_counts(raw_seqs)
#         dates_c, cases = prep_cases(raw_cases)
#         dates_v, vacc = prep_vaccination(raw_vacc)

#         idx_retain_s = [i for i,d in enumerate(dates_s) if d in dates_c] 
#         idx_retain_c = [i for i,d in enumerate(dates_c) if d in dates_s]

#         self.dates = [dates_s[i] for i in idx_retain_s]

#         self.seq_counts = seq_counts[idx_retain_s, :]
#         self.cases = cases[idx_retain_c]
        
#         idx_retain_v = [i for i,d in enumerate(dates_v) if d in self.dates]
#         self.vacc = vacc[idx_retain_v]

#     def make_numpyro_input(self, k=20):
#         data = dict()
#         data["cases"] = self.cases
#         data["seq_counts"] = self.seq_counts
#         data["N"] = self.seq_counts.sum(axis=1)
#         data["vacc"] = self.vacc
#         data["X"] = make_breakpoint_splines(self.cases, k)
#         return data
