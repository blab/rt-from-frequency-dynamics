import pandas as pd
from .datahelpers import prep_dates, prep_cases, prep_sequence_counts


class CaseData:
    def __init__(self, raw_cases):
        self.dates, self.raw_cases = prep_cases(raw_cases)


class VariantData:
    def __init__(self, raw_cases, raw_seqs):
        self.dates, date_to_index = prep_dates(
            pd.concat((raw_cases.date, raw_seqs.date))
        )
        self.cases = prep_cases(raw_cases, date_to_index=date_to_index)
        self.seq_names, self.seq_counts = prep_sequence_counts(
            raw_seqs, date_to_index=date_to_index
        )

    def make_numpyro_input(self):
        data = dict()
        data["cases"] = self.cases
        data["seq_counts"] = self.seq_counts
        data["N"] = self.seq_counts.sum(axis=1)
        data["seq_names"] = self.seq_names
        return data
