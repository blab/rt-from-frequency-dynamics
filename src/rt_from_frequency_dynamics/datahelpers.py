import datetime
import numpy as np
import pandas as pd


def prep_dates(raw_data: pd.DataFrame):
    raw_data['date'] = pd.to_datetime(raw_data['date'])
    dmn = raw_data['date'].min()
    dmx = raw_data['date'].max()
    dates = [dmn+datetime.timedelta(days=d) for d in range(0, 1+(dmx-dmn).days)]
    date_to_index = {d: i for (i, d) in enumerate(dates)}
    return dates, date_to_index


def prep_cases(raw_cases: pd.DataFrame):  # List if dates, cases
    dates, date_to_index = prep_dates(raw_cases)

    C = np.zeros(len(dates))
    for index, row in raw_cases.iterrows():
        C[date_to_index[row.date]] += row.cases

    return dates, C


def format_seq_names(raw_names):
    if "other" in raw_names:
        names = []
        for s in raw_names:
            if s != "other":
                names.append(s)
        names.append("other")
    return names


def counts_to_matrix(raw_seqs, seq_names):
    dates, date_to_index = prep_dates(raw_seqs)
    C = np.zeros((len(dates), len(seq_names)))
    for i, s in enumerate(seq_names):
        for _, row in raw_seqs[raw_seqs.variant == s].iterrows():
            C[date_to_index[row.date], i] += row.sequences
    return dates, C


def prep_sequence_counts(raw_seqs: pd.DataFrame):
    raw_seq_names = pd.unique(raw_seqs.variant)
    raw_seq_names.sort()
    seq_names = format_seq_names(raw_seq_names)
    dates, C = counts_to_matrix(raw_seqs, seq_names)
    return seq_names, dates, C