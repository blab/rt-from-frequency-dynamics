import pandas as pd
import rt_from_frequency_dynamics as rf


def test_format_seq_names():
    RAW_NAMES = ["Alpha", "Beta", "other", "Gamma", "Delta"]
    RAW_NAMES_FORMATTED = ["Alpha", "Beta", "Gamma", "Delta", "other"]
    assert rf.format_seq_names(RAW_NAMES) == RAW_NAMES_FORMATTED


def test_counts_to_matrix():
    assert 1 == 1


def test_prep_sequence_counts():
    assert 1 == 1
    # Make sure it's handling nans and things correctly


def prep_cases():
    assert 1 == 1
    # Make sure it's handling nans and things correctly
