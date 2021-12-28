"""
Prepare a data frame of case counts and variant frequencies.
"""
import argparse
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "Prepare data frames of case counts and variant frequencies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Define inputs.
    parser.add_argument(
        "--metadata",
        default="https://data.nextstrain.org/files/ncov/open/metadata.tsv.gz",
        help="path to SARS-CoV-2 metadata in Nextstrain format for the ncov workflow. Local and remote (HTTP, S3, etc.) paths are valid.",
    )
    parser.add_argument(
        "--cases",
        default="https://data.cdc.gov/api/views/9mfq-cb36/rows.csv?accessType=DOWNLOAD",
        help="path to CDC case counts by state and date.",
    )
    parser.add_argument(
        "--clades-to-analyze",
        default="variants-us_data-prep.clades.tsv",
        help="tab-delimited mapping of Nextstrain clade ('clade') to common variant name ('variant'). All clades defined in this file will be included in the output."
    )
    parser.add_argument(
        "--states",
        default="variants-us_data-prep.states.tsv",
        help="tab-delimited mapping of full state/territory names ('name') to standard abbreviations ('abbreviation'). Used to map case counts to genomic counts.",
    )

    # Define parameters.
    parser.add_argument(
        "--start-date",
        default="2021-01-01",
        help="earliest date (inclusive) to consider genomic and case count records.",
    )
    parser.add_argument(
        "--end-date",
        default="2021-12-01",
        help="latest date (inclusive) to consider genomic and case count records.",
    )
    parser.add_argument(
        "--min-records-per-division",
        type=int,
        default=5000,
        help="minimum total genomic records required per division to include in output.",
    )
    parser.add_argument(
        "--metadata-chunk-size",
        type=int,
        default=100000,
        help="maximum metadata records to read into memory at once during initial pass. Increasing this value increases peak memory usage.",
    )

    # Define outputs.
    parser.add_argument(
        "--output-sequences",
        default="variants-us_location-variant-sequence-counts.tsv",
        help="tab-delimited genome counts per date, location, and variant.",
    )
    parser.add_argument(
        "--output-cases",
        default="variants-us_location-case-counts.tsv",
        help="tab-delimited case counts per date and location.",
    )

    args = parser.parse_args()

    # Load Nextstrain-curated metadata TSV.
    metadata_reader = pd.read_csv(
        args.metadata,
        sep="\t",
        usecols=(
            "strain",
            "date",
            "country",
            "division",
            "Nextstrain_clade"
        ),
        dtype={
            "country": "category",
            "division": "category",
            "Nextstrain_clade": "category",
        },
        chunksize=args.metadata_chunk_size,
    )

    # Iterate through metadata in chunks to control peak memory usage.
    metadata_chunks = []
    for metadata in metadata_reader:
        # Filter metadata.
        metadata = metadata.query("(country == 'USA') & (division != 'USA')")

        # Subset to recent samples, dropping records with ambiguous dates ("?" or "2021-06", etc.).
        unambiguous_dates = (metadata["date"] != "?") & (metadata["date"].str.count("-") == 2)
        metadata = metadata[unambiguous_dates].copy()

        # Convert date strings to date types for easier operations.
        metadata["date"] = pd.to_datetime(metadata["date"])

        # Subset by start and end date.
        date_since_start_date = (metadata["date"] >= args.start_date)
        date_before_end_date = (metadata["date"] <= args.end_date)
        metadata = metadata[(date_since_start_date) & (date_before_end_date)].copy()

        # Remove records without Nextstrain clades.
        metadata = metadata[~pd.isnull(metadata["Nextstrain_clade"])].copy()

        metadata_chunks.append(metadata)

    # Merge all chunks that passed all filters.
    metadata = pd.concat(
        metadata_chunks,
        ignore_index=True,
    )

    # Clade definitions. Map Nextstrain clade names (e.g., "20H (Beta, V2)") to
    # common variant names (e.g., "Beta").
    clades_df = pd.read_csv(
        args.clades_to_analyze,
        sep="\t",
    )
    clade_to_variant = dict(clades_df.loc[:, ["clade", "variant"]].values)

    # Map full names of US states to abbreviations and vice versa.
    states_df = pd.read_csv(
        args.states,
        sep="\t",
    )
    abbreviations_to_full_state_names = dict(states_df.loc[:, ["abbreviation", "name"]].values)

    # Determine variants to analyze and map Nextstrain clades to variant names.
    metadata["variant"] = metadata["Nextstrain_clade"].map(clade_to_variant).fillna("other").astype(str)

    # Keep states with at least a minimum number of metadata records.
    state_tallies = metadata["division"].value_counts()

    # In addition to getting states with enough records.
    states_to_analyze = sorted(state_tallies[(state_tallies > args.min_records_per_division)].index.tolist())

    # Filter data to records for states with enough sequences.
    metadata = metadata[metadata["division"].isin(states_to_analyze)].copy()

    # Export data frame of variant frequencies. Provision counts by date, state,
    # and variant.
    counts_by_date_state_variant = metadata.groupby(
        [
            "date",
            "division",
            "variant",
        ],
        observed=True,
        as_index=False,
    )["strain"].count().rename(
        columns={
            "strain": "sequences",
            "division": "location",
        }
    ).sort_values(["location", "variant", "date"])

    counts_by_date_state_variant.to_csv(
        args.output_sequences,
        sep="\t",
        index=False,
    )

    # State-level case data
    #
    # Download with https://data.cdc.gov/api/views/9mfq-cb36/rows.csv?accessType=DOWNLOAD
    cases = pd.read_csv(
        args.cases,
        parse_dates=["submission_date"],
        usecols=[
            "submission_date",
            "state",
            "new_case",
            "new_death",
        ],
    ).sort_values([
        "submission_date",
        "state",
    ])

    # Drop any records with missing "new case" or "new death" values.
    cases = cases[(~pd.isnull(cases["new_case"])) & (~pd.isnull(cases["new_death"]))].copy()

    # Export data frame of case counts.
    # Filter to cases between start and end date.
    cases = cases[(cases["submission_date"] >= args.start_date) & (cases["submission_date"] <= args.end_date)].copy()

    # Replace negative new case values with zeros.
    cases.loc[cases["new_case"] < 0, "new_case"] = 0

    # Annotate full names for states.
    cases["location"] = cases["state"].map(abbreviations_to_full_state_names)

    # Confirm that none of the states have missing valus (indicating missing
    # information in the abbreviation-to-name mapping).
    assert cases["location"].isnull().sum() == 0

    # Filter cases to states for analysis, based on genomic data above.
    cases = cases[cases["location"].isin(states_to_analyze)].reset_index()

    # Sum cases across all states and dates, accounting for states/divisions
    # with more than one abbreviation in the case data (e.g., "NYC" and "NY" for
    # "New York").
    total_cases = cases.groupby(
        [
            "location",
            "submission_date"
        ],
        observed=True,
        as_index=False,
    )["new_case"].sum().rename(columns={
        "submission_date": "date",
        "new_case": "case",
    })

    total_cases.to_csv(
        args.output_cases,
        sep="\t",
        index=False,
    )
