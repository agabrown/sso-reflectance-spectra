"""
From the Gaia DR3 reflectance spectra data create an SSO "object" table which only contains one line per SSO with the following fields preserved: "source_id", "number_mp", "denomination", "num_of_spectra".

This file can be combined in topcat with the MP3C data on the asteroids and then used for further analysis of the classified asteroids.

Anthony Brown Nov 2023 - Nov 2023
"""
import argparse

from reflspectratools import load_dr3_data
from astropy.table import Table


def create_table(args):
    """
    Create the table from the Gaia DR3 reflectance spectra data.

    Parameters
    ----------

    args : dict
        Command line arguments.

    Returns
    -------

    Nothing
    """
    ssodata = load_dr3_data(clipspectra=False)

    sso_objects = (
        ssodata.groupby(by="number_mp", as_index=False)
        .head(1)
        .reset_index()
        .filter(["source_id", "number_mp", "denomination", "num_of_spectra"])
    )

    Table.from_pandas(sso_objects).write(
        "./data/DR3ReflectanceSpectraObjects.fits", format="fits"
    )


def parseCommandLineArguments():
    """
    Set up command line parsing.
    """
    parser = argparse.ArgumentParser(
        description="""Create SSO object file from Gaia DR3 reflectance spectra data."""
    )
    cmdargs = vars(parser.parse_args())
    return cmdargs


if __name__ in ("__main__"):
    create_table(parseCommandLineArguments())
