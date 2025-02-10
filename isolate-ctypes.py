"""
Use the Multi-layer Perceptron classifier to separate C-type asteroids from the overall set of asteroids with relfectance spectra.

Anthony Brown Dec 2023 - Dec 2023
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse

from sklearn.neural_network import MLPClassifier

from astropy.table import Table

from tqdm import tqdm

from agabpylib.plotting.plotstyles import useagab, apply_tufte
from agabpylib.stats.robuststats import rse
from reflspectratools import (
    load_dr3_data,
    create_ml_training_classes,
    extract_spectra,
    prep_classifier_inputs,
)


def separate_asteroids(args):
    """
    Carry out all the steps to do the MLP classification of C-type vs. other.

    Parameters
    ----------

    args : dict
        Command line arguments

    Returns
    -------

    Nothing
    """
    ssospectra = load_dr3_data(clipspectra=False)
    n_asteroids = ssospectra.number_mp.unique().size
    wavelengths = ssospectra["wavelength"].unique()
    wavelengths.sort()
    ssotypes = create_ml_training_classes(ssospectra, filter=True)

    ssoparams = Table.read("./data/DR3ReflectanceSpectraMP3C.fits").to_pandas()
    ssoparams["denomination"] = ssoparams.denomination.str.decode("utf-8")
    ssoparams["parent_name"] = ssoparams.parent_name.str.decode("utf-8")

    training_spectra, training_labels, data_to_be_classified = prep_classifier_inputs(
        ssospectra, ssotypes, rng=None
    )
    for i in range(args["naugment"]):
        ts, tl, dtbc = prep_classifier_inputs(
            ssospectra, ssotypes, rng=np.random.default_rng(i)
        )
        training_spectra = np.vstack([training_spectra, ts])
        training_labels = np.concatenate([training_labels, tl])
        data_to_be_classified = np.vstack([data_to_be_classified, dtbc])

    probability_matrix = np.zeros((args["nmlpc"], (1 + args["naugment"]) * n_asteroids))

    for j in tqdm(range(args["nmlpc"])):
        clf = MLPClassifier(
            solver="adam",
            alpha=1e-05,
            batch_size=100,
            hidden_layer_sizes=(200, 200, 200, 200),
            max_iter=200,
            random_state=j,
            early_stopping=True,
            n_iter_no_change=25,
        )
        clf.fit(training_spectra, training_labels)

        ctype_index = np.argwhere(clf.classes_ == 1).ravel()[0]
        probability_matrix[j, :] = clf.predict_proba(data_to_be_classified)[
            :, ctype_index
        ]

    ssoparams["ctype_median_prob"] = np.median(
        probability_matrix.reshape(args["nmlpc"] * (1 + args["naugment"]), n_asteroids),
        axis=0,
    )
    ssoparams["ctype_rse_prob"] = rse(
        probability_matrix.reshape(args["nmlpc"] * (1 + args["naugment"]), n_asteroids),
        ax=0,
    )
    Table.from_pandas(ssoparams).write(
        "./outputs/DR3ReflSpecCvsOtherType.fits", format="fits", overwrite=True
    )


def parseCommandLineArguments():
    """
    Set up command line parsing.
    """
    parser = argparse.ArgumentParser(
        description="""Separate C-type asteroids from the rest using an MLP classifier"""
    )
    parser.add_argument(
        "--naugment",
        dest="naugment",
        type=int,
        default=10,
        help="""Add naugment copies of each spectrum, by varying reflectances within their uncertainties (assumed to be Gaussian) """,
    )
    parser.add_argument(
        "--nmlpc",
        dest="nmlpc",
        type=int,
        default=10,
        help="""Run the MLP classifier nmlpc times with a different random state each time.""",
    )
    cmdargs = vars(parser.parse_args())
    return cmdargs


if __name__ in ("__main__"):
    separate_asteroids(parseCommandLineArguments())
