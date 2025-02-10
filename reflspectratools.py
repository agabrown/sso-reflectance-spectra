"""
Useful functions for the exploration and analysis of the Gaia DR3 SSO reflectance spectra data.

Anthony Brown Nov 2023 - Nov 2023
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd

from scipy.interpolate import Akima1DInterpolator
from scipy.stats import scoreatpercentile

from astropy.table import Table
from astropy.visualization import ImageNormalize, LogStretch
import astropy.units as u
import astropy.constants as c

from agabpylib.plotting.plotstyles import useagab, apply_tufte

_ajup = 5.2026 * u.au
_pjup = np.sqrt(4 * np.pi**2 / (c.G * (c.M_sun + c.M_jup)) * _ajup**3).to(u.yr)
_resonances = {
    "5:1": 0,
    "4:1": 0,
    "3:1": 0,
    "5:2": 0,
    "7:3": 0,
    "2:1": 0,
    "3:2": 0,
    "1:1": 0,
}
for res in _resonances.keys():
    pvals = res.split(":")
    ratio = float(pvals[1]) / float(pvals[0])
    _resonances[res] = np.power(ratio, 2 / 3) * _ajup.value


def get_resonances():
    """
    Provides a dict with the semi-major axes of the orbits in resonance with Jupiter. The mapping is from for example "4:1" to the value of the semi-major axis.

    Parameters
    ----------
    None

    Returns
    -------
    resonances : dict
        Mapping from resonance name to semi-major axis (in au).
    """
    return _resonances


def name_to_mpcnum(ssodf, names):
    """
    Look up asteroid MPC numbers for the input list of names.

    Parameters
    ----------
    ssodf : Pandas DataFrame
        SSO reflectance spectra table in same format (columns/names) as in Gaia DR3 archive.
    names : list of str
        List of asteroid names

    Returns
    -------
    mpcnums : list of int
        List of MPC numbers
    """
    if isinstance(names, str):
        if np.any(ssodf["denomination"].str.fullmatch(names)):
            return ssodf.loc[
                ssodf["denomination"].str.fullmatch(names)
            ].number_mp.unique()[0]
        else:
            raise ValueError(f"Asteroid {names} not found")
    else:
        result = []
        for name in names:
            if np.any(ssodf["denomination"].str.fullmatch(name)):
                result.append(
                    ssodf.loc[
                        ssodf["denomination"].str.fullmatch(name)
                    ].number_mp.unique()[0]
                )
            else:
                raise ValueError(f"Asteroid {name} not found")
        return result


def mpcnum_to_name(ssodf, nums):
    """
    Look up asteroid names for the input list of numbers.

    Parameters
    ----------
    ssodf : Pandas DataFrame
        SSO reflectance spectra table in same format (columns/names) as in Gaia DR3 archive.
    nums : list of int
        List of asteroid numbers

    Returns
    -------
    names : list of str
        List of asteroid names
    """
    if isinstance(nums, int):
        return ssodf.loc[ssodf["number_mp"] == nums].denomination.unique()[0]
    else:
        return [
            ssodf.loc[ssodf["number_mp"] == n].denomination.unique()[0] for n in nums
        ]


def extract_spectra(ssodf, nums):
    """
    Extract spectra for the input list of MPC numbers from the Gaia DR3 data file.

    Parameters
    ----------
    ssodf : Pandas DataFrame
        SSO reflectance spectra table in same format (columns/names) as in Gaia DR3 archive.
    nums : list of int
        List of asteroid numbers

    Returns
    -------
    spectra : float array
        Array of spectra of shape (len(nums), number of wavelengths)
    """
    nwavs = ssodf.wavelength.unique().size
    y = ssodf.loc[np.isin(ssodf.number_mp, nums)].reflectance_spectrum.to_numpy()

    return y.reshape((int(y.size / nwavs), nwavs))


def prep_classifier_inputs(
    ssodf, ssotypes, classlabels=("C", "other"), rng=None, wrange=None, ssolist=None
):
    """
    Prepare the input for a given ML classifier. This consists of extracting the labeled spectra, creating the labels for the ML algorithm, and extracting all spectra to be classified.

    Parameters
    ----------
    ssodf : Pandas DataFrame
        SSO reflectance spectra table in same format (columns/names) as in Gaia DR3 archive.
    ssotypes : dict
        Mapping from classes to lists of corresponding asteroid MPC numbers
    classlabels : tuple
        Two two class labels to be mapped to [1,0] for the ML classifier
    rng : numpy.random.Generator
        If an instance of Generator is provided the spectra will be varied randomly with the noise on each value (assumed to be normally distributed around the observed reflectance value).
    wrange : tuple of ints
        Restrict the extract wavelength range to the input indices (using numpy slicing conventions)
    ssolist : list
        List of mpc numbers for the asteroid spectra to extract as the data to be classified.

    Returns
    -------
    training_spectra : array-like
        Array of training spectra
    training_labels : array-like
        Array of training labels
    data_to_be_classified : array-like
        Array of spectra to be classified
    """
    nwavs = ssodf.wavelength.unique().size
    if wrange is None:
        wrange = (0, nwavs)
    n_asteroids = ssodf.number_mp.unique().size
    ssodf_copy = ssodf.copy()

    if rng is not None:
        ssodf_copy.reflectance_spectrum = rng.normal(
            ssodf_copy.reflectance_spectrum, ssodf_copy.reflectance_spectrum_err
        )

    training_spectra = np.vstack(
        [
            extract_spectra(ssodf_copy, ssotypes[classlabels[0]]),
            extract_spectra(ssodf_copy, ssotypes[classlabels[1]]),
        ]
    )[:, wrange[0] : wrange[1]]

    if ssolist is not None:
        data_to_be_classified = extract_spectra(ssodf_copy, ssolist)[
            :, wrange[0] : wrange[1]
        ]
    else:
        data_to_be_classified = np.reshape(
            ssodf_copy.reflectance_spectrum.to_numpy(), (n_asteroids, nwavs)
        )[:, wrange[0] : wrange[1]]

    data_to_be_classified[np.isnan(data_to_be_classified)] = 0

    training_labels = np.concatenate(
        [
            np.ones(len(ssotypes[classlabels[0]])),
            np.zeros(len(ssotypes[classlabels[1]])),
        ]
    )

    return training_spectra, training_labels, data_to_be_classified


def overplot_spectrum(
    ssodf, ax, mpcnumbers, interpolate=True, addlegend=False, **kwargs
):
    """
    Overplot an SSO reflectance spectrum on an existing matplotlib.axes.Axes instance.

    Parameters
    ----------
    ssodf : Pandas DataFrame
        SSO reflectance spectra table in same format (columns/names) as in Gaia DR3 archive.
    ax : matplotlib.axes.Axes
        Axes instance in which to plot the spectrum. This is assumed to have the horizontal and vertical axes in units of nm (wavelength) and dimensionless (relative reflectance)
    mpcnumbers : list of int
        Minor Planet Centre asteroid designations
    interpolate : boolean
        If true interpolate the spectrum between the input data points.

    Returns
    -------
    Nothing
    """
    wavelengths = ssodf["wavelength"].unique()
    wavelengths.sort()
    num_fine = int((wavelengths[-1] - wavelengths[0]) / 2)
    x_fine = np.linspace(wavelengths[0], wavelengths[-1], num_fine)
    astnames = mpcnum_to_name(ssodf, mpcnumbers)
    for num, name in zip(mpcnumbers, astnames):
        np.where(ssodf.number_mp == num)
        y = ssodf.loc[np.where(ssodf.number_mp == num)].reflectance_spectrum
        elem = ax.plot(wavelengths, y, "o", label=f"{num} {name}", **kwargs)
        if interpolate:
            y_fine = Akima1DInterpolator(
                wavelengths[np.logical_not(np.isnan(y))], y[np.logical_not(np.isnan(y))]
            )(x_fine, extrapolate=True)
            ax.plot(x_fine, y_fine, "-", c=elem[0].get_c(), **kwargs)
    if addlegend:
        ax.legend()


def plot_spectra_collection(
    ssodf,
    ax,
    fig,
    mpcnumbers,
    plotall=False,
    ylims=(0.5, 2),
    wbinfactor=2,
    colmap=plt.colormaps["viridis"],
    cempty="#ffffff",
    cbarticks=None,
):
    """
    Plot a collection of asteroid spectra as a density image based on interpolated versions of the spectra. This is based on the example https://matplotlib.org/stable/gallery/statistics/time_series_histogram.html#sphx-glr-gallery-statistics-time-series-histogram-py

    Parameters
    ----------
    ssodf : Pandas DataFrame
        SSO reflectance spectra table in same format (columns/names) as in Gaia DR3 archive.
    ax : matplotlib.axes.Axes
        Axes instance on which to create the plot.
    fig : matplotlib.figure.Figure
        Figure instance which contains ax
    mpcnumbers : list of int
        Minor Planet Centre asteroid designations.
    plotall : boolean
        If true plot all spectra in ssodf (and ignore the mpcnumbers list).
    ylims : tuple of floats
        Plot limits for vertical axis (relative reflectance)
    wbinfactor : int
        Factor by which to increase the standard binsize of 1 nm (use even numbers for the Gaia DR3 wavelength sampling)
    colmap : Colormap
        The color map as a Colormap instance
    cempty : str
        Color for empty pixels
    cbarticks : list of floats
        List of tick values for color bar

    Returns
    -------
    Nothing
    """
    wavelengths = ssodf["wavelength"].unique()
    wavelengths.sort()
    num_fine = int((wavelengths[-1] - wavelengths[0]) / wbinfactor)
    x_fine = np.linspace(wavelengths[0], wavelengths[-1], num_fine)
    if plotall:
        y = ssodf.reflectance_spectrum.to_numpy().reshape(
            (len(ssodf.number_mp.unique()), wavelengths.size)
        )
    else:
        y = extract_spectra(ssodf, mpcnumbers)

    y_fine = np.concatenate(
        [
            Akima1DInterpolator(
                wavelengths[np.logical_not(np.isnan(y_row))],
                y_row[np.logical_not(np.isnan(y_row))],
            )(x_fine, extrapolate=True)
            for y_row in y
        ]
    )
    x_fine = np.broadcast_to(x_fine, (y.shape[0], num_fine)).ravel()

    cmap = colmap.with_extremes(bad=cempty)
    h, xedges, yedges = np.histogram2d(
        x_fine,
        y_fine,
        bins=[num_fine, 150],
        range=[[wavelengths[0], wavelengths[-1]], [0.5, 2]],
    )
    clipmax = scoreatpercentile(h[h > 0], 99.5)
    h[h == 0] = np.nan

    imnorm = ImageNormalize(h, vmax=clipmax, stretch=LogStretch(1000))
    pcm = ax.pcolormesh(xedges, yedges, h.T, cmap=cmap, rasterized=True, norm=imnorm)
    fig.colorbar(pcm, ax=ax, label="# points", ticks=cbarticks)
    ax.set_ylim(ylims)
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("Normalized reflectance")


def plot_ctype_families(dr3hband, fam, pmin=0.8, smax=0.1, savefig=False):
    """
    For the given family plot the inverse diameter and $H$ vs the proper semi-major axis, shown the hydrated and non-hydrated asteroids.

    Parameters
    ----------
    dr3hband : pandas DataFrame
        Table of C-type asteroids with median probabilities to have an h-band. Based on output from the ClassifyAsteroids notebook.
    fam : str
        Family name
    fig : matplotlib.figure.Figure
        Figure instance which contains axfam
    pmin : float
        Threshold for considering and asteroid to have an hband (p>=pmin, for p<=1-pmin the asteroid is considered to have no h-band)
    smax : float
        Maximum of RSE on of NN probabilities to have an h-band. This is how robustly an asteroid is classified as having an h-band (or not).
    savefig : boolean
        If true save a png of the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plots.
    """
    hband, nohband = get_hband_nohband_indices(dr3hband)
    hband = hband & (dr3hband.parent_name == fam)
    nohband = nohband & (dr3hband.parent_name == fam)
    dubious = ~(hband | nohband) & (dr3hband.parent_name == fam)
    num_hband = dr3hband.source_id[hband].size
    num_nohband = dr3hband.source_id[nohband].size
    num_dubious = dr3hband.source_id[dubious].size

    useagab(fontsize=24, axislinewidths=2, linewidths=3, lenticks=8)
    fig = plt.figure(figsize=(2 * 12, 8))
    gs = GridSpec(1, 2, figure=fig)
    axfam = []
    axfam.append(fig.add_subplot(gs[0, 0]))
    axfam.append(fig.add_subplot(gs[0, 1]))
    for ax in axfam:
        apply_tufte(ax)

    symsize = 10
    hsym = "o"
    nohsym = "^"
    dubsym = "+"

    axfam[0].plot(
        dr3hband.a_p[dubious],
        dr3hband.inverse_diameter[dubious],
        dubsym,
        ms=symsize,
        c="k",
        label="Dubious",
    )

    axfam[0].plot(
        dr3hband.a_p[nohband],
        dr3hband.inverse_diameter[nohband],
        nohsym,
        ms=symsize,
        label="No h-band",
    )

    axfam[0].plot(
        dr3hband.a_p[hband],
        dr3hband.inverse_diameter[hband],
        hsym,
        ms=symsize,
        label="h-band",
    )

    axfam[0].set_xlabel(r"$a_\mathrm{p}$ [au]")
    axfam[0].set_ylabel(r"$1/D$ [km$^{-1}$]")

    axfam[1].plot(
        dr3hband.a_p[dubious],
        dr3hband.h[dubious],
        dubsym,
        ms=symsize,
        c="k",
        label="Dubious",
    )
    axfam[1].plot(
        dr3hband.a_p[nohband],
        dr3hband.h[nohband],
        nohsym,
        ms=symsize,
        label="No h-band",
    )
    axfam[1].plot(
        dr3hband.a_p[hband], dr3hband.h[hband], hsym, ms=symsize, label="h-band"
    )
    axfam[1].set_xlabel(r"$a_\mathrm{p}$ [au]")
    axfam[1].set_ylabel(r"$H$")

    k = 2
    for res, ares in _resonances.items():
        if (
            axfam[0].get_xlim()[0] - 0.05 < ares
            and ares < axfam[0].get_xlim()[1] + 0.05
        ):
            kleur = f"C{k}"
            axfam[0].axvline(x=ares, ls="--", color=kleur, label=f"{res} resonance")
            axfam[1].axvline(x=ares, ls="--", color=kleur, label=f"{res} resonance")
            k = k + 1
    axfam[1].set_xlim(axfam[0].get_xlim())

    axfam[0].legend(fontsize=14)
    axfam[1].legend(fontsize=14)

    fig.suptitle(
        f"{fam}: {num_dubious+num_hband+num_nohband}, {num_hband} hydrated, {num_nohband} non-hydrated, {num_dubious} dubious"
    )

    if savefig:
        plt.savefig(f"{fam}.png")

    return fig


def load_dr3_data(
    dr3file="./data/DR3ReflectanceSpectra.fits.gz", filefmt="fits", clipspectra=False
):
    """
    Read the Gaia DR3 reflectabce spectra into a Pandas DataFrame.

    The reflectance spectra are corrected according to the prescriptions by [Tinaut-Ruano et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023A%26A...669L..14T/abstract). The factors by which to multiply the `reflectance_spectrum` column are listed in their table 1.

    | Wavelength (nm) | Correction Factor |
    | -- | -- |
    | 374.0 | 1.07 |
    | 418.0 | 1.05 |
    | 462.0 | 1.02 |
    | 506.0 | 1.01 |

    Parameters
    ----------

    dr3file : str
        Path to file with Gaia DR3 reflectance spectra data. Assumed to have been obtained with the query
        ```sql
        select * from gaiadr3.sso_reflectance_spectrum
        ```
    filefmt : str
        Astropy file format string
    clipspectra : boolean
        If true remove the first and last three wavelength bins from the spectra.

    Returns
    -------

    df : Pandas DataFrame
        DataFrame with the Gaia DR3 reflectance spectra data
    """
    ssodata = Table.read(dr3file, format=filefmt).to_pandas()

    # convert the bytes in this column to actual strings
    ssodata["denomination"] = ssodata["denomination"].str.decode("utf-8")

    # sort by source (MPC number) and for each source by wavelength
    ssodata = ssodata.sort_values(by=["number_mp", "wavelength"])
    ssodata = ssodata.reset_index().drop(columns=["index"])

    # Apply corrections
    ssodata.loc[
        np.where(ssodata.wavelength == 374.0)[0], ("reflectance_spectrum")
    ] *= 1.07
    ssodata.loc[
        np.where(ssodata.wavelength == 418.0)[0], ("reflectance_spectrum")
    ] *= 1.05
    ssodata.loc[
        np.where(ssodata.wavelength == 462.0)[0], ("reflectance_spectrum")
    ] *= 1.02
    ssodata.loc[
        np.where(ssodata.wavelength == 506.0)[0], ("reflectance_spectrum")
    ] *= 1.01

    ssodata.loc[
        np.where(ssodata.wavelength == 374.0)[0], ("reflectance_spectrum_err")
    ] *= 1.07
    ssodata.loc[
        np.where(ssodata.wavelength == 418.0)[0], ("reflectance_spectrum_err")
    ] *= 1.05
    ssodata.loc[
        np.where(ssodata.wavelength == 462.0)[0], ("reflectance_spectrum_err")
    ] *= 1.02
    ssodata.loc[
        np.where(ssodata.wavelength == 506.0)[0], ("reflectance_spectrum_err")
    ] *= 1.01

    wavelengths = ssodata["wavelength"].unique()
    wavelengths.sort()

    # Create a normalized version of the reflectance spectra, where at each wavelength the value is divided by the mean over all spectra (at the same wavelength)
    ssodata["reflectance_spectrum_normalized"] = ssodata["reflectance_spectrum"]
    for w in wavelengths:
        ssodata.loc[
            np.where(ssodata.wavelength == w)[0], ("reflectance_spectrum_normalized")
        ] /= np.mean(
            ssodata.loc[np.where(ssodata.wavelength == w)[0], ("reflectance_spectrum")]
        )

    if clipspectra:
        ssodata = ssodata.drop(
            index=ssodata.loc[
                (ssodata.wavelength < wavelengths[3])
                | (ssodata.wavelength > wavelengths[-4])
            ].index
        )
        ssodata = ssodata.reset_index().drop(columns=["index"])

    return ssodata


def create_ml_training_classes(
    ssodata,
    infile="./data/Asteroid_Classification_vAB-finalclasses.csv",
    filefmt="ascii.csv",
    filter=True,
):
    """
    Read in the asteroid classifications by Joost and Marco and create lists of asteroid MPC numbers for the various types used in the ML classification stages.

    Parameters
    ----------
    ssodata : Pandas DataFrame
        Gaia DR3 reflectance spectra data
    infile : str
        Path to file with asteroid classifications
    filefmt : str
        Astropy file format string
    filter : boolean
        If true filter out asteroids with spectra that have refectance_spectrum_flag set to '2' for one or more wavelengths

    Returns
    -------
    classmaps : dict
        Mapping from classes to lists of corresponding asteroid MPC numbers
    """
    ssotypes = Table.read(infile, format=filefmt).to_pandas()

    if filter:
        grouped = ssodata.groupby(by="number_mp").max()
        numsdr3 = grouped[grouped.reflectance_spectrum_flag != 2].index.to_list()
    else:
        numsdr3 = ssodata.number_mp.unique().tolist()

    ctype_nums = ssotypes.loc[ssotypes.AdjustedType == "C"].MPCnumber.to_list()
    ctype_nums_dr3 = list(set(ctype_nums).intersection(set(numsdr3)))

    othertype_nums = ssotypes.loc[
        ssotypes.AdjustedType.isin(["S", "X", "Other"])
    ].MPCnumber.to_list()
    othertype_nums_dr3 = list(set(othertype_nums).intersection(set(numsdr3)))

    resttype_nums = ssotypes.loc[
        ~ssotypes.AdjustedType.isin(["C", "S", "X", "Other"])
    ].MPCnumber.to_list()
    resttype_nums_dr3 = list(set(resttype_nums).intersection(set(numsdr3)))

    ctype_hband_nums = ssotypes.loc[
        (ssotypes.AdjustedType == "C") & (ssotypes.AdjustedBand == "yes")
    ].MPCnumber.to_list()
    ctype_hband_nums_dr3 = list(set(ctype_hband_nums).intersection(set(numsdr3)))

    ctype_nohband_nums = ssotypes.loc[
        (ssotypes.AdjustedType == "C") & (ssotypes.AdjustedBand == "no")
    ].MPCnumber.to_list()
    ctype_nohband_nums_dr3 = list(set(ctype_nohband_nums).intersection(set(numsdr3)))

    return {
        "C": ctype_nums_dr3,
        "other": othertype_nums_dr3,
        "rest": resttype_nums_dr3,
        "C_h": ctype_hband_nums_dr3,
        "C_no_h": ctype_nohband_nums_dr3,
    }


def get_hband_nohband_indices(dr3hband, pmin=0.8, smax=0.1):
    """
    From the table of Gaia DR3 asteroids classified as C-type extract the indices of asteroids with h-band and without h-band.

    Parameters
    ----------
    dr3hband : pandas DataFrame
        Table of C-type asteroids with median probabilities to have an h-band. Based on output from the ClassifyAsteroids notebook.
    pmin : float
        Threshold for considering and asteroid to have an hband (p>=pmin, for p<=1-pmin the asteroid is considered to have no h-band)
    smax : float
        Maximum of RSE on of NN probabilities to have an h-band. This is how robustly an asteroid is classified as having an h-band (or not).

    Returns
    -------
    hband, nohband : boolean arrays
        Arrays to be used as index into dataframe holding C-type asteroids median probabilities to have an h-band.
    """
    hband = (dr3hband.hband_mean_prob >= pmin) & (dr3hband.hband_rse_prob <= smax)
    nohband = (dr3hband.hband_mean_prob <= 1 - pmin) & (dr3hband.hband_rse_prob <= smax)
    return hband, nohband


def create_families_table(dr3hband, pmin=0.8, smax=0.1):
    """
    From the table of Gaia DR3 asteroids classified as C-type extract a table listing the number of asteroids with h-band and without h-band for each asteroid family.

    Parameters
    ----------
    dr3hband : pandas DataFrame
        Table of C-type asteroids with median probabilities to have an h-band. Based on output from the ClassifyAsteroids notebook.
    pmin : float
        Threshold for considering and asteroid to have an hband (p>=pmin, for p<=1-pmin the asteroid is considered to have no h-band)
    smax : float
        Maximum of RSE on of NN probabilities to have an h-band. This is how robustly an asteroid is classified as having an h-band (or not).

    Returns
    -------
    families : pandas DataFrame
        Table that lists for each family the number of asteroids with h-band, with no h-band, and the sum of these two values.
    """
    hband, nohband = get_hband_nohband_indices(dr3hband, pmin=pmin, smax=smax)
    hband_families = (
        dr3hband[hband]
        .groupby(by="parent_name", as_index=False)
        .count()
        .sort_values(by=["source_id", "parent_name"])
        .filter(["parent_name", "source_id"])
        .reset_index()
        .drop(columns=["index"])
        .rename(columns={"source_id": "N_hband"})
    )

    nohband_families = (
        dr3hband[nohband]
        .groupby(by="parent_name", as_index=False)
        .count()
        .sort_values(by=["source_id", "parent_name"])
        .filter(["parent_name", "source_id"])
        .reset_index()
        .drop(columns=["index"])
        .rename(columns={"source_id": "N_nohband"})
    )

    dubious_families = (
        dr3hband[np.logical_not(hband | nohband)]
        .groupby(by="parent_name", as_index=False)
        .count()
        .sort_values(by=["source_id", "parent_name"])
        .filter(["parent_name", "source_id"])
        .reset_index()
        .drop(columns=["index"])
        .rename(columns={"source_id": "N_dubious"})
    )

    families_total = (
        dr3hband.groupby(by="parent_name", as_index=False)
        .count()
        .sort_values(by=["source_id", "parent_name"])
        .filter(["parent_name", "source_id"])
        .reset_index()
        .drop(columns=["index"])
        .rename(columns={"source_id": "N_total"})
    )

    families = (
        pd.merge(
            pd.merge(
                pd.merge(families_total, hband_families, on="parent_name", how="outer"),
                nohband_families,
                on="parent_name",
                how="outer",
            ),
            dubious_families,
            on="parent_name",
            how="outer",
        )
        .convert_dtypes()
        .fillna(value=0)
    )

    return families
