import glob
import logging
import os

import numpy as np
import pandas as pd
import yaml
from scipy import interpolate

from tardis.util.base import (
    atomic_number2element_symbol,
    is_valid_nuclide_or_elem,
)

logger = logging.getLogger(__name__)

XG_FILE_NAMES = ["vel", "rho", "temp"]
DAT_FILE_NAMES = ["lum_observed", "T_eff", "vel_photo", "lum_photo"]


def xg_to_dict(fname):
    """
    parse the .xg file from SNEC output into a dictionary.
    credit: Brandon Barker and Chelsea Harris

    Parameters
    ----------
    fname : str

    Returns
    -------
    snec_xg_data : dictionary
    """
    snec_xg_data = {}
    with open(fname) as rf:
        for line in rf:
            cols = line.split()
            # Beginning of time data - make key for this time
            if "Time" in line:
                time = float(cols[-1])
                snec_xg_data[time] = []
            # In time data -- build x,y arrays
            elif len(cols) == 2:
                snec_xg_data[time].append(np.fromstring(line, sep=" "))
            # End of time data (blank line) -- make list into array
            else:
                snec_xg_data[time] = np.array(snec_xg_data[time])

    return snec_xg_data


def parse_snec_to_tardis(
    snec_folder_path,
    tardis_config_output_path=None,
    time_in_days=None,
    save_shells_above_photosphere_only=False,
    include_dilution_factor=False,
    comps_profile_file_path=None,
    comp_use_boxcared=True,
    snec_data_folder_path_comp_boxcar=None,
    use_vel_diff=False,
    num_keep_shells=45,
):
    """
    Purpose:
    ---------
    Parse the SNEC output to the TARDIS input config and csvy files.

    TARDIS Configuration Mapping from SNEC:
        supernova:
            luminosity_requested -- from lum_observed.dat at selected time (first col is time in seconds, second col is luminosity in erg/s)
            time_explosion -- the time key from any xg file (which has time in seconds as keys, the xg files are profile snapshots)

        plasma:
            initial_t_inner -- from T_eff.dat (Effective T of the photosphere) at selected time

        csvy_model: see description in subsection below

    CSVY model information:
        model_density_time_0 -- same as time_explosion
        model_isotope_time_0 -- the initial time of the composition profile used in SNEC
        v_inner_boundary -- vel_photo.dat at selected time (first col is time in seconds, second col is photospheric velocity in cm/s)
        velocity -- vel.xg (second column at selected time, first col is mass grid in unit of g)
        density  -- rho.xg (second column at selected time, first col is mass grid in unit of g)
        t_rad (optional)  -- temp.xg (second column at selected time, first col is mass grid in unit of g)
        composition -- profile/xx_comps.snec (need to match mass coordinate with velocity) at selected time

    ----------

    Parameters
    ----------
        snec_folder_path: str
            The path to the folder that contains the SNEC Data and profiles
        tardis_config_output_path: None or str
            If str -- The path to the folder that contains the input from TARDIS
            If None -- The output tardis files will not be produced
        time_in_days: list/array or None, default None
            If given, only the selected time (unit in days) will be used to generate the TARDIS input files, if None will use all available time steps
        calculate_dilution_factor: bool
            If True, calculate the dilution factor using the optical depth
        comps_profile_file_path: str
            The path to the composition profile file from SNEC if specified
        comp_use_boxcared: bool, default True
            Use the boxcar smoothed composition profile from SNEC instead of the input file
        snec_data_folder_path_comp_boxcar: str or None, default None
            The path to the folder that contains the boxcar smoothed composition profile, if None will use the snec_folder_path
        use_vel_diff: bool
            If True, use the velocity adjacent difference to check homologous expansion,
            otherwise assume homologous for time step except the first one
        num_keep_shells: int
            The ROUGH number of shells to keep in the TARDIS model
    ----------
    output: (saved in tardis_config_output_path)
        time_series_log.csv
        tardis_config: yml files at each time
        tardis_model: csvy at each time

    """
    # get the folder name
    snec_folder_name = snec_folder_path.split("/")[-1]
    snec_data_folder_path = f"{snec_folder_path}/Data"

    # read the snec data
    if comps_profile_file_path is None:
        try:
            comp_files = glob.glob(f"{snec_folder_path}/profiles/*_comps.snec")
            snec_comps_profile_file_path = comp_files[0]
            if len(comp_files) > 1:
                Warning(
                    "More than one composition profile file found, please specify one."
                )
        except:
            ValueError("No composition profile file found.")
    else:
        snec_comps_profile_file_path = comps_profile_file_path

    dict_SNEC_output = snec_data_to_dict(snec_data_folder_path)

    # read the composition profile
    if comp_use_boxcared == True:
        if snec_data_folder_path_comp_boxcar is None:
            snec_data_folder_path_comp_boxcar = snec_data_folder_path
            print(
                "Using the same folder for boxcar smoothed composition profile."
            )
        df_snec_comps = snec_boxcar_comps_profile_to_dataframe(
            snec_comps_profile_file_path,
            snec_data_folder_path_comp_boxcar,
            dict_SNEC_output["mass"],
        )
    else:
        df_comps = snec_comps_profile_to_dataframe(snec_comps_profile_file_path)
        # interpolate the composition profile to the mass grid of the SNEC output
        df_snec_comps = interpolate_composition_profile(
            df_comps, dict_SNEC_output
        )

    # generate the time mask for the selected time steps
    selected_time_mask = generate_time_mask(
        dict_SNEC_output["time"],
        dict_SNEC_output["vel"],
        dict_SNEC_output["vel_photo_profile"],
        use_vel_diff=use_vel_diff,
    )

    # filter the data to the selected time steps
    for param in ["time"] + xg_params + [item + "_itp" for item in dat_params]:
        dict_SNEC_output[param] = dict_SNEC_output[param][selected_time_mask]

    if tardis_config_output_path is not None:
        # create the output folder
        tardis_model_path = f"{tardis_config_output_path}/{snec_folder_name}"
        if not os.path.exists(tardis_model_path):
            os.makedirs(tardis_model_path)

        if time_in_days is None:
            # write tardis config and csvy file for each time step
            for time_index, _ in enumerate(dict_SNEC_output["time"]):
                new_csvy_path = f"{tardis_model_path}/{snec_folder_name}_tardis_csvy_{time_index}.csvy"
                new_config_path = f"{tardis_model_path}/{snec_folder_name}_tardis_config_{time_index}.yml"
                save_tardis_config_and_csvy(
                    dict_SNEC_output,
                    time_index,
                    df_snec_comps,
                    new_csvy_path,
                    new_config_path,
                    include_dilution_factor=include_dilution_factor,
                    save_shells_above_photosphere_only=save_shells_above_photosphere_only,
                    num_keep_shells=num_keep_shells,
                )
        else:
            # write tardis config and csvy file for the selected time steps only (unit in days)
            for time in time_in_days:
                time_index = np.argmin(
                    np.abs(dict_SNEC_output["time"] - time * 24 * 3600)
                )
                if (
                    np.abs(
                        dict_SNEC_output["time"][time_index] / 24 / 3600 - time
                    )
                    > 1
                ):
                    Warning(
                        f"Time {time} day is not found in the SNEC output within +/-1d range."
                    )
                    continue
                new_csvy_path = f"{tardis_model_path}/{snec_folder_name}_tardis_csvy_{time}_day.csvy"
                new_config_path = f"{tardis_model_path}/{snec_folder_name}_tardis_config_{time}_day.yml"
                save_tardis_config_and_csvy(
                    dict_SNEC_output,
                    time_index,
                    df_snec_comps,
                    new_csvy_path,
                    new_config_path,
                    include_dilution_factor=include_dilution_factor,
                    save_shells_above_photosphere_only=save_shells_above_photosphere_only,
                    num_keep_shells=num_keep_shells,
                )

    return dict_SNEC_output, df_snec_comps


def save_tardis_config_and_csvy(
    dict_SNEC_output,
    time_index,
    df_snec_comps,
    new_csvy_path,
    new_config_path,
    include_dilution_factor=False,
    save_shells_above_photosphere_only=True,
    num_keep_shells=60,
):
    # get the time in day and photosphere index
    time_in_day = dict_SNEC_output["time"][time_index] / (60 * 60 * 24)
    photosphere_idx = int(dict_SNEC_output["index_photo_itp"][time_index])

    # attached velocity, density, t_rad, dilution_factor, and composition profile data into a df
    df_profiles_non_comp = pd.DataFrame(
        {
            "velocity": dict_SNEC_output["vel"][time_index],
            "density": dict_SNEC_output["rho"][time_index],
            "t_rad": dict_SNEC_output["temp"][time_index],
        },
    )
    if include_dilution_factor == True:
        # TODO: this is not TRUE wolfgang said, geometric dilution -- which is the tardis use, try without it
        raise ValueError(
            "calculate_dilution_factor function currently is wrong, please revise!"
        )
        # df_profiles_non_comp["dilution_factor"] = calculate_dilution_factor(
        #     dict_SNEC_output["tau"][time_index]
        # )
    df_profiles = df_profiles_non_comp.join(
        df_snec_comps.reset_index(drop=True)
    )

    # limit the shells numebers -- computational cost - This need to filtered before anything else ensure the index is correct
    if save_shells_above_photosphere_only == True:
        df_profiles = df_profiles[max([0, photosphere_idx - 15]) :]

    # filter out the zero density and velocity shells on the outer region
    df_profiles = df_profiles.loc[
        (df_profiles.density > 0) & (df_profiles.velocity > 0)
    ]

    # # filter out the inner shells that's purposely replaced with pure He
    # df_profiles = df_profiles.loc[df_profiles.He4 < 1]

    # filter out the outer shells that has t_radiative too low for TARDIS -> but this cause trouble though so replace low T shells with 500K instead
    df_profiles = df_profiles.loc[df_profiles.t_rad > 0]

    # discard the shells that has velocity backwards (which technically is in non-homologous expansion, but we can cut those out if there are only a few of them)
    while (df_profiles["velocity"].diff() <= 0).any():
        df_profiles = delete_non_increasing_neighbour(df_profiles, "velocity")

    # discard the element columns that has all zero values
    df_profiles = df_profiles.loc[:, (df_profiles != 0).any(axis=0)]

    if num_keep_shells is not None:
        shell_gap_length = max(
            [int(np.floor(df_profiles.shape[0] / num_keep_shells)), 1]
        )
        df_csv = df_profiles[::shell_gap_length]
        # if the last none-zero density shell is not in, append it
        if df_profiles.index[-1] not in df_csv.index:
            df_csv = pd.concat([df_csv, df_profiles[-1:]])
    else:
        df_csv = df_profiles

    # write the tardis csvy file
    modify_csvy_headers = {
        "name": new_csvy_path.split("/")[-1],
        "model_density_time_0": f"{time_in_day:.3f} day",
        "v_inner_boundary": f"{dict_SNEC_output['vel_photo_itp'][time_index]:.6e} cm/s",
    }
    write_tardis_csvy(modify_csvy_headers, df_csv, new_csvy_path)

    # write the tardis config file
    modify_parameters = {
        "supernova": {
            "luminosity_requested": f"{dict_SNEC_output['lum_observed_itp'][time_index]} erg/s",
            "time_explosion": f"{time_in_day:.3f} day",
        },
        "plasma": {
            "initial_t_inner": f"{dict_SNEC_output['T_eff_itp'][time_index]} K"
        },
    }
    write_tardis_config(
        modify_parameters,
        new_config_path,
        csvy_model_path=new_csvy_path.split("/")[-1],
    )


def delete_non_increasing_neighbour(df, col_name):
    subdf = df[(df[col_name].diff() > 0)]
    if df.iloc[0][col_name] < subdf.iloc[0][col_name]:
        subdf = pd.concat([df.iloc[:1], subdf])
    return subdf


def calculate_dilution_factor(optical_depth):
    dilution_factor = np.exp(-optical_depth)
    return dilution_factor


def generate_time_mask(
    time, velocity_arrays, vel_photo_profile, use_vel_diff=False
):
    """
    Purpose:
    ---------
    Generate the time mask for the selected time steps.

    ----------

    Parameters
    ----------
        time: numpy array
            The time array
        velocity_arrays: numpy array
            The velocity 2D arrays
        vel_photo_profile: dictionary
            The dictionary that contains the photospheric velocity profile (time, vel_photo)
        use_vel_diff: bool
            If True, use the velocity adjacent difference to check homologous expansion,
            otherwise assume homologous for time step except the first one
    ----------
    output:
        selected_time_mask: numpy array
            The time mask for the selected time steps
    """
    ## select the time window that the ejecta is in homologous expansion and in photospheric phase
    if use_vel_diff == True:
        # calculate the velocity adjacent difference for use of checking homologous expansion
        diff_vel = np.diff(velocity_arrays, axis=1)
        homologous_mask = np.all(diff_vel >= 0, axis=1)
    else:
        homologous_mask = time > 0

    # taking the time when photospheric velocity gets to 0 as the time upper limit
    photospheric_time_limit = vel_photo_profile["time"][
        vel_photo_profile["vel_photo"] > 0
    ][-1]
    photospheric_mask = time <= photospheric_time_limit

    selected_time_mask = np.where(homologous_mask & photospheric_mask)

    if selected_time_mask[0].size == 0:
        ValueError("No time step selected for homologous + photospheric phase.")

    return selected_time_mask


def snec_data_to_dict(snec_data_folder_path):
    """
    Purpose:
    ---------
    Parse the output from SNEC (a time series) to a single dictionary.


    ----------

    Parameters
    ----------
        snec_data_folder_path: str
            The path to the folder that contains the Data output from SNEC

    ----------
    output:
        A dictionary contains selected information of the SNEC output
    """
    dict_SNEC_output = {"vel": [], "rho": [], "temp": [], "tau": []}

    # read in the time steps that larger than 0 seconds (skipping the first time step)
    for i, param in enumerate(xg_params):
        param_data = xg_to_dict(f"{snec_data_folder_path}/{param}.xg")
        if param == "vel":
            dict_SNEC_output["time"] = np.array(list(param_data.keys()))[
                1:
            ]  # the first time step is time 0
            dict_SNEC_output["mass"] = param_data[0].T[0]
        else:
            # check if the simulation time matches
            assert np.array_equal(
                dict_SNEC_output["time"], np.array(list(param_data.keys()))[1:]
            )

        for time, data in param_data.items():
            if time > 0:
                # check if the mass grid matches
                assert np.array_equal(dict_SNEC_output["mass"], data.T[0])
                dict_SNEC_output[param].append(data.T[1])

        dict_SNEC_output[param] = np.array(dict_SNEC_output[param])

    for i, param in enumerate(dat_params):
        param_data = np.loadtxt(f"{snec_data_folder_path}/{param}.dat")
        # check if the simulation time matches
        dict_SNEC_output[param + "_profile"] = {
            "time": param_data.T[0],
            param: param_data.T[1],
        }
        # interpolate the data to the time grid
        f_itp = interpolate.interp1d(
            param_data.T[0],
            param_data.T[1],
            kind="linear",
            fill_value=(
                param_data.T[1][0],
                param_data.T[1][-1],
            ),
            bounds_error=False,
        )
        dict_SNEC_output[param + "_itp"] = f_itp(dict_SNEC_output["time"])
        if param == "index_photo":
            dict_SNEC_output[param + "_itp"] = dict_SNEC_output[
                param + "_itp"
            ].astype(int)

    # # get the minimum above-zero photospheric velocity
    # if cut_inner_region:
    #     first_shell_above_photosphere_all_time = np.min(
    #         dict_SNEC_output["index_photo_profile"]["index_photo"][
    #             dict_SNEC_output["vel_photo_profile"]["vel_photo"] > 0
    #         ]
    #     )

    #     # cut the inner regions (defined as the shells below the lowest above-zero photospheric velocity)
    #     cut_index = max(
    #         [(int(first_shell_above_photosphere_all_time) - 1), 1]
    #     )  # avoid the first shell which has velocity 0
    #     dict_SNEC_output["mass"] = dict_SNEC_output["mass"][cut_index:]
    #     for i, param in enumerate(xg_params):
    #         dict_SNEC_output[param] = dict_SNEC_output[param][:, cut_index:]

    return dict_SNEC_output


def snec_comps_profile_to_dataframe(snec_comps_profile_file_path):
    """
    Purpose:
    ---------
    Parse the SNEC composition profile to a dataframe.
    """
    # Read the file and extract the second and third lines into arrays
    with open(snec_comps_profile_file_path) as file:
        lines = file.readlines()
        mass_numbers = np.array(lines[1].split()).astype(int)
        atomic_numbers = np.array(lines[2].split()).astype(int)

    # Convert the atomic numbers to element symbols
    element_symbols = [
        atomic_number2element_symbol(atomic_number) + str(mass_number)
        for atomic_number, mass_number in zip(
            atomic_numbers[1:], mass_numbers[1:]
        )
    ]
    # check if the nuiclide is valid
    for element_symbol in element_symbols:
        if is_valid_nuclide_or_elem(element_symbol) == False:
            Warning(
                f"{element_symbol} is not valid nuiclide in tardis database."
            )

    # Read the remaining lines into a DataFrame
    df_abundance = pd.read_csv(
        snec_comps_profile_file_path,
        skiprows=3,
        delim_whitespace=True,
        header=None,
    )
    df_abundance.columns = ["mass", "radius", "neutron"] + element_symbols

    return df_abundance


def snec_boxcar_comps_profile_to_dataframe(
    snec_comps_profile_file_path, snec_data_folder_path, snec_mass_grid
):
    """
    Purpose:
    ---------
    Parse the SNEC composition profile that are artificially smoothed by boxcar to a dataframe.
    """
    # Read the file and extract the second and third lines into arrays
    with open(snec_comps_profile_file_path) as file:
        lines = file.readlines()
        mass_numbers = np.array(lines[1].split()).astype(int)
        atomic_numbers = np.array(lines[2].split()).astype(int)

    # Convert the atomic numbers to element symbols
    element_symbols = [
        atomic_number2element_symbol(atomic_number) + str(mass_number)
        for atomic_number, mass_number in zip(
            atomic_numbers[1:], mass_numbers[1:]
        )
    ]

    # get composition profile by isotopes
    df_abundance = pd.DataFrame(index=snec_mass_grid, columns=element_symbols)
    for i, element_symbol in enumerate(element_symbols):
        # check if the nuiclide is valid
        if is_valid_nuclide_or_elem(element_symbol) == False:
            Warning(
                f"{element_symbol} is not valid nuiclide in tardis database."
            )

        # Read the smoothed composition profile from the snec data folder
        iso_id = (
            i + 2
        )  # +2 due to 1.fortran start with 1, 2. first col is neutrinos
        file = f"{snec_data_folder_path}/iso_id_{iso_id}_init_frac.dat"
        df_init_frac = pd.read_csv(
            file,
            sep=r"\s+",
            header=None,
        )
        df_abundance[element_symbol] = df_init_frac[1].values

    return df_abundance


def interpolate_composition_profile(df_comps, dict_SNEC_output):
    """
    Purpose:
    ---------
    Interpolate the composition profile to the mass grid of the SNEC output.

    ----------

    Parameters
    ----------
        df_comps: dataframe
            The dataframe that contains the composition profile
        dict_SNEC_output: dictionary
            The dictionary that contains the SNEC output

    ----------
    output:
        A dataframe that contains the interpolated composition profile
    """
    # Get the mass grid from the data
    data_mass_grid = dict_SNEC_output["mass"]

    # Get the composition profile mass grid
    profile_mass_grid = df_comps["mass"].values

    # Get the composition profile data
    profile_data = df_comps.drop(columns=["mass", "radius", "neutron"])

    # Interpolate the composition profile to the data mass grid
    df_interpolated_comps = pd.DataFrame(
        index=data_mass_grid, columns=profile_data.columns
    )
    for column in profile_data.columns:
        f_itp = interpolate.interp1d(
            profile_mass_grid,
            profile_data[column],
            kind="linear",
            fill_value=(
                profile_data[column].values[0],
                profile_data[column].values[-1],
            ),
            bounds_error=False,
        )
        df_interpolated_comps[column] = f_itp(data_mass_grid)

    return df_interpolated_comps


def write_tardis_config(
    modify_parameters, new_config_path, csvy_model_path=None
):
    """
    Purpose:
    ---------
    Write the TARDIS config file for a specific time step.

    ----------

    Parameters
    ----------
        modified_parameters: dict
            The dictionary that contains the to-be modified parameters
        new_config_path: str
            The path to the new config file
    """
    # load in the sample config yml
    with open(tardis_sample_config_path) as file:
        config = yaml.safe_load(file)

    # Modify the config dictionary as needed
    for key1, params in modify_parameters.items():
        for key2, value in params.items():
            config[key1][key2] = value

    if csvy_model_path is not None:
        config["csvy_model"] = csvy_model_path

    # Save the modified config back to a new YAML file
    with open(new_config_path, "w") as file:
        yaml.safe_dump(config, file, sort_keys=False)


def get_fields_names(column_names):
    fields = [
        {
            "name": "velocity",
            "unit": "cm/s",
            "desc": "velocities of shell outer bounderies.",
        },
        {
            "name": "density",
            "unit": "g/cm^3",
            "desc": "density within shell with corresponding outer velocity.",
        },
    ]

    if ("velocity" not in column_names) or ("density" not in column_names):
        ValueError("velocity and density are required fields.")
    else:
        column_names.remove("velocity")
        column_names.remove("density")

    if "t_rad" in column_names:
        fields.append(
            {
                "name": "t_rad",
                "unit": "K",
                "desc": "radiative temperature within shell with corresponding outer velocity.",
            }
        )
        column_names.remove("t_rad")

    if "dilution_factor" in column_names:
        fields.append(
            {
                "name": "dilution_factor",
                "desc": "dilution factor within shell with corresponding outer velocity.",
            }
        )
        column_names.remove("dilution_factor")

    # assume the rest are fractional abundance
    element_names = [atomic_number2element_symbol(i + 1) for i in range(118)]
    for element in column_names:
        # check the column is a valid element
        if is_valid_nuclide_or_elem(element):
            fields.append(
                {
                    "name": element,
                    "desc": f"fractional {element} abundance",
                }
            )
        else:
            Warning(f"{element} is not valid nuiclide in tardis database.")

    return fields


def write_tardis_csvy(modify_csvy_headers, df_csv, new_csvy_path):
    """
    Purpose:
    ---------
    Write the TARDIS model csvy file for a specific time step.

    ----------

    Parameters
    ----------
        modify_csvy_headers: dict
            The dictionary that contains the to-be modified headers
        new_csvy_path: str
            The path to the new csvy file
    """
    # Read the sample csvy file
    with open(tardis_sample_csvy_path) as file:
        csvy_lines = file.readlines()

    # Find the lines between "---" and datatype -- these are the headers
    start_index = csvy_lines.index("---\n")
    end_index = csvy_lines.index("datatype:\n")

    # Parse the lines as YAML
    yml_lines = csvy_lines[start_index:end_index]
    yml_data = yaml.safe_load("".join(yml_lines))

    # Modify the header dictionary as needed
    for key, value in modify_csvy_headers.items():
        yml_data[key] = value

    # add the datatype fields
    fields = get_fields_names(df_csv.columns.to_list())
    yml_data["datatype"] = {"fields": fields}

    # Convert the yml data back to lines
    yml_lines = yaml.dump(yml_data, sort_keys=False).splitlines()
    yml_lines = [line + "\n" for line in yml_lines]

    # Convert the csv data to lines
    csv_lines = df_csv.to_csv(
        index=False, float_format="%.5e", sep=","
    ).splitlines()
    csv_lines = [line + "\n" for line in csv_lines]

    # Save the updated csvy data
    updated_csvy_lines = (
        csvy_lines[: start_index + 1] + yml_lines + ["---\n"] + csv_lines
    )
    with open(new_csvy_path, "w") as file:
        file.writelines(updated_csvy_lines)
