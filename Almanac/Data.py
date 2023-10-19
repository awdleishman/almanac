import requests
import meteostat as ms
from datetime import datetime
from string import punctuation


def get_weather_data(place, start, end=datetime.now()):
    """
     A function for getting daily weather data for a particular location from
     start date to end date.

     Parameters:

     place : str
         A string containing the common name of the location to fetch weather
         data for.

         Examples: 'Chicago, IL', 'South Bend, IN, 46617, US', 'Paris, France'

     start : str | datetime
         A string containing the start date to fetch weather data for. Must be
         in the form %Y-%m-%d. Example: '2021-12-25'.
         This may also be a datetime object.

     end : str | datetime, optional (Defaults to the current date)
         A string containing the end date to fetch weather data for. Must be
         in the form %Y-%m-%d. Example: '2021-12-25'.
         This may also be a datetime object.


    Returns:

    data : pandas.DataFrame
        A data frame containing the following fields:
            time: The date : datetime
            tavg: The average air temperature in C : float
            tmin: The minimum air temperature in C : float
            tmax: The maximum air temperature in C : float
            prcp: The daily precipitation total in mm : float
            snow: The snow depth in mm : float
            wdir: The average wind direction in degrees : float
            wspd: The average wind speed in km/h : float
            wpgt: The peak wind gust in km/h : float
            pres: The average sea-level air pressure in hPa : float
            tsun: The daily sunshine total in minutes : float
    """

    # Base url for searching geocode.maps.co
    base_url = "https://geocode.maps.co/search?q="

    # Remove punctuation from place and replace spaces with + for the search
    # url
    place_no_punc = ""

    for char in place:
        if char not in punctuation:
            place_no_punc = place_no_punc + char

    place_url = place_no_punc.replace(" ", "+")

    # Build search URL from base URL and place
    url = base_url + place_url

    # Request place information from the created url. The information is
    # returned as a dict assigned as place_info
    r = requests.get(url)
    place_info = r.json()[0]

    # Extract the lat and lon info from place_info
    lat = float(place_info["lat"])
    lon = float(place_info["lon"])

    # Make sure start and end are of type = datetime. This needs work to
    # accept more date formats that may be entered
    if type(start) is str:
        start = datetime.strptime(start, "%Y-%m-%d")

    if type(end) is str:
        end = datetime.strptime(end, "%Y-%m-%d")

    location = ms.Point(lat, lon)

    data = ms.Daily(location, start, end)
    data = data.fetch()
    return data


def binarize(df, cols, thresh=0):
    """
    A function to binarize weather data.
    Sets values less than 0 to 1 and sets values greater then 0 to 0.

    Parameters:

    df : pandas.DataFrame
        A DataFrame object containing weather data.

    cols : str | list
        A string or list of strings containing
        the names of columns in df to binarize.

    thresh : float
        A number to split the binarization on.
        Values below thresh are assigned 1,
        values above thresh are assigned 0.


    Returns:

    df : pandas.DataFrame
        A DataFrame containing the original columns of df and
        the new columns of binarized data.
    """

    if isinstance(cols, str):
        df[cols + "_bin"] = df[cols]

        for i, temp in enumerate(df[cols + "_bin"]):
            if temp <= thresh:
                df[cols + "_bin"].iloc[i] = 1
            else:
                df[cols + "_bin"].iloc[i] = 0

    elif isinstance(cols, list):
        for col in cols:
            df[col + "_bin"] = df[col]

            for i, temp in enumerate(df[col + "_bin"]):
                if temp <= thresh:
                    df[col + "_bin"].iloc[i] = 1
                else:
                    df[col + "_bin"].iloc[i] = 0

    return df


def get_frost_dates(data):
    """
    A function that finds the dates of the first and last frost of a season
    for every year in data.
    First frost is defined as the first day after summer where the minimum
    temp is <= 0C.
    Last frost is defined as the last day before summer where the minimum temp
    is <= 0C.


    Parameters:

    data : pandas.DataFrame
        A DataFrame object containing weather data.


    Returns:

    first_frost : list
        A list of the first frost dates of each year.

    last_frost : list
        A list of the last frost dates of each year.
    """

    # If binarized column does not exist,
    # create it.
    if "tmin_bin" in data.columns:
        pass
    else:
        data = binarize(data, "tmin")

    # Create the dict data_year
    # This is a dictionary of DataFrames
    # for each year of data with years as keys
    data_year = {}

    for y in data.index.year.unique():
        data_year[y] = data.loc[data.index.year == y]

    # Create lists of the first and last frost dates.
    last_frost = []
    first_frost = []

    # Divide the years in half.
    # Note this split assumes a location in the Northern Hemisphere
    for y in data_year.keys():
        first_half = data_year[y].loc[data_year[y].index.month < 7]
        second_half = data_year[y].loc[data_year[y].index.month > 7]

        try:
            last_frost.append(
                first_half["tmin_bin"][first_half["tmin_bin"] == 1].index[-1]
            )
        except Exception:
            last_frost.append(None)

        try:
            first_frost.append(
                second_half["tmin_bin"][second_half["tmin_bin"] == 1].index[0]
            )
        except Exception:
            first_frost.append(None)

    return first_frost, last_frost
