import requests
import pandas as pd
import numpy as np
import meteostat as ms
from datetime import datetime
from string import punctuation

def get_weather_data(place,start,end=datetime.now()):
    
    '''
     A function for getting daily weather data for a particular location from start date to end date.
     
     Parameters:
     
     place : str
         A string containing the common name of the location to fetch weather data for.
         
         Examples: 'Chicago, IL', 'South Bend, IN, 46617, US', 'Paris, France'
         
     start : str | datetime
         A string containing the start date to fetch weather data for. Must be in the form %Y-%m-%d. Example: '2021-12-25'.
         This may also be a datetime object.
         
     end : str | datetime, optional (Defaults to the current date)
         A string containing the end date to fetch weather data for. Must be in the form %Y-%m-%d. Example: '2021-12-25'.
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
    '''
    
    
    #Base url for searching geocode.maps.co
    base_url = 'https://geocode.maps.co/search?q='
    
    
    #Remove punctuation from place and replace spaces with + for the search url
    place_no_punc = ''

    for char in place:
        if char not in punctuation:
            place_no_punc = place_no_punc+char
            
    place_url = place_no_punc.replace(' ','+')
    
    #Build search URL from base URL and place
    url = base_url + place_url
    
    
    #Request place information from the created url. The information is returned as a dict assigned as place_info
    r = requests.get(url)
    place_info = r.json()[0]
    
    #Extract the lat and lon info from place_info
    lat = float(place_info['lat'])
    lon = float(place_info['lon'])
   


    
    #Make sure start and end are of type = datetime. This needs work to accept more date formats that may be entered
    if type(start)==str:
        start = datetime.strptime(start,'%Y-%m-%d')
        
    if type(end)==str:
        end = datetime.strptime(end,'%Y-%m-%d')
        
    
    location = ms.Point(lat,lon)
    
    data = ms.Daily(location,start,end)
    data = data.fetch()
    return data