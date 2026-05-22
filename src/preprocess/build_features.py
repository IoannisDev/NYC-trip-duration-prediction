import pandas as pd
import geopandas as gp
import numpy as np
from pandas import DataFrame
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GEOJSON_PATH = os.path.join(BASE_DIR, "data", "processed", "boundary.geojson")

def haversine_form(lat1:list,long1:list,lat2:list,long2:list) -> list:
    lat1 , long1,lat2,long2 = map(np.radians, [lat1, long1, lat2, long2])
    dlat = lat2-lat1
    dlong = long2-long1
    a  = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2)*np.sin(dlong/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 3956
    return c*r

def make_feat(df:DataFrame) ->DataFrame:
    df['distance'] = haversine_form(df['pickup_latitude'],df['pickup_longitude'],df['dropoff_latitude'],df['dropoff_longitude'])
    df = df[df['distance'] > 0]            
    df = df[df['distance'] <= 30]   
    df = df[df['trip_duration']>=1]
    df = df[df['trip_duration'] <= 120] 
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df.drop(['vendor_id','dropoff_datetime','store_and_fwd_flag'],axis =1,inplace=True)
    df['day_of_week'] = df['pickup_datetime'].dt.day_of_week
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    df['is_rush_hour'] = df['pickup_hour'].isin([7,8,9,16,17,18,19]).astype(int)
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
    mask = (
    (df['dropoff_longitude'].between(-74.26, -73.70)) &
    (df['dropoff_latitude'].between(40.49, 40.92)) & 
    (df['pickup_longitude'].between(-74.26,-73.70)) & 
    (df['pickup_latitude'].between(40.49,40.92))
    )
    df_clean = df[mask]
    boroughs = gp.read_file(GEOJSON_PATH)
    gdf = gp.GeoDataFrame(df_clean, geometry=gp.points_from_xy(df_clean['dropoff_longitude'], df_clean['dropoff_latitude']), crs="EPSG:4326")
    gdf = gdf.rename_geometry('points_geom')
    boroughs['geometry'] = boroughs['geometry'].buffer(0)
    boroughs = boroughs.to_crs("EPSG:4326")
    result = gp.sjoin(gdf, boroughs[['BoroName', 'geometry']], how='left', predicate='within')

    return result



    
