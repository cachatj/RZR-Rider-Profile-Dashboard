# ingest director of gpx

I am using gpxpy, and polars to ingest all .gpx files in a data folder & preprocess for analytics. For each of the gpx files, I would like to add elevation (feet) & speed (mph) data to all points on track. I would like to separate the tracks based on the date of the timestamp of latitude and longitude coordinates. There are over 100 individually tracked rides within these GPX files, but when read only 19 tracks are being counted. If parts of the track don't have time(stamp)s recorded, this option will try to extrapolate and add them where possible.



gpx files should be ingested, parsed & loaded into a polars dataframe identified by "track_id" which is the date of the start_timestamp. The Latitude and Longitude GPS data should be smoothed & any noise or errors filtered out (> 100 mph). Then the tracks can be smoothed and compressed. From here we will calculate analytics like total distance traveled, total trip duration, Steepness, turn angle, and Tortuosity will also be measured. 



Ultimately, the goal of this process is to answer details about "What Kind of Driver is this??"


-----



I am building a data report that profiles everytime I took my Polaris RZR vehicle riding. I have the location data (datetime_local, latitude, longitude, device_id) in the dataframe `jc_rzr_ALL`. A full schema is provided below. This report will highlight summary statistics about my riding habits that can be used to build a picture of what kind of rider I am. Do I ride mostly weekends? How many long-distance Trips, or well known RZR riding locations did I visit? What was my average speed, min and max speed? how many total miles traveled? Where is my home located (city, state). In this message, please assist in generating the following:

please demonstrate the best way to re-create the GitHub commit visualization, where annual timeline is boxed out by days & the intensity of the day represents how long the rides were that day. So in essence, a heatmap of all my rides. You can use the `calplot`, `plotly` or the `seaborn` library to create the heatmap. 

I would like to have a year-long heatmap, where each day is repesented by a single cell & the months are organized or formatted like they would be in the calendar. This should have a brighter color for longer rides, and text representing the total hours ridden should also be visualable on top. 

There should also be a "day of thw week" heatmap, where Monday - Sunday are labeled along the x-axis as categorical labels, and the color should be brigher for the day of the week that contains the most, and longest duration rides. This heatmap will illustrate of the rider is a weekend-warrior or weekday-rider. 

it would also be great to generate a folium `HeatMapWithTime` from the `jc_rzr_ALL` dataframe. 


Heatmap Example Code
```py
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import numpy as np

%config InlineBackend.figure_format = 'retina'
%matplotlib inline

style.use('ggplot')

fig = go.Figure(data=go.Heatmap(
                   z=[[1, None, 30, 50, 1], [20, 1, 60, 80, 30], [30, 60, 1, -10, 20]],
                   x=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
                   y=['Morning', 'Afternoon', 'Evening'],
                   hoverongaps = False))
fig.show()



np.random.seed(1)

programmers = ['Alex','Nicole','Sara','Etienne','Chelsea','Jody','Marianne']

base = datetime.datetime.today()
dates = base - np.arange(180) * datetime.timedelta(days=1)
z = np.random.poisson(size=(len(programmers), len(dates)))

fig = go.Figure(data=go.Heatmap(
        z=z,
        x=dates,
        y=programmers,
        colorscale='Viridis'))

fig.update_layout(
    title='GitHub commits per day',
    xaxis_nticks=36)

fig.show()

all_month_year_df = pd.pivot_table(df, values="precipitation",index=["month"],
                                   columns=["year"],
                                   fill_value=0,
                                   margins=True)
named_index = [[calendar.month_abbr[i] if isinstance(i, int) else i for i in list(all_month_year_df.index)]]
all_month_year_df = all_month_year_df.set_index(named_index)
all_month_year_df

all_month_year_percentage_df = pd.pivot_table(df, values="precipitation",index=["month"], columns=["year"],
                                              aggfunc=lambda x: (x>MIN_PRECIPITATION_MM_DRY).sum()/len(x),
                                              fill_value=0,
                                              margins=True)
all_month_year_percentage_df = all_month_year_percentage_df.set_index([[calendar.month_abbr[i] if isinstance(i, int)
                                                                        else i for i in list(all_month_year_percentage_df.index)]])
def plot_heatmap(df, title):
    plt.figure(figsize = (14, 10))
    ax = sns.heatmap(df, cmap='RdYlGn_r',
                     robust=True,
                     fmt='.2f', annot=True,
                     linewidths=.5, annot_kws={'size':11},
                     cbar_kws={'shrink':.8, 'label':'Precipitation (mm)'})
    
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)
    plt.title(title, fontdict={'fontsize':18}, pad=14);

plot_heatmap(all_month_year_df, 'Average Precipitations')

plt.figure(figsize = (14, 10))
ax = sns.heatmap(all_month_year_percentage_df, cmap = 'RdYlGn_r', annot=True, fmt='.0%',
                 vmin=0, vmax=1, linewidths=.5, annot_kws={"size": 16})
cbar = ax.collections[0].colorbar
cbar.set_ticks([0, .25, .50,.75, 1])
cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, fontsize = 14)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 0, fontsize = 14)
ax.tick_params(rotation = 0)
plt.title('Percentage of days in the month with rain', fontdict={'fontsize':18}, pad=14);
```

heatmap_with_time in Folium example
```py
import numpy as np

np.random.seed(3141592)
initial_data = np.random.normal(size=(100, 2)) * np.array([[1, 1]]) + np.array(
    [[48, 5]]
)

move_data = np.random.normal(size=(100, 2)) * 0.01

data = [(initial_data + move_data * i).tolist() for i in range(100)]

time_ = 0
N = len(data)
itensify_factor = 30
for time_entry in data:
    time_ = time_+1
    for row in time_entry:
        weight = min(np.random.uniform()*(time_/(N))*itensify_factor, 1)
        row.append(weight)

m = folium.Map([48.0, 5.0], zoom_start=6)

hm = folium.plugins.HeatMapWithTime(data, index=time_index, auto_play=True, max_opacity=0.3)

hm.add_to(m)

m
```

KeplerGL Config File
```json
config = {
  "version": "v1",
  "config": {
     "visState": {
        "filters": [
            {
              "dataId": ["my_data"],
              "id": "11111",
              "name": ["some_col_name"],
              "type": "multiSelect",
              "value": [],
              "enlarged": False,
              "plotType": "histogram",
              "animationWindow": "free",
              "yAxis": None,
              "speed": 1,
            }
        ],
        "layers": [
            {
              "id": "22222",
              "type": "point",
              "config": {
                  "dataId": "my_data",
                  "label": "my_data",
                  "color": [30, 150, 190],
                  "highlightColor": [252, 242, 26, 255],
                  "isVisible": True,
                  "visConfig": {
                      "radius": 5,
                      "fixedRadius": False,
                      "opacity": 0.8,
                      "outline": False,
                      "thickness": 2,
                      "strokeColor": None,
                       ...
                   },
                   "hidden": False
              }
           }  
         ],
         "interactionConfig": {
             "tooltip": {
                 "fieldsToShow": {
                     "my_data": [
                         {"name": "col_1", "format": None},
                         {"name": "col_2", "format": None}  
                    ]
                 },
                 "compareMode": False,
                 "compareType": "absolute",
                 "enabled": True,
             },
             "brush": {"size": 0.5, "enabled": False},
             "geocoder": {"enabled": False},
             "coordinate": {"enabled": False},
         },
         "layerBlending": "normal",
         "splitMaps": [],
         "animationConfig": {"currentTime": None, "speed": 1},
     },
     "mapState": {
         "bearing": 0,
         "dragRotate": False,
         "latitude": 40.710394,
         "longitude": -74.000288,
         "pitch": 0,
         "zoom": 12.41,
         "isSplit": False,
     },
     "mapStyle": {
         "styleType": "dark",
         "topLayerGroups": {},
         "visibleLayerGroups": {
             "label": True,
             "road": True,
             "border": False,
             "building": True,
             "water": True,
             "land": True,
             "3d building": False,
         },
         "threeDBuildingColor": [
             9.665468314072013,
             17.18305478057247,
             31.1442867897876,
         ],
         "mapStyles": {},
      },
  },
}
```

Basically, there are 3 main sections — visState, mapState and mapStyle.

MapState is used to define where on the world map you’ll start off every time you reload the page, the starting/default location.

MapStyles set the general theme of the KeplerGl application. It has few preset options available like dark, muted night, light, muted light, and satellite.

VizState is the more interesting one. It defines the layers (whether it’s point, hexagon, line etc ) and what dataset each layer to use — note that I put the dataId in bold, that is how you bind your data with the config, will show an example in the next section. You can also define filters that you can apply to one or more sets and base on one or more fields in that dataset. Additionally, here’s where you can define what you want to show in your tooltip (or other interactive option) and how.


<jc_rzr_ALL>
[
  {
    "name": "device_id",
    "mode": "NULLABLE",
    "type": "STRING",
    "description": null,
    "fields": []
  },
  {
    "name": "latitude",
    "mode": "NULLABLE",
    "type": "FLOAT",
    "description": null,
    "fields": []
  },
  {
    "name": "longitude",
    "mode": "NULLABLE",
    "type": "FLOAT",
    "description": null,
    "fields": []
  },
  {
    "name": "datetime_local",
    "mode": "NULLABLE",
    "type": "TIMESTAMP",
    "description": null,
    "fields": []
  },
  {
    "name": "min_of_day",
    "mode": "NULLABLE",
    "type": "INTEGER",
    "description": null,
    "fields": []
  },
  {
    "name": "hour_of_day",
    "mode": "NULLABLE",
    "type": "INTEGER",
    "description": null,
    "fields": []
  },
  {
    "name": "period_of_day",
    "mode": "NULLABLE",
    "type": "STRING",
    "description": null,
    "fields": []
  },
  {
    "name": "date",
    "mode": "NULLABLE",
    "type": "DATE",
    "description": null,
    "fields": []
  },
  {
    "name": "time_local",
    "mode": "NULLABLE",
    "type": "STRING",
    "description": null,
    "fields": []
  },
  {
    "name": "day_of_month",
    "mode": "NULLABLE",
    "type": "INTEGER",
    "description": null,
    "fields": []
  },
  {
    "name": "day_of_year",
    "mode": "NULLABLE",
    "type": "INTEGER",
    "description": null,
    "fields": []
  },
  {
    "name": "day_of_week",
    "mode": "NULLABLE",
    "type": "INTEGER",
    "description": null,
    "fields": []
  },
  {
    "name": "day_of_week_name",
    "mode": "NULLABLE",
    "type": "STRING",
    "description": null,
    "fields": []
  },
  {
    "name": "is_workday",
    "mode": "NULLABLE",
    "type": "BOOLEAN",
    "description": null,
    "fields": []
  },
  {
    "name": "is_weekend",
    "mode": "NULLABLE",
    "type": "BOOLEAN",
    "description": null,
    "fields": []
  },
  {
    "name": "is_business_hours",
    "mode": "NULLABLE",
    "type": "BOOLEAN",
    "description": null,
    "fields": []
  },
  {
    "name": "month_name",
    "mode": "NULLABLE",
    "type": "STRING",
    "description": null,
    "fields": []
  },
  {
    "name": "month",
    "mode": "NULLABLE",
    "type": "INTEGER",
    "description": null,
    "fields": []
  },
  {
    "name": "quarter",
    "mode": "NULLABLE",
    "type": "INTEGER",
    "description": null,
    "fields": []
  },
  {
    "name": "datetime_index",
    "mode": "NULLABLE",
    "type": "TIMESTAMP",
    "description": null,
    "fields": []
  },
  {
    "name": "move_activity",
    "mode": "NULLABLE",
    "type": "STRING",
    "description": null,
    "fields": []
  },
  {
    "name": "stay_activity",
    "mode": "NULLABLE",
    "type": "STRING",
    "description": null,
    "fields": []
  },
  {
    "name": "altitude1_minOverlap",
    "mode": "NULLABLE",
    "type": "INTEGER",
    "description": null,
    "fields": []
  },
  {
    "name": "altitude2_hourOverlap",
    "mode": "NULLABLE",
    "type": "INTEGER",
    "description": null,
    "fields": []
  },
  {
    "name": "altitude3_min2max",
    "mode": "NULLABLE",
    "type": "FLOAT",
    "description": null,
    "fields": []
  },
  {
    "name": "h3_lvl10_index",
    "mode": "NULLABLE",
    "type": "STRING",
    "description": null,
    "fields": []
  },
  {
    "name": "h3_lvl4_index",
    "mode": "NULLABLE",
    "type": "STRING",
    "description": null,
    "fields": []
  },
  {
    "name": "timestamp_unix",
    "mode": "NULLABLE",
    "type": "FLOAT",
    "description": null,
    "fields": []
  },
  {
    "name": "timestamp_utc",
    "mode": "NULLABLE",
    "type": "TIMESTAMP",
    "description": null,
    "fields": []
  },
  {
    "name": "timezone",
    "mode": "NULLABLE",
    "type": "STRING",
    "description": null,
    "fields": []
  },
  {
    "name": "trajectory_id",
    "mode": "NULLABLE",
    "type": "STRING",
    "description": null,
    "fields": []
  },
  {
    "name": "source",
    "mode": "NULLABLE",
    "type": "STRING",
    "description": null,
    "fields": []
  },
  {
    "name": "last_modified_on",
    "mode": "NULLABLE",
    "type": "DATE",
    "description": null,
    "fields": []
  }
]
</jc_rzr_ALL>



---

we need to calculate heading and avg & max speed, compare to values in original GPX files. in MPH.
- distance in miles
- elevation chagne in ft
- min elev in ft, max in ft
- duration moving
- duration stopped
- 