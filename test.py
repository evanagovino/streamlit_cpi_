import streamlit as st
import pandas as pd
#import matplotlib.pyplot as plt
import plotly.express as px
import streamlit.components.v1 as components
from jinja2 import Template
#import _functions as functions
from datetime import datetime, timedelta
import requests
import json
import copy

st.set_page_config(layout="wide")

#pd.options.display.float_format = '{:.2%}'.format

period_translation = {
    'M01': 'January',
    'M02': 'February',
    'M03': 'March',
    'M04': 'April',
    'M05': 'May',
    'M06': 'June',
    'M07': 'July',
    'M08': 'August',
    'M09': 'September',
    'M10': 'October',
    'M11': 'November',
    'M12': 'December',
    'S01': 'June',
    'S02': 'December'    
}

categories = ['All items', 'Food and beverages', 'Housing', 'Apparel', 'Transportation', 'Medical care', 'Recreation', 'Education and communication', 'Other goods and services']

def clean_df(target_url):
    response = requests.get(target_url)
    df = pd.DataFrame([i.split('\t') for i in response.content.decode('utf-8').split('\n')])
    for col in df.columns:
        df[col] = df[col].str.strip()
    df.columns = df.iloc[0]
    df = df.iloc[1:, :-1].reset_index(drop=True)
    return df.dropna(axis=0)

def adjust_prices(temp):
    temp['value'] = temp['value'].astype('float')
    temp['adjusted_value'] = temp['value']
    mask = temp[temp['base_period'] == '1987=100'].index
    temp['adjusted_value'][mask] = temp['adjusted_value'][mask] * 1.137
    mask = temp[temp['base_period'] == 'DECEMBER 1997=100'].index
    temp['adjusted_value'][mask] = temp['adjusted_value'][mask] * 1.6017
    mask = temp[(temp['base_period'] == 'DECEMBER 2001=100') & (temp['item_name'] == 'All items')].index
    temp['adjusted_value'][mask] = temp['adjusted_value'][mask] * 1.7862
    mask = temp[(temp['base_period'] == 'DECEMBER 2001=100') & (temp['item_name'] == 'Apparel')].index
    temp['adjusted_value'][mask] = temp['adjusted_value'][mask] * 1.2393
    mask = temp[(temp['base_period'] == 'DECEMBER 2001=100') & (temp['item_name'] == 'Food and beverages')].index
    temp['adjusted_value'][mask] = temp['adjusted_value'][mask] * 1.7592
    mask = temp[(temp['base_period'] == 'DECEMBER 2001=100') & (temp['item_name'] == 'Housing')].index
    temp['adjusted_value'][mask] = temp['adjusted_value'][mask] * 1.7911
    mask = temp[(temp['base_period'] == 'DECEMBER 2001=100') & (temp['item_name'] == 'Medical care')].index
    temp['adjusted_value'][mask] = temp['adjusted_value'][mask] * 2.7502
    mask = temp[(temp['base_period'] == 'DECEMBER 2001=100') & (temp['item_name'] == 'Other goods and services')].index
    temp['adjusted_value'][mask] = temp['adjusted_value'][mask] * 2.7606
    mask = temp[(temp['base_period'] == 'DECEMBER 2001=100') & (temp['item_name'] == 'Transportation')].index
    temp['adjusted_value'][mask] = temp['adjusted_value'][mask] * 1.5493
    mask = temp[(temp['base_period'] == 'DECEMBER 2001=100') & (temp['item_name'] == 'Education and communication')].index
    temp['adjusted_value'][mask] = temp['adjusted_value'][mask] * 1.0631 * 1.6017
    mask = temp[(temp['base_period'] == 'DECEMBER 2001=100') & (temp['item_name'] == 'Recreation')].index
    temp['adjusted_value'][mask] = temp['adjusted_value'][mask] * 1.0582 * 1.6017
    mask = temp[(temp['base_period'] == 'DECEMBER 2017=100') & (temp['item_name'] == 'All items')].index
    temp['adjusted_value'][mask] = temp['adjusted_value'][mask] * 2.4925
    mask = temp[(temp['base_period'] == 'DECEMBER 2017=100') & (temp['item_name'] == 'Apparel')].index
    temp['adjusted_value'][mask] = temp['adjusted_value'][mask] * 1.2774
    mask = temp[(temp['base_period'] == 'DECEMBER 2017=100') & (temp['item_name'] == 'Food and beverages')].index
    temp['adjusted_value'][mask] = temp['adjusted_value'][mask] * 2.5057
    mask = temp[(temp['base_period'] == 'DECEMBER 2017=100') & (temp['item_name'] == 'Housing')].index
    temp['adjusted_value'][mask] = temp['adjusted_value'][mask] * 2.6123
    mask = temp[(temp['base_period'] == 'DECEMBER 2017=100') & (temp['item_name'] == 'Medical care')].index
    temp['adjusted_value'][mask] = temp['adjusted_value'][mask] * 4.7661
    mask = temp[(temp['base_period'] == 'DECEMBER 2017=100') & (temp['item_name'] == 'Other goods and services')].index
    temp['adjusted_value'][mask] = temp['adjusted_value'][mask] * 4.1036
    mask = temp[(temp['base_period'] == 'DECEMBER 2017=100') & (temp['item_name'] == 'Transportation')].index
    temp['adjusted_value'][mask] = temp['adjusted_value'][mask] * 2.0274
    mask = temp[(temp['base_period'] == 'DECEMBER 2017=100') & (temp['item_name'] == 'Education and communication')].index
    temp['adjusted_value'][mask] = temp['adjusted_value'][mask] * 1.3881 * 1.6017
    mask = temp[(temp['base_period'] == 'DECEMBER 2017=100') & (temp['item_name'] == 'Recreation')].index
    temp['adjusted_value'][mask] = temp['adjusted_value'][mask] * 1.1539 * 1.6017
    return temp

def fetch_and_clean_data():
    df = clean_df('https://download.bls.gov/pub/time.series/cu/cu.data.0.Current')
    items = clean_df('https://download.bls.gov/pub/time.series/cu/cu.item')
    series = clean_df('https://download.bls.gov/pub/time.series/cu/cu.series')
    area = clean_df('https://download.bls.gov/pub/time.series/cu/cu.area')
    series = series.merge(area, on='area_code').merge(items, on='item_code')
    df = df.merge(series, on='series_id')
    area_names = ['East South Central',
                 'Pacific',
                 'Mountain',
                 'Phoenix-Mesa-Scottsdale, AZ',
                 'West South Central',
                 'San Francisco-Oakland-Hayward, CA',
                 'Los Angeles-Long Beach-Anaheim, CA',
                 'Riverside-San Bernardino-Ontario, CA',
                 'San Diego-Carlsbad, CA',
                 'Denver-Aurora-Lakewood, CO',
                 'New England',
                 'Philadelphia-Camden-Wilmington, PA-NJ-DE-MD',
                 'Middle Atlantic',
                 'South Atlantic',
                 'Miami-Fort Lauderdale-West Palm Beach, FL',
                 'Tampa-St. Petersburg-Clearwater, FL',
                 'Atlanta-Sandy Springs-Roswell, GA',
                 'East North Central',
                 'St. Louis, MO-IL',
                 'Chicago-Naperville-Elgin, IL-IN-WI',
                 'West North Central',
                 'Baltimore-Columbia-Towson, MD',
                 'Washington-Arlington-Alexandria, DC-VA-MD-WV',
                 'Boston-Cambridge-Newton, MA-NH',
                 'Detroit-Warren-Dearborn, MI',
                 'Minneapolis-St.Paul-Bloomington, MN-WI',
                 'New York-Newark-Jersey City, NY-NJ-PA',
                 'Houston-The Woodlands-Sugar Land, TX',
                 'Dallas-Fort Worth-Arlington, TX',
                 'Seattle-Tacoma-Bellevue WA',
                 'U.S. city average']
    item_names = ['All items', 
                  'Apparel', 
                  'Food and beverages', 
                  'Housing', 
                  'Medical care', 
                  'Other goods and services', 
                  'Transportation', 
                  'Education and communication', 
                  'Recreation']
    df= df[(df['area_name'].isin(area_names)) & (df['item_name'].isin(item_names))].reset_index(drop=True)
    df = adjust_prices(df)
    return df


class Category_Weights:

    def __init__(self):
        self.baseline_weights = {
        'Food and beverages': 0.14259,
        'Housing': 0.42363,
        'Apparel': 0.02458,
        'Transportation': 0.18182,
        'Medical care': 0.08487,
        'Recreation': 0.05108,
        'Education and communication': 0.06406,
        'Other goods and services': 0.02737
        }
        self.reset_weights()

    def update_value(self, category, value):
        self.category_weights[category] = value

    def reset_value(self, category):
        self.category_weights[category] = self.baseline_weights[category]

    def reset_weights(self):
        self.category_weights = copy.deepcopy(self.baseline_weights)

def f(x):
    return datetime.strptime(x['period'] + str(x['year']), '%B%Y')

def growth_diff(temp, date_input, days=180):
    date_input = datetime.fromtimestamp(temp[temp.index <= date_input].index.max().timestamp())
    day_difference = (temp[temp.index >= date_input - timedelta(days=days)].index[-1] - temp[temp.index >= date_input - timedelta(days=days)].index[0]).days
    original_price = temp[temp.index >= date_input - timedelta(days=days)]['value'][0]
    price_difference = temp[temp.index >= date_input - timedelta(days=days)]['value'][-1] - temp[temp.index >= date_input - timedelta(days=days)]['value'][0]
    return price_difference / original_price / (day_difference / 365)

class Data:
    
    def __init__(self, df):
        self.df = df
        self.area_name = self.df['area_name'][0]
        
    def process_df_for_graph(self, item_name = "All items"):
        temp = self.df[self.df['item_name'] == item_name].reset_index(drop=True)
        temp['value'] = temp['value'].astype('float')
        temp = temp[~temp['period'].isin(['M13', 'S03'])].reset_index(drop=True)
        temp['period'] = temp['period'].map(period_translation)
        temp['timeframe'] = temp.apply(lambda x: f(x), axis=1)
        temp.index = temp['timeframe']
        temp = temp.sort_index()
        return temp[temp.index > (datetime.now() - timedelta(days=365*10))]

    def get_weighted_value(self, category_weights, date_input=datetime.now(), days=365):
        temp = self.return_values(category_weights, days=days)
        temp['weights'] = temp['category'].map(category_weights)
        return temp['weights'] @ temp['inflation']

    def return_values(self, category_weights, date_input=datetime.now(), days=365):
        l = []
        for category in category_weights:
            l.append([self.area_name, category, days, growth_diff(getattr(self, clean_string(category)), date_input=date_input, days=days), self.return_latest_value(category)])
        return pd.DataFrame(l, columns=['location', 'category', 'days', 'inflation', 'latest_price'])

    def return_latest_value(self, category):
        return getattr(self, clean_string(category))['adjusted_value'][-1]

def clean_string(x):
    return x.lower().replace(' ','_')

def initiate_instance(df, categories, area_name = 'U.S. city average'):
    _ = df[df['area_name'] == area_name].reset_index(drop=True)
    x = Data(_)
    for j in categories:
        string_name = j.lower().replace(' ','_')
        setattr(x, string_name, x.process_df_for_graph(j))
    return x

# def update_from_area():
#     if 'area' in st.session_state:
#         del st.session_state['area']

def update_from_zip_code(zip_code):
    if 'zip_code' in st.session_state:
        if int(zip_code) in st.session_state.zip_code_fips_match.zip_code.unique():
            del st.session_state['zip_code']

def update_category():
    if 'category' in st.session_state:
        del st.session_state['category']

def inflation_comparison(temp, days=365):
    return [(timeframe, growth_diff(temp, date_input=timeframe, days=365)) for timeframe in temp.index]

def price_comparison(temp, days=365):
    return temp.groupby(temp.index)['adjusted_value'].mean()

def _build_metric(label, value, percentile):
    #print('percentile', percentile)
    html_text = """
    <style>
    .metric {
       font-family: sans-serif;
       text-align: center;
    }
    .metric .value {
       font-size: 30px;
       line-height: 1.6;
    }
    .metric .label {
       letter-spacing: 2px;
       font-size: 14px;
       text-transform: uppercase;
    }
    </style>
    <div class="metric">
       <div class="label">
          {{ label }}
       </div>
       <div class="value">
          {{ value }}%
       </div>
       <div class="label">
          {{ percentile }}%
       </div>
    </div>
    """
    html = Template(html_text)
    return html.render(label=label, value=value, percentile=percentile)

#Process Data
if 'df' not in st.session_state:
    #st.session_state.df = pd.read_csv('df.csv')
    st.session_state.df = fetch_and_clean_data()
    st.session_state.national_classification = initiate_instance(st.session_state.df, categories, area_name = 'U.S. city average')

if 'zip_code_fips_match' not in st.session_state:
    st.session_state.zip_code_fips_match = pd.read_csv('zip_fips_match.csv')
    st.session_state.zip_code_fips_match['STCOUNTYFP'] = st.session_state.zip_code_fips_match['STCOUNTYFP'].astype('string')
    st.session_state.zip_code_fips_match['STCOUNTYFP'] = [i if len(i) == 5 else '0' + i for i in st.session_state.zip_code_fips_match['STCOUNTYFP']]
    st.session_state.zip_code_fips_match['classification'] = st.session_state.zip_code_fips_match['local_classification'].fillna(st.session_state.zip_code_fips_match['regional_classification'])
    st.session_state.all_classifications = list(st.session_state.zip_code_fips_match['regional_classification'].unique()) + list(st.session_state.zip_code_fips_match['local_classification'].unique())
    response = requests.get('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json')
    st.session_state.countries = json.loads(response.content)

if 'geo_data' not in st.session_state:
    #category = [categories]
    temp = st.session_state.zip_code_fips_match.groupby(['STCOUNTYFP', 'classification'])['drop_column'].count().reset_index()
    l = []
    for classification in temp['classification'].unique():
        #print(category, classification)
        _ = initiate_instance(st.session_state.df, categories, area_name = classification)
        #_ = _.return_values([category], days=365)
        temp_df = _.return_values(categories)
        #inflation = _.return_values(categories)['inflation'].values[0]
        #latest_price = _.return_values([category])['latest_price'].values[0]
        _ = temp[temp['classification'] == classification].reset_index(drop=True).merge(temp_df, how='left', left_on='classification', right_on='location')
        #_['inflation'] = inflation
        #_['latest_price'] = latest_price
        #_['FIPS'] = temp[temp['classification'] == classification]['STCOUNTYFP'].values[0]
        l.append(_)
    st.session_state.geo_data = pd.concat(l)
    #print(st.session_state.geo_data.head(), st.session_state.geo_data.dtypes)



if 'category_weights' not in st.session_state:
    st.session_state.category_weights = Category_Weights()

def above_mean(col):
  is_above = col > 0
  return ['background-color: red' if v else 'background-color: green' for v in is_above]

# def show_map():
#     pass

#area_names = ['Pacific','East South Central','West South Central','Mountain','Phoenix-Mesa-Scottsdale, AZ','San Francisco-Oakland-Hayward, CA','Los Angeles-Long Beach-Anaheim, CA','Riverside-San Bernardino-Ontario, CA','San Diego-Carlsbad, CA','Denver-Aurora-Lakewood, CO','New England','Washington-Arlington-Alexandria, DC-VA-MD-WV','Philadelphia-Camden-Wilmington, PA-NJ-DE-MD','Middle Atlantic','South Atlantic','Miami-Fort Lauderdale-West Palm Beach, FL','Tampa-St.Petersburg-Clearwater, FL','Atlanta-Sandy Springs-Roswell, GA','West North Central','East North Central','St. Louis, MO-IL','Chicago-Naperville-Elgin, IL-IN-WI','Boston-Cambridge-Newton, MA-NH','Baltimore-Columbia-Towson, MD','Detroit-Warren-Dearborn, MI','Minneapolis-St.Paul-Bloomington, MN-WI','New York-Newark-Jersey City, NY-NJ-PA','Houston-The Woodlands-Sugar Land, TX','Dallas-Fort Worth-Arlington, TX','Seattle-Tacoma-Bellevue WA']

@st.cache(suppress_st_warning=True)
def show_map(category, variable='inflation'):
    print('running', datetime.now())
    temp = st.session_state.geo_data[st.session_state.geo_data['category'] == category]
    fig = px.choropleth(temp, geojson=st.session_state.countries, locations='STCOUNTYFP', color=variable, scope='usa', hover_data=['classification', variable, 'location'], color_continuous_scale="YlOrRd")
    fig.update_layout(coloraxis_colorbar_title=variable.capitalize())
    return fig
    #st.plotly_chart(fig, use_container_width=True)

def update_from_zip_code(zip_code):
  if int(zip_code) in st.session_state.zip_code_fips_match.zip_code.unique():
      if 'zip_code' in st.session_state:
          del st.session_state['zip_code']
      st.session_state['zip_code'] = int(zip_code)
      st.session_state['county_name'] = st.session_state.zip_code_fips_match[st.session_state.zip_code_fips_match['zip_code'] == st.session_state['zip_code']]['COUNTYNAME'].values[0]
      st.session_state['state'] = st.session_state.zip_code_fips_match[st.session_state.zip_code_fips_match['zip_code'] == st.session_state['zip_code']]['State'].values[0]
      st.sidebar.subheader(f'Zip Code: {st.session_state.zip_code}')
      st.sidebar.subheader(f'County Name: {st.session_state.county_name}')
      st.sidebar.subheader(f'State: {st.session_state.state}')
      if len(st.session_state.zip_code_fips_match[st.session_state.zip_code_fips_match['zip_code'] == st.session_state['zip_code']]) > 0:
          local_classification = st.session_state.zip_code_fips_match[st.session_state.zip_code_fips_match['zip_code'] == st.session_state['zip_code']]['local_classification'].values[0]
          regional_classification = st.session_state.zip_code_fips_match[st.session_state.zip_code_fips_match['zip_code'] == st.session_state['zip_code']]['regional_classification'].values[0]
          if local_classification == local_classification:
              st.session_state['local_classification'] = initiate_instance(st.session_state.df, categories, area_name=local_classification)
              st.sidebar.subheader(f"Local Area: {local_classification}")
          if regional_classification == regional_classification:
              st.session_state['regional_classification'] = initiate_instance(st.session_state.df, categories, area_name=regional_classification)
              st.sidebar.subheader(f"Regional Area: {regional_classification}")
  else:
      st.sidebar.write(f'{zip_code} not in database')

def transform_df_for_comparison(df, local=None, regional=None, value='inflation'):
    df = df.pivot(index='category',
                  columns='location',
                  values=value)
    df = df.rename(columns={'U.S. city average': 'National'})
    column_order = []
    if local:
        df = df.rename(columns={local: 'Local'})
        column_order.append('Local')
        df['Difference'] = df['Local'] - df['National']
    if regional:
        df = df.rename(columns={regional: 'Regional'})
        column_order.append('Regional')
        df['Difference'] = df['Regional'] - df['National']
    column_order.append('National')
    if 'Difference' in df.columns:
        column_order.append('Difference')
    return df[column_order]

def transform_df_for_bar_graph(classification, category_weights):
    comparison_graph = []
    for category in category_weights:
        _ = pd.DataFrame(inflation_comparison(getattr(st.session_state[classification], clean_string(category))), columns=['timeframe', 'inflation'])
        _['category'] = category
        comparison_graph.append(_)
    _ = pd.concat(comparison_graph)
    _['inflation_part'] = _['category'].map(st.session_state.category_weights.category_weights) * _['inflation']
    _ = _.groupby(['timeframe','category'])[['inflation_part', 'inflation']].mean().reset_index()
    _['%_of_inflation'] = _['inflation_part'] / _['inflation']
    return _


#Update Zip Code
zip_code = st.sidebar.text_input("Enter Zip Code", placeholder='10001')
if st.sidebar.button('Pick Random Zip Code'):
    zip_code = st.session_state.zip_code_fips_match['zip_code'].sample().values[0]
if zip_code:
    update_from_zip_code(zip_code)

checkbox = st.sidebar.checkbox('CPI Basket Customization')
if checkbox:
    st.sidebar.subheader('CPI Customization')
    drive = st.sidebar.checkbox('Do you drive?', value=True)
    if drive == False:
        st.session_state.category_weights.update_value("Transportation", 0.01)
    else:
        st.session_state.category_weights.reset_value("Transportation")
    for category in st.session_state.category_weights.category_weights:
        val = st.sidebar.slider(category, 0.0, 1.0, value=st.session_state.category_weights.category_weights[category])
        if val:
            st.session_state.category_weights.update_value(category, val)
if 'county_name' in st.session_state:
    st.header(f'{st.session_state.county_name} Inflation Calculator', anchor='custom-inflation-calculator')
else:
    st.header('Inflation Calculator', anchor='custom-inflation-calculator')
if 'national_classification' in st.session_state:
    periods = [
    '6 Mt. Inflation',
    '1 Yr. Inflation',
    '2 Yrs. Inflation'
    ]
    columns = st.columns(3)
    for position, value in enumerate([180, 365, 730]):
        with columns[position]:
            baseline_value = st.session_state['national_classification'].get_weighted_value(st.session_state.category_weights.baseline_weights, days=value)
            if 'local_classification' in st.session_state:
                new_value = st.session_state['local_classification'].get_weighted_value(st.session_state.category_weights.category_weights, days=value)
            elif 'regional_classification' in st.session_state:
                new_value = st.session_state['regional_classification'].get_weighted_value(st.session_state.category_weights.category_weights, days=value)
            else:
                new_value = st.session_state['national_classification'].get_weighted_value(st.session_state.category_weights.baseline_weights, days=value)
            value = _build_metric(label=periods[position],
                                  value=round(new_value * 100, 2),
                                  percentile=round(baseline_value * 100, 2)
                                  )
            components.html(value)
    df_list = []
    if 'local_classification' in st.session_state:
        df_list.append(st.session_state['local_classification'].return_values(st.session_state.category_weights.category_weights))
    if 'regional_classification' in st.session_state:
        df_list.append(st.session_state['regional_classification'].return_values(st.session_state.category_weights.category_weights))
    df_list.append(st.session_state['national_classification'].return_values(st.session_state.category_weights.category_weights))
    df_ = pd.concat(df_list)
    columns = st.columns(2)
    with columns[0]:
        if 'local_classification' in st.session_state:
            df = transform_df_for_comparison(df_, local=st.session_state.local_classification.area_name, regional=st.session_state.regional_classification.area_name, value='inflation')
        elif 'regional_classification' in st.session_state:
            df = transform_df_for_comparison(df_, regional=st.session_state.regional_classification.area_name, value='inflation')
        else:
            df = transform_df_for_comparison(df_, value='inflation')
        st.subheader('Inflation Comparison by Category')
        if 'Difference' in df.columns:
            st.dataframe(df.style.format("{:.2%}").bar(align='mid', color=['green', 'red'], subset=['Difference']))
        else:
            st.dataframe(df.style.format("{:.2%}"))
    with columns[1]:
        if 'local_classification' in st.session_state:
            df = transform_df_for_comparison(df_, local=st.session_state.local_classification.area_name, regional=st.session_state.regional_classification.area_name, value='latest_price')
        elif 'regional_classification' in st.session_state:
            df = transform_df_for_comparison(df_, regional=st.session_state.regional_classification.area_name, value='latest_price')
        else:
            df = transform_df_for_comparison(df_, value='latest_price')
        st.subheader('Price Comparison by Category')
        if 'Difference' in df.columns:
            st.write(df.style.format("${:.2f}").bar(align='mid', color=['green', 'red'], subset=['Difference']))
        else:
            st.dataframe(df.style.format("${:.2f}"))
    st.subheader('Local Inflation Drivers')
    if 'zip_code' in st.session_state:
        if 'local_classification' in st.session_state:
            df = transform_df_for_bar_graph('local_classification', st.session_state.category_weights.category_weights)
        elif 'regional_classification' in st.session_state:
            df = transform_df_for_bar_graph('regional_classification', st.session_state.category_weights.category_weights)
        print(df.head())
        fig = px.bar(
              data_frame=df,
              x='timeframe',
              y='inflation_part',
              hover_data=['inflation'],
              color='category',
              barmode='stack'
              )
        fig.update_layout(xaxis_title=None, yaxis_title='Inflation', legend_title_text=None)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write('Select zip code to see local inflation drivers.')
    #Geo-Anlaysis
    category = st.selectbox('Pick a category for geo-analysis', categories, on_change=update_category)
    if category not in st.session_state:
        st.session_state['category'] = category
    columns = st.columns(2)
    with columns[0]:
        st.subheader('Regional Inflation Comparison')
        fig = show_map(st.session_state.category, 'inflation')
        st.plotly_chart(fig, use_container_width=True)
    with columns[1]:
        st.subheader('Regional Price Comparison')
        fig = show_map(st.session_state.category, 'latest_price')
        st.plotly_chart(fig, use_container_width=True)
st.subheader('Methodology')

markdown_text = """
##### Data Sources
All data come directly from [BLS](https://download.bls.gov/pub/time.series/cu/cu.data.0.Current).

Component weights for CPI items are pulled from the [U.S. city average for December 2021](https://www.bls.gov/cpi/tables/relative-importance/2021.htm).

County-level FIPS code to Zip-Code matching provided by [this dataset](https://www.kaggle.com/danofer/zipcodes-county-fips-crosswalk).

##### Census Divisions

The country is broken up into nine census divisions and individual metro areas where possible. See the BLS [regional resources page](https://www.bls.gov/cpi/regional-resources.htm) for details.

##### CPI Index

This data uses the eight major categories of household items. See the BLS [CPI index page](https://www.bls.gov/opub/hom/cpi/concepts.htm#cpi-index-values) for details.
https://www.bls.gov/opub/hom/cpi/concepts.htm#cpi-index-values

##### Price Methodology

The [BLS data source](https://download.bls.gov/pub/time.series/cu/cu.data.0.Current) uses price values pegged to different time frames. All data used here are readjusted so that **1984 = 100** for all price values. Note that all prices shown are relative to this value.

##### Data Notes

All inflation rates are annualized one-year rates unless otherwise noted.

Data are **not** seasonally adjusted.




"""
st.markdown(markdown_text)





title_alignment = """
<style>
#custom-inflation-calculator {
  text-align: center
}

#inflation-comparison-by-category {
  text-align: center
}

#price-comparison-by-category {
  text-align: center
}

#local-inflation-drivers  {
  text-align: center
}

#regional-inflation-comparison  {
  text-align: center
}

#regional-price-comparison  {
  text-align: center
}

#methodology  {
  text-align: center
}

#T_16f55_ {
    margin: auto
}

</style>
"""

st.markdown(title_alignment, unsafe_allow_html=True)


