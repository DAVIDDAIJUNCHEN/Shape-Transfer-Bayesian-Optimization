#import non-ML dependencies
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sqlalchemy import create_engine

#set target variable
target = ['average_price']

pwd="123"
username="postgres"
db_name="tornodo"
host="localhost"

# Step 1: connect to database and download data for analysis
postgres_str = ('postgresql://{username}:{password}@{ipaddress}/{dbname}'.format(username=username,password=pwd,ipaddress=host,dbname=db_name))
conn = create_engine(postgres_str)

df_all = pd.read_sql('Select * from home_prices',con=conn)
recession = pd.read_sql('Select * from recession_indicator',con=conn)
mortgage_rates = pd.read_sql('Select * from interest_rate',con=conn)
inflation = pd.read_sql('Select * from inflation',con=conn)

# Step 2: 
#convert the '_date' field to pandas date type period
df_all._date = df_all._date.apply(lambda x: pd.Period(x))
#set index to date
df_all.set_index('_date',inplace=True)

#filter dataset to toronto only, require avg price to >0, and take only community/date groups that have
#more than 30 quarters of data available
df_all_toronto = df_all.loc[df_all.area=='Toronto',:]
df_all_toronto = df_all_toronto.loc[df_all_toronto['average_price']>0,:]

#to get community/buildingtypes with more that 30 data points, we group and use the size attribute
element_group_sizes = df_all_toronto.groupby(['community','building_type']).size()>30
#select only those where size>30 was true
element_group_sizes=element_group_sizes[element_group_sizes==1]

#sized groups
grps=tuple(zip(element_group_sizes.reset_index().iloc[:,0].to_list(),element_group_sizes.reset_index().iloc[:,1].to_list()))

#build new dataframe, toronto only, from groups created above
df_all_toronto_clean=pd.concat([df_all_toronto.groupby(['community','building_type']).get_group(x) for x in grps])

#sort for time series analysis (so each group of community/building type is ordered by time
df_all_toronto_clean.sort_values(by=['community','building_type','_year','quarter'],inplace=True)
df_all_toronto_clean.drop(['area','municipality', 'dollar_volume','_no'],inplace=True,axis=1)

#create a time index with all periods from data start to end, ultimately to ensure there are no gaps in the series
all_qtrs=pd.period_range(df_all_toronto_clean.index.get_level_values(level=0).min(),df_all_toronto_clean.index.get_level_values(level=0).max())
idx=pd.DataFrame(index=all_qtrs,columns=df_all_toronto_clean.columns)


#here we gather missing dates by group. not used later, but kept in case of future need
h=[]
for x in grps:
    g = df_all_toronto_clean.groupby(['community', 'building_type']).get_group(x)

    h.append(pd.DataFrame(index=[x for x in all_qtrs if x not in  g.index.to_list()],columns=x))
missing=pd.concat([x for x in h])

#here we rebuild the df_all_toronto_clean frame again, and reindex using the above created idx, which has all periods...
#between data start and end, and use ffill to forward fill missing points
#improvement here would be to only ffill the average_price, as we have full quarterly data series for exogenous vars
df_all_toronto_clean=pd.concat([df_all_toronto_clean.groupby(['community','building_type']).get_group(x).\
    reset_index().\
    set_index('_date').\
    sort_index().\
    reindex_like(idx).\
    ffill() for x in grps])

#correcting the year_quarter_key following the reindexing and ffill procedure
df_all_toronto_clean.assign(year_quarter_key=\
                                df_all_toronto_clean.index.year*10+df_all_toronto_clean.index.quarter,\
                            _year=df_all_toronto_clean.index.year,\
                            quarter=df_all_toronto_clean.index.quarter,inplace=True)

#preparing to merge price data with exogenous vars...not actually necessary in retrospect...
#did this to work with facebook prophet
#pd.merge does not keep index when not merging on it I think (might be pandas bug), so keeping index to add back at end
idx = df_all_toronto_clean.index
idx.name='_date'
df_all_toronto_clean_with_rates=df_all_toronto_clean.\
    merge(mortgage_rates,right_on='year_quarter_key',left_on='year_quarter_key',how='left')
#df_all_toronto_clean_with_rates=df_all_toronto_clean_with_rates.join(inflation_q['CPI_TRIM'],how='inner')

df_all_toronto_clean_with_rates=df_all_toronto_clean_with_rates.\
    merge(inflation,right_on='year_quarter_key',left_on='year_quarter_key',how='left')
#df_all_toronto_clean_with_rates=df_all_toronto_clean_with_rates.join(recession_q['CANRECDM'],how='inner')
df_all_toronto_clean_with_rates=df_all_toronto_clean_with_rates.\
    merge(recession[['year_quarter_key','canrecdm']],right_on='year_quarter_key',left_on='year_quarter_key',how='left')

#add back index after merge
df_all_toronto_clean_with_rates.set_index(idx,inplace=True)

#split data for train/test
#since time series, can't use random split, instead picked a cut off date;
#using large portion of data for training, so as to aviod having to forecast too far out from training period

X = df_all_toronto_clean_with_rates.reset_index().set_index(['community','building_type','_date'])[['average_price','avg_five_year_rates','cpi_trim','canrecdm']]
X.sort_index(inplace=True)
X=X.dropna()
y = X['average_price'].to_frame()
X.drop('average_price',inplace=True,axis=1)
cutoff='2018Q4'
y_test=y.loc[(y.index.get_level_values('_date')>cutoff)]
X_test=X.loc[(X.index.get_level_values('_date')>cutoff)]
X_train=X.loc[(X.index.get_level_values('_date')<=cutoff)]
y_train=y.loc[(y.index.get_level_values('_date')<=cutoff)]

# output pandas data frame to csv
X_train.to_csv("Tornodo_X.csv", index=False)
y_train.to_csv("Tornodo_Y.csv", index=False)
