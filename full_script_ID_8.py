############ TEAM 8 ###########

# Importing the necessary libraries that you need
import os
import pandas as pd
from datetime import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

plt.style.use('ggplot')

startTime = time.time()


###### NEED TO CHANGE PATH!!! ######
PATH = 'drive/MyDrive/Credit Line Increase Competition21/data'

data_paths = os.listdir(PATH)
os.chdir(PATH)

# FUNCTIONS
##########################################################################

def missing_zero_values_table(df):
  """ Will find NA and zero value statistics for dataframe
  
  :df: any pandas dataframe
  
  """
  zero_val = (df == 0.00).astype(int).sum(axis=0)
  mis_val = df.isnull().sum()
  mis_val_percent = 100 * df.isnull().sum() / len(df)
  mz_table = pd.concat([zero_val, mis_val, mis_val_percent], axis=1)
  mz_table = mz_table.rename(
  columns = {0 : 'Zero Values', 1 : 'Missing Values', 2 : '% of Total Values'})
  mz_table['Total Zero Missing Values'] = mz_table['Zero Values'] + mz_table['Missing Values']
  mz_table['% Total Zero Missing Values'] = 100 * mz_table['Total Zero Missing Values'] / len(df)
  mz_table['Data Type'] = df.dtypes
  mz_table = mz_table[mz_table.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
  print ("Your selected dataframe has " + str(df.shape[1]) + " columns and " 
               + str(df.shape[0]) + " Rows.\nThere are " + str(mz_table.shape[0]) +
              " columns that have missing values.")
  return mz_table

def combine_tables(unsecured, credit_scores, pscu, rates):
  """ Will run our full preprocessing procedure and join the given tables

  :unsecured_path: 'ucf_unsecured_debt.csv'
  :pscu_path: 'ucf_PSCU_UCF_2021_Data.csv'
  :credit_score_path: '20211104_APPEND_ucf_bureau_score_history.csv'
  :rates_path: '20211104_APPEND_ucf_rates_history.csv'

  """

  # Obtaining each user's most recent rates
  rates = rates.sort_values(['ProcessDate', 'credit_line_Rate'], ascending=False).groupby('ucfID').first().drop(['ProcessDate'], axis = 1)
  rates['credit_line_Rate'] /= 1000

  # Unsecured table pre-processing, taking sum of each 
  unsecured = unsecured.drop_duplicates()
  unsecured = unsecured.fillna(0)
  unsecured = unsecured.groupby('ucfID').sum()

  # Credit Score History table pre-processing
  credit_scores.drop_duplicates()
  credit_scores['credit_bureau_score_date'] = pd.to_datetime(credit_scores['credit_bureau_score_date'])
  credit_scores = credit_scores[credit_scores['credit_bureau_score_date'] >= '2016-09-30']

  most_recent_score = pd.merge(pd.DataFrame(credit_scores.groupby('ucfID')['credit_bureau_score_date'].max()), 
                               credit_scores, on = ['ucfID', 'credit_bureau_score_date'])
  
  most_recent_score = pd.merge(np.round(credit_scores.groupby('ucfID').mean()), most_recent_score, on = 'ucfID')
  most_recent_score.rename(columns = {'credit_bureau_score_x': 'recent_cred_score',
                                      'credit_bureau_score_y': 'avg_score_5_yrs'},
                           inplace = True)
  most_recent_score.drop('credit_bureau_score_date', axis = 1, inplace = True)

  # Changing all dates to number of days, beginning from start of competition
  ################################################################################
  pscu = pscu.drop_duplicates()

  dates = ['card_open_date', 'credit_line_last_change_date', 'card_expiration_date',
            'credit_bureau_score_date', 'last_purchase_date', 'last_cash_advance_date',
            'last_payment_date', 'last_transaction_date', 'card_account_transfer_upgrade_date']

  pscu[dates] = pscu[dates].apply(pd.to_datetime)

  current_time = datetime.strptime('2021/09/15 00:00:00', '%Y/%m/%d %H:%M:%S')

  for i in dates:
    pscu[i] = (current_time - pscu[i]).dt.days

  for i in dates:
    pscu.loc[pscu[i] == 18884, i] = 0

  pscu = pscu.groupby('ucfID').max()

  # PSCU table and joining the tables altogether
  # Take max value of each column to combat repeated ucfID
  print("The shape of the df before joining with the unsecured table:")
  print(pscu.shape)
  general = pd.merge(pscu, unsecured, on = 'ucfID', how = 'inner')
  print("\nThe shape of the df is now:")
  print(general.shape)
  print("\nNow joining with credit scores table:")
  general = pd.merge(general, most_recent_score, on = 'ucfID', how = 'inner')
  print("\nThe shape of the df is now:")
  print(general.shape)
  print("\nNow joining with rates table:")
  general = pd.merge(general, rates, on = 'ucfID', how = 'inner')
  print("\nThe shape of the df after joining all tables is:")
  print(general.shape)

  return general

def preprocess_df(combined_df):
  """ Will preprocess our DataFrame and Produce Transformed Variables

  :combined_df: all tables combined
  
  DATA PREPROCESSING STEPS:
  1. TRANSFORM: re-summing lifetime_delinquincy, turn negative balance to 0
  ['lifetime_delinquent_cycle_count', 'year_to_date_average_daily_balance_amount'] 

  2. REMOVE: columns with 100% null values and 100% 0 values
  ['predictive_growth_score', 'predictive_hardship_score', 
   'predictive_hardship_score_date', 'predictive_attrition_score', 
   'predictive_attrition_score_date', 'primary_cardholder_debt_to_income_ratio', 
   'predictive_growth_score_date', 'unpaid_dispute_amount']

  3.REMOVE: Columns that had errors in start of competition
  ['credit_bureau_score', 'primary_cardholder_personal_monthly_income', 
   'primary_cardholder_disposable_monthly_income', 'credit_bureau_score_date']

  4. CREATE: Binarizing Overlimits 1 = Yes Overlimit, 0 = No Overlimit
  ['overlimit_binary']

  5. REMOVE: Credit Line Amounts above $30,0000 (based off Addition giving max $30k)
  ['credit_line_amount']

  6. CREATE: Binarizing of Cash Advance in 3 years
  ['cash_advance_3_yrs']

  7. CREATE: Binarizing of Cash Advance in 7 years
  ['cash_advance_7_yrs']

  8. CREATE: Average annual historical disputes, n_disputes/card_open_date
  ['avg_annual_historical_dispute']

  9. CREATE: Average annual 1 and 2 cycle delinquincies, n_delinquincies/card_open 
  ['avg_annual_delinquent_1_cycle', 'avg_annual_delinquent_2_cycle']

  10. REMOVE: Abnormal values in 2nd cycle disputes
  ['avg_annual_delinquent_2_cycle']

  11. REMOVE: Abnormally high credit card lengths of credit > 48 years
  ['card_open_date']

  12. TRANSFORM: Cards 15 days expired, will just change to 0
  ['card_expiration_date']

  13. TRANSFORM: Binarizing returned check for year, 1 = Returned, 0 = No Returns
  ['year_to_date_return_check_count']
 
  14. CREATE: Percentage credit used (cred_line_amt - avail_cred)/(cred_line_amt)
  ['perc_credit_used']

  15. CREATE: Percentage Change of most recent cred_score/avg_5_yrs_cred_score
  ['perc_change_credit_score']

  16. TRANSFORM: Binarizing user's last purchase date in past 365 day 
  ['last_purchase_date']

  17. REMOVE: More columns that may complicate analysis
  ['year_to_date_overlimit_months_count', 'lifetime_delinquent_cycle_count',
   'year_to_date_high_balance_amount', 'previous_year_high_balance_amount', 
   'lifetime_high_balance_amount', 'card_account_transfer_upgrade_date', 
   'available_credit_amount', 'current_balance_amount', 'card_expiration_date',
   'last_payment_date']

  """

  # 1
  combined_df = combined_df.drop_duplicates()
  combined_df['lifetime_delinquent_cycle_count'] = combined_df['lifetime_delinquent_1_cycle_count'] + combined_df['lifetime_delinquent_2_cycles_count']
  combined_df.loc[combined_df['year_to_date_average_daily_balance_amount'] < 0, 'year_to_date_average_daily_balance_amount'] = 0
  
  # 2
  null_columns = ['predictive_growth_score', 'predictive_hardship_score', 'predictive_hardship_score_date', 
                  'predictive_attrition_score', 'predictive_attrition_score_date',
                  'primary_cardholder_debt_to_income_ratio', 'predictive_growth_score_date', 
                  'unpaid_dispute_amount']
  combined_df = combined_df.drop(null_columns, axis = 1)
  combined_df = combined_df.fillna(0)
  
  # 3
  errors = ['credit_bureau_score', 'primary_cardholder_personal_monthly_income', 
            'primary_cardholder_disposable_monthly_income', 'credit_bureau_score_date']
  combined_df = combined_df.drop(errors, axis = 1)

  # 4
  combined_df['overlimit_binary'] = combined_df['year_to_date_overlimit_months_count']
  combined_df.loc[combined_df['overlimit_binary'] > 0, 'overlimit_binary'] = 1

  # 5
  combined_df = combined_df[combined_df['credit_line_amount'] < 30000]

  # 6
  combined_df['cash_advance_3_yrs'] = combined_df['last_cash_advance_date']
  combined_df.loc[combined_df['last_cash_advance_date'] <= 365*3, 'cash_advance_3_yrs'] = 1
  combined_df.loc[combined_df['last_cash_advance_date'] > 365*3, 'cash_advance_3_yrs'] = 0

  # 7
  combined_df['cash_advance_7_yrs'] = combined_df['last_cash_advance_date']
  combined_df.loc[combined_df['last_cash_advance_date'] <= 365*7, 'cash_advance_7_yrs'] = 1
  combined_df.loc[combined_df['last_cash_advance_date'] > 365*7, 'cash_advance_7_yrs'] = 0

  # 8
  combined_df['avg_annual_historical_dispute'] = combined_df['historical_dispute_count']/combined_df['card_open_date']*365
  
  # 9
  combined_df['avg_annual_delinquent_1_cycle'] = (combined_df['lifetime_delinquent_1_cycle_count'])/combined_df['card_open_date']*365
  combined_df['avg_annual_delinquent_2_cycle'] = (combined_df['lifetime_delinquent_2_cycles_count'])/combined_df['card_open_date']*365
  
  # 10
  combined_df = combined_df[combined_df['avg_annual_delinquent_2_cycle'] < combined_df['avg_annual_delinquent_2_cycle'].quantile(.99)]
  
  # 11
  combined_df = combined_df[combined_df['card_open_date'] < 17500]
  
  # 12
  combined_df.loc[combined_df['card_expiration_date'] > 0, 'card_expiration_date'] = 0
  
  # 13 
  combined_df.loc[combined_df['year_to_date_return_check_count'] > 0, 'year_to_date_return_check_count'] = 1
  
  # 14
  combined_df['avg_perc_credit_used'] = combined_df['year_to_date_average_daily_balance_amount']/combined_df['credit_line_amount']
  
  # 15
  combined_df['perc_change_credit_score'] = (combined_df['recent_cred_score'] - combined_df['avg_score_5_yrs']) / combined_df['avg_score_5_yrs'] * 100

  # 16 
  combined_df.loc[combined_df['last_purchase_date'] < 365, 'last_purchase_date'] = 1

  # 17 
  more_removals = ['lifetime_delinquent_cycle_count', 'year_to_date_high_balance_amount', 
                   'previous_year_high_balance_amount', 'lifetime_high_balance_amount', 
                   'card_account_transfer_upgrade_date', 'year_to_date_overlimit_months_count', 
                   'available_credit_amount', 'current_balance_amount', 'card_expiration_date',
                   'last_payment_date']
  combined_df = combined_df.drop(more_removals, axis = 1)

  return combined_df.fillna(0)

def IncreaseConditions(table):
  """ No Increase Conditions

  Applying multiple conditioning on a dataframe to find users that fall into the 
  'No Credit Increase' category.

  If they have ...
  - credit_bureau_score               <   660
  - avg_credit_score_5_yrs            <   660
  - card_open_date                    <   365/2
  - avg_perc_credit_used              >   .40
  - avg_annual_historical_dispute     >   1/2
  - avg_annual_delinquent_1_cycle     >   1/2
  - avg_annual_delinquent_2_cycle     >   1/5
  - overlimit_binary                  =   1   
  - credit_line_amount                >=  30000
  - unpaid_billed_interest            >   avg_daily_balance * credit_line_Rate
  - credit_line_last_change_date      <   365/2
  - year_to_date_purchase_net_count   =   0
  - last_transaction_date             >   365/2

  """

  table.index = table['ucfID']

  a = (table['recent_cred_score'] < 660)
  b = (table['avg_score_5_yrs'] < 660)
  c = (table['card_open_date'] < 365/2)
  d = (table['avg_perc_credit_used'] > .40)
  e = (table['avg_annual_historical_dispute'] > 1)
  f = (table['avg_annual_delinquent_1_cycle'] > 1/2)
  g = (table['avg_annual_delinquent_2_cycle'] > 1/5)
  h = (table['overlimit_binary'] == 1)
  i = (table['credit_line_amount'] >= 30000)
  j = (table['year_to_date_unpaid_billed_interest_amount']/(table['avg_perc_credit_used']*table['credit_line_amount']) > table['credit_line_Rate']/100)
  k = (table['credit_line_last_change_date'] < 365/2)
  l = (table['year_to_date_purchase_net_count'] == 0)
  m = (table['last_transaction_date'] > 365/2)

  no_increase = table.loc[a | b | c | d | e | f | g | h | i | j | k | l | m, :]

  yes_increase = np.setdiff1d(table['ucfID'], no_increase.index)
  users_pass = table.loc[table.index.isin(yes_increase), :]
  users_pass.drop(['ucfID'], axis = 1, inplace = True)

  return users_pass

def testingIncrease(table):
  """ Applying a baseline condition of Users that get a credit line increase
  Mainly looking at credit score
  
  Without conditioning on other variables, our baseline for users that fall into
  'Yes Credit Increase' category.

  If they have ...
  - credit_bureau_score               >=   675
  - card_open_date                    <   365/2
  - credit_line_amount                <   30000
  - year_to_date_purchase_net_amount  >   0
  - credit_line_last_change_date      >=  365/2
  - year_to_date_purchase_net_count   >   0
  - last_transaction_date             <=  365/2

  """

  a = (table['recent_cred_score'] >= 675)
  b = (table['card_open_date'] >= 365/2)
  c = (table['credit_line_amount'] < 30000)
  d = (table['credit_line_last_change_date'] >= 365/2)
  e = (table['year_to_date_purchase_net_count'] > 0)
  f = (table['last_transaction_date'] <= 365/2)
 
  return table.loc[a, :]

def rounder(value):
  """ Custom credit limit rounding

  Helper function to later on round credit line increases

  """

  if (value < 300):
    return round(value/50.0)*50.0
  
  elif (value < 2000):
    return round(value/100.0)*100.0
  
  else:
    return round(value/250.0)*250.0

def credit_line_model(user_df, scoring_df):
  """ Full Credit Line Model

  Split each column into quantiles and group each user in respective quantile.
  Creating a scoring matrix filled with values between 0.1's and 1's 

  Input
  ---------------------------------
  :user_df: User's who we want to predict an increase
  :scoring_df: Original Customer Dataframe with historical data

  Output
  ---------------------------------
  :scoring_matrix: The scores of columns based off of historical customer data
  :bins: The respective bins that group 

  """

  # "Positive" columns 
  modeling_cols_pos = ['avg_score_5_yrs', 'card_open_date', 'recent_cred_score'] 

  # "Negative" columns
  modeling_cols_neg = ['avg_annual_delinquent_1_cycle', 'avg_annual_delinquent_2_cycle', 'avg_perc_credit_used', 
                       'year_to_date_unpaid_billed_interest_amount', 'previous_year_unpaid_billed_interest_amount', 
                       'avg_annual_historical_dispute', 'year_to_date_cash_advance_count']

  # Column names for scoring matrix
  matrix = ['avg_score_5_yrs_score',               #35 ch 
            'card_open_date_score',                #35 ch 
            'recent_cred_score_score',             #35 ch 
            'avg_annual_historical_dispute_score', #35 ch 
            'avg_annual_delinquent_1_cycle_score', #35 ph 
            'avg_annual_delinquent_2_cycle_score', #35 ph 
            'avg_perc_credit_used_score',          #35 ph
            'year_to_date_unpaid_billed_interest_amount_score', #30 ao 
            'previous_year_unpaid_billed_interest_amount_score', #30 ao 
            'year_to_date_cash_advance_count_score'] #30 ao 

  # Initializing bins and creating a temporary list
  bins = {}
  temp = [1.0]

  # Creating scores for positive columns
  for i in modeling_cols_pos:
    
    # Splits data into 10 quantiles and calculating bins
    x, y = pd.qcut(scoring_df.loc[:, i], q = 10, labels = np.arange(0.1, 1.1, 0.1), retbins=True)

    bins[i] = y

    # Grouping users into previous found quantile bin to find score
    user_df[f'{i}_score'] = pd.cut(user_df[i], bins = y, labels = np.arange(0.1, 1.1, 0.1), include_lowest=True)
  
  # Creating scores for negative columns
  for i in modeling_cols_neg:

    # Splits data into quantiles (not always a split of 10)
    x, y = pd.qcut(scoring_df.loc[:, i], q = 10, retbins=True, duplicates = 'drop')

    # If there are 10 quantiles, then use similar method to above, but negative scoring system
    if len(y) == 11:

      bins[i] = y

      user_df[f'{i}_score'] = pd.cut(user_df[i], bins = y, labels = np.arange(1.0, 0, -0.1), include_lowest=True)

    # Need to calculate a different negative scoring system for sparse columns
    else:

      bins[i] = y
      length_labels = len(y) - 2

      for j in range(length_labels):

        temp.append((length_labels -j)/10)
      # Scoring system for negative sparse column
      user_df[f'{i}_score'] = pd.cut(user_df[i], bins = y, labels = temp, include_lowest=True)
      
      temp = [1.0]

  return user_df[matrix].astype(np.float64), bins

def calculating_rates(scoring_matrix):
  """ Calculating Rates based off of linear combination with weighted columns

  This is a very customizeable weighing system we based on similar FICO scoring metrics.

  Could be a more data-driven way to pick weights, but this is what we felt was
  accurate for the time given.

  The max perfect_score% is chosen based off of average credit limit increases
  are generally between a 10%-25% increase. Since we are automatically increasing
  limits, then 20% seems to be a generous max amount.

  :scoring_matrix: Will be the output from credit_line_model() function

  """
  # Equates to the max x% credit line increase for "Ideal" users
  perfect_score = 20

  # Flexible weighing system
  weights = {'avg_score_5_yrs_score': 12.5,
          'card_open_date_score': 7.5,
          'recent_cred_score_score': 12.5,
          'avg_annual_historical_dispute_score': 2.5,
          'avg_annual_delinquent_1_cycle_score': 10,
          'avg_annual_delinquent_2_cycle_score': 15,
          'avg_perc_credit_used_score': 10,
          'year_to_date_unpaid_billed_interest_amount_score': 12.5,
          'previous_year_unpaid_billed_interest_amount_score': 12.5,
          'year_to_date_cash_advance_count_score': 5}

  # Creating a dataframe for ease of use
  rates = pd.DataFrame()

  # Array multiplication
  for i in list(weights.keys()):
    rates[f'{i}'] = weights[i] * scoring_matrix[i]

  return pd.DataFrame((rates.sum(axis = 1)/100)*perfect_score/100, columns=['Rates'])

def credit_line_increase(credit_rates_df, original_df):
  """ Calculate User's respective new credit limit

  Input
  ---------------------------------
  :credit_rates_df: Will be output from calculating_rates() function
  :original_df: First table where credit_line_amount is a present column

  Output
  ---------------------------------
  :credit_increase_df: Predicted Cred Increase, Current Cred Lim, New Cred Lim

  """

  # Calculates the credit line increase amount
  increases = pd.DataFrame(credit_rates_df['Rates'].values * 
                           original_df.loc[original_df['ucfID'].isin(credit_rates_df.index), :]['credit_line_amount'].values,
                           index = credit_rates_df.index, columns=['increase_amt'])
  
  # Inner join with original df to finalize credit line summation
  credit_increase_df = increases.merge(original_df, left_on = increases.index, 
                                       right_on = original_df['ucfID'], how = 'inner')[['ucfID', 'increase_amt', 'credit_line_amount']]
  # Custom rounding function
  credit_increase_df['increase_amt'] = credit_increase_df['increase_amt'].apply(rounder)

  # row summation to calculate all credit lines
  credit_increase_df['new_credit_line'] = credit_increase_df.sum(axis = 1).astype(np.float64)
  # Changing all credit lines above 30,000 to max at 30,000
  credit_increase_df.loc[credit_increase_df['new_credit_line'] >= 30000, 'new_credit_line'] = 30000

  return credit_increase_df


############################################################################################

# DATA PREPROCESSING
############################################################################################

general_df = pd.read_csv('ucf_PSCU_UCF_2021_Data.csv', index_col=0)
rates_df = pd.read_csv('20211104_APPEND_ucf_rates_history.csv')
scores_df = pd.read_csv('20211104_APPEND_ucf_bureau_score_history.csv')
unsecured_df = pd.read_csv('ucf_unsecured_debt.csv')

combined_df = combine_tables(unsecured_df, scores_df, general_df, rates_df)

full_df = preprocess_df(combined_df)

full_df.to_csv('full_df.csv')

############################################################################################

# DATA MODELING
############################################################################################

# Reading in the processed table
df = pd.read_csv('full_df.csv', index_col = 0)
df.head()

# Feature Selection
############################################################################################
x = df.drop(['ucfID', 'overlimit_binary', 'avg_perc_credit_used', 
             'year_to_date_average_daily_balance_amount', 'credit_line_Rate'], axis = 1)
y1 = df['overlimit_binary']

x_train, x_test, y_train, y_test = train_test_split(x, y1, test_size=0.2, random_state=0)
# fit model no training data
model = XGBClassifier()
model.fit(x_train, y_train)
# make predictions for test data

feature_important = model.get_booster().get_score(importance_type='weight')
keys = list(feature_important.keys())
values = list(feature_important.values())

data_overlimit = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)

# dispute and any columns made to calculate historical dispute
x = df.drop(['ucfID', 'avg_annual_historical_dispute', 'historical_dispute_count', 
             'card_open_date', 'credit_line_Rate'], axis = 1)
y1 = df['avg_annual_historical_dispute']

x_train, x_test, y_train, y_test = train_test_split(x, y1, test_size=0.2, random_state=0)
# fit model no training data
model = XGBRegressor()
model.fit(x_train, y_train)
# make predictions for test data

feature_important = model.get_booster().get_score(importance_type='weight')
keys = list(feature_important.keys())
values = list(feature_important.values())

data_dispute = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)

# CREDIT SCORE take out any relating credit score columns
x = df.drop(['ucfID', 'recent_cred_score', 'avg_score_5_yrs', 'perc_change_credit_score',
             'credit_line_Rate'], axis = 1)
y1 = df['recent_cred_score']

x_train, x_test, y_train, y_test = train_test_split(x, y1, test_size=0.2, random_state=0)
# fit model no training data
model = XGBRegressor()
model.fit(x_train, y_train)
# make predictions for test data

feature_important = model.get_booster().get_score(importance_type='weight')
keys = list(feature_important.keys())
values = list(feature_important.values())

data_delinquint2 = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)

# CREDIT SCORE take out any relating credit score columns
x = df.drop(['ucfID', 'lifetime_delinquent_1_cycle_count', 'avg_annual_delinquent_1_cycle',
             'credit_line_Rate'], axis = 1)
y1 = df['lifetime_delinquent_1_cycle_count']

x_train, x_test, y_train, y_test = train_test_split(x, y1, test_size=0.2, random_state=0)
# fit model no training data
model = XGBRegressor()
model.fit(x_train, y_train)
# make predictions for test data

feature_important = model.get_booster().get_score(importance_type='weight')
keys = list(feature_important.keys())
values = list(feature_important.values())

data_delinquint1 = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)

# Take out life 1 cycle and 2 cycle 
x = df.drop(['ucfID','avg_annual_delinquent_1_cycle', 'avg_annual_delinquent_2_cycle', 
             'lifetime_delinquent_1_cycle_count', 'lifetime_delinquent_2_cycles_count', 
             'credit_line_Rate'], axis = 1)
y1 = df['avg_annual_delinquent_2_cycle']

x_train, x_test, y_train, y_test = train_test_split(x, y1, test_size=0.2, random_state=0)
# fit model no training data
model = XGBRegressor()
model.fit(x_train, y_train)
# make predictions for test data

feature_important = model.get_booster().get_score(importance_type='weight')
keys = list(feature_important.keys())
values = list(feature_important.values())

data_delinquint2 = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)

# CREDIT SCORE take out any relating credit score columns
x = df.drop(['ucfID', 'cash_advance_3_yrs', 'last_cash_advance_date', 'credit_line_Rate'], axis = 1)
y1 = df['cash_advance_3_yrs']

x_train, x_test, y_train, y_test = train_test_split(x, y1, test_size=0.2, random_state=0)
# fit model no training data
model = XGBClassifier()
model.fit(x_train, y_train)
# make predictions for test data

feature_important = model.get_booster().get_score(importance_type='weight')
keys = list(feature_important.keys())
values = list(feature_important.values())

data_advance = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)

# These will be columns we will definitely need to include in our model
definitive_cols = ['ucfID', 'credit_line_amount', 'recent_cred_score', 'lifetime_delinquent_1_cycle_count', 
                   'avg_annual_delinquent_2_cycle', 'avg_annual_historical_dispute', 'cash_advance_3_yrs',
                   'overlimit_binary', 'credit_line_Rate', 'avg_perc_credit_used', 
                   'year_to_date_average_daily_balance_amount', 'historical_dispute_count', 
                   'card_open_date', 'avg_score_5_yrs', 'perc_change_credit_score', 
                   'lifetime_delinquent_2_cycles_count']

# The top 10 predictors for each XGB model
signficant_cols = np.concatenate((data_overlimit.index[:10].values, 
                                  data_delinquint2.index[:10].values,
                                  data_dispute.index[:10].values,
                                  data_advance.index[:10].values,
                                  definitive_cols))

# Putting the significant features into a df
significant_df = pd.DataFrame({'important_feats': signficant_cols})
# Only taking unique values so no repeats show up
columns = np.unique(significant_df['important_feats'].values)

increase_customers = IncreaseConditions(df)

# 1st parameter is a test dataset we want to use the model on
# 2nd parameter is our df with historical customer data
# Calculates the scoring matrix
full_df, full_bins = credit_line_model(increase_customers, increase_customers)

# Calculates the credit line increase %
full_rates = calculating_rates(full_df)

# Calculates actual credit line increase amount
full_table = credit_line_increase(full_rates, df)

# Save file to csv
full_table.to_csv('predictions.csv')

print (f'The script took {time.time() - startTime} second !')