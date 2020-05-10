import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime

plt.style.use('fivethirtyeight')

confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
recoveries_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
latest_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/05-07-2020.csv')
us_medical_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports_us/05-07-2020.csv')

cols = confirmed_df.keys()

#Confirmed Cases, Deaths And Recoveries DataFrame & Here World Variable means SAARC (South Asian Association for Regional Cooperation)

confirmed = confirmed_df.loc[:, cols[4]:cols[-1]]
deaths = deaths_df.loc[:, cols[4]:cols[-1]]
recoveries = recoveries_df.loc[:, cols[4]:cols[-1]]
dates = confirmed.keys()
world_cases = []
total_deaths = []
mortality_rate = []
recovery_rate = []
total_recovered = []
total_active = []

#Country wise Confirmed Cases List:

bangladesh_cases = []
india_cases = []
bhutan_cases = []
maldives_cases = []
nepal_cases = []
pakistan_cases = []
srilanka_cases = []
afghanistan_cases = []

#Country wise Deaths List:

bangladesh_deaths = []
india_deaths = []
bhutan_deaths = []
maldives_deaths = []
nepal_deaths = []
pakistan_deaths = []
srilanka_deaths = []
afghanistan_deaths = []

#Country wise Recoveries List:

bangladesh_recoveries = []
india_recoveries = []
bhutan_recoveries = []
maldives_recoveries = []
nepal_recoveries = []
pakistan_recoveries = []
srilanka_recoveries = []
afghanistan_recoveries = []

#adding datas in our list from the dataset according to dates

for i in dates:
    # case studies
    bangladesh_cases.append(confirmed_df[confirmed_df['Country/Region'] == 'Bangladesh'][i].sum())
    india_cases.append(confirmed_df[confirmed_df['Country/Region'] == 'India'][i].sum())
    bhutan_cases.append(confirmed_df[confirmed_df['Country/Region'] == 'Bhutan'][i].sum())
    maldives_cases.append(confirmed_df[confirmed_df['Country/Region'] == 'Maldives'][i].sum())
    nepal_cases.append(confirmed_df[confirmed_df['Country/Region'] == 'Nepal'][i].sum())
    pakistan_cases.append(confirmed_df[confirmed_df['Country/Region'] == 'Pakistan'][i].sum())
    srilanka_cases.append(confirmed_df[confirmed_df['Country/Region'] == 'Sri Lanka'][i].sum())
    afghanistan_cases.append(confirmed_df[confirmed_df['Country/Region'] == 'Afghanistan'][i].sum())

    bangladesh_deaths.append(deaths_df[deaths_df['Country/Region'] == 'Bangladesh'][i].sum())
    india_deaths.append(deaths_df[deaths_df['Country/Region'] == 'India'][i].sum())
    bhutan_deaths.append(deaths_df[deaths_df['Country/Region'] == 'Bhutan'][i].sum())
    maldives_deaths.append(deaths_df[deaths_df['Country/Region'] == 'Maldives'][i].sum())
    nepal_deaths.append(deaths_df[deaths_df['Country/Region'] == 'Nepal'][i].sum())
    pakistan_deaths.append(deaths_df[deaths_df['Country/Region'] == 'Pakistan'][i].sum())
    srilanka_deaths.append(deaths_df[deaths_df['Country/Region'] == 'Sri Lanka'][i].sum())
    afghanistan_deaths.append(deaths_df[deaths_df['Country/Region'] == 'Russia'][i].sum())

    bangladesh_recoveries.append(recoveries_df[recoveries_df['Country/Region'] == 'Bangladesh'][i].sum())
    india_recoveries.append(recoveries_df[recoveries_df['Country/Region'] == 'India'][i].sum())
    bhutan_recoveries.append(recoveries_df[recoveries_df['Country/Region'] == 'Bhutan'][i].sum())
    maldives_recoveries.append(recoveries_df[recoveries_df['Country/Region'] == 'Maldives'][i].sum())
    nepal_recoveries.append(recoveries_df[recoveries_df['Country/Region'] == 'Nepal'][i].sum())
    pakistan_recoveries.append(recoveries_df[recoveries_df['Country/Region'] == 'Pakistan'][i].sum())
    srilanka_recoveries.append(recoveries_df[recoveries_df['Country/Region'] == 'Sri Lanka'][i].sum())
    afghanistan_recoveries.append(recoveries_df[recoveries_df['Country/Region'] == 'Afghanistan'][i].sum())

    #Summation of the SAARC CONFIRMED CASES are taken into the variable confirmed_sum

    confirmed_sum = confirmed_df[confirmed_df['Country/Region'] == 'Bangladesh'][i].sum() + \
                    confirmed_df[confirmed_df['Country/Region'] == 'India'][i].sum() + \
                    confirmed_df[confirmed_df['Country/Region'] == 'Bhutan'][i].sum() + \
                    confirmed_df[confirmed_df['Country/Region'] == 'Maldives'][i].sum() + \
                    confirmed_df[confirmed_df['Country/Region'] == 'Nepal'][i].sum() + \
                    confirmed_df[confirmed_df['Country/Region'] == 'Pakistan'][i].sum() + \
                    confirmed_df[confirmed_df['Country/Region'] == 'Afghanistan'][i].sum() + \
                    confirmed_df[confirmed_df['Country/Region'] == 'Sri Lanka'][i].sum()

    #Summation of the SAARC DEATH CASES are taken into the variable death_sum

    death_sum = deaths_df[deaths_df['Country/Region'] == 'Bangladesh'][i].sum() + \
                deaths_df[deaths_df['Country/Region'] == 'India'][i].sum() + \
                deaths_df[deaths_df['Country/Region'] == 'Bhutan'][i].sum() + \
                deaths_df[deaths_df['Country/Region'] == 'Maldives'][i].sum() + \
                deaths_df[deaths_df['Country/Region'] == 'Nepal'][i].sum() + \
                deaths_df[deaths_df['Country/Region'] == 'Pakistan'][i].sum() + \
                deaths_df[deaths_df['Country/Region'] == 'Afghanistan'][i].sum() + \
                deaths_df[deaths_df['Country/Region'] == 'Sri Lanka'][i].sum()

    #Summation of the SAARC RECOVERIES are taken into the variable recovered_sum

    recovered_sum = recoveries_df[recoveries_df['Country/Region'] == 'Bangladesh'][i].sum() + \
                    recoveries_df[recoveries_df['Country/Region'] == 'India'][i].sum() + \
                    recoveries_df[recoveries_df['Country/Region'] == 'Bhutan'][i].sum() + \
                    recoveries_df[recoveries_df['Country/Region'] == 'Maldives'][i].sum() + \
                    recoveries_df[recoveries_df['Country/Region'] == 'Nepal'][i].sum() + \
                    recoveries_df[recoveries_df['Country/Region'] == 'Pakistan'][i].sum() + \
                    recoveries_df[recoveries_df['Country/Region'] == 'Afghanistan'][i].sum() + \
                    recoveries_df[recoveries_df['Country/Region'] == 'Sri Lanka'][i].sum()

    # SAARC confirmed, deaths, recovered, and active

    world_cases.append(confirmed_sum)
    total_deaths.append(death_sum)
    total_recovered.append(recovered_sum)
    total_active.append(confirmed_sum - death_sum - recovered_sum)

    # calculate mortality and recovery rates and ignoring the rates when confirmed cases =0

    if (confirmed_sum != 0):
        mortality_rate.append(death_sum / confirmed_sum)
        recovery_rate.append(recovered_sum / confirmed_sum)
    else:
        mortality_rate.append(0)
        recovery_rate.append(0)

#Function of daily increasing(Confirmed Cases, Deaths) list of all SAARC COUNTRIES:

def daily_increase(data):
    d = []
    for i in range(len(data)):
        if i == 0:
            d.append(data[0])
        else:
            d.append(data[i] - data[i - 1])
    return d


#daily confirmed cases both in terms of SAARC as a whole and as well as the individual countries of SAARC

world_daily_increase = daily_increase(world_cases)
bangladesh_daily_increase = daily_increase(bangladesh_cases)
india_daily_increase = daily_increase(india_cases)
bhutan_daily_increase = daily_increase(bhutan_cases)
nepal_daily_increase = daily_increase(nepal_cases)
pakistan_daily_increase = daily_increase(pakistan_cases)
srilanka_daily_increase = daily_increase(srilanka_cases)
maldives_daily_increase = daily_increase(maldives_cases)
afghanistan_daily_increase = daily_increase(afghanistan_cases)

#daily deaths  cases both in terms of SAARC as a whole and as well as the individual countries of SAARC

world_daily_death = daily_increase(total_deaths)
bangladesh_daily_death = daily_increase(bangladesh_deaths)
india_daily_death = daily_increase(india_deaths)
bhutan_daily_death = daily_increase(bhutan_deaths)
nepal_daily_death = daily_increase(nepal_deaths)
pakistan_daily_death = daily_increase(pakistan_deaths)
srilanka_daily_death = daily_increase(srilanka_deaths)
maldives_daily_death = daily_increase(maldives_deaths)
afghanistan_daily_death = daily_increase(afghanistan_deaths)

#daily recoveries  cases both in terms of SAARC as a whole and as well as the individual countries of SAARC

world_daily_recovery = daily_increase(total_recovered)
bangladesh_daily_recovery = daily_increase(bangladesh_recoveries)
india_daily_recovery = daily_increase(india_cases)
bhutan_daily_recovery = daily_increase(bhutan_cases)
nepal_daily_recovery = daily_increase(nepal_recoveries)
pakistan_daily_recovery = daily_increase(pakistan_recoveries)
srilanka_daily_recovery = daily_increase(srilanka_recoveries)
maldives_daily_recovery = daily_increase(maldives_recoveries)
afghanistan_daily_recovery = daily_increase(afghanistan_recoveries)

#reshaping the data

days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)
world_cases = np.array(world_cases).reshape(-1, 1)
total_deaths = np.array(total_deaths).reshape(-1, 1)
total_recovered = np.array(total_recovered).reshape(-1, 1)


# setting the prediction parameter of days = 10 days in future

days_in_future = 10
future_forcast = np.array([i for i in range(len(dates) + days_in_future)]).reshape(-1, 1)
adjusted_dates = future_forcast[:-10]

# future forecasting from the start date

start = '1/22/2020'
start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
future_forcast_dates = []
for i in range(len(future_forcast)):
    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))

#Creating a split of the train and  test data using the train_Test_split

X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22, world_cases, test_size=0.36,shuffle=False)

#setting up the SVM model

svm_confirmed = SVR(shrinking=True, kernel='poly', gamma=0.01, epsilon=1, degree=6, C=0.1)
svm_confirmed.fit(X_train_confirmed, y_train_confirmed)
svm_pred = svm_confirmed.predict(future_forcast)

# check against testing data using a plot and printing out the mean_absolute_error as well as mean_squared_error

svm_test_pred = svm_confirmed.predict(X_test_confirmed)
plt.plot(y_test_confirmed)
plt.plot(svm_test_pred)
plt.legend(['Test Data', 'SVM Predictions'])
plt.show()
print('MAE:', mean_absolute_error(svm_test_pred, y_test_confirmed))
print('MSE:', mean_squared_error(svm_test_pred, y_test_confirmed))

#setting up the Polynomial Regression

poly = PolynomialFeatures(degree=5)
poly_X_train_confirmed = poly.fit_transform(X_train_confirmed)
poly_X_test_confirmed = poly.fit_transform(X_test_confirmed)
poly_future_forcast = poly.fit_transform(future_forcast)

bayesian_poly = PolynomialFeatures(degree=4)
bayesian_poly_X_train_confirmed = bayesian_poly.fit_transform(X_train_confirmed)
bayesian_poly_X_test_confirmed = bayesian_poly.fit_transform(X_test_confirmed)
bayesian_poly_future_forcast = bayesian_poly.fit_transform(future_forcast)


# bayesian ridge polynomial regression
tol = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
normalize = [True, False]

bayesian_grid = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2' : alpha_2, 'lambda_1': lambda_1, 'lambda_2' : lambda_2,
                 'normalize' : normalize}

bayesian = BayesianRidge(fit_intercept=False)
bayesian_search = RandomizedSearchCV(bayesian, bayesian_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)
bayesian_search.fit(bayesian_poly_X_train_confirmed, y_train_confirmed)


bayesian_confirmed = bayesian_search.best_estimator_
test_bayesian_pred = bayesian_confirmed.predict(bayesian_poly_X_test_confirmed)
bayesian_pred = bayesian_confirmed.predict(bayesian_poly_future_forcast)
print('MAE:', mean_absolute_error(test_bayesian_pred, y_test_confirmed))
print('MSE:',mean_squared_error(test_bayesian_pred, y_test_confirmed))

plt.plot(y_test_confirmed)
plt.plot(test_bayesian_pred)
plt.legend(['Test Data', 'Bayesian Ridge Polynomial Predictions'])



#bayesian_poly = BayesianRidge(tol=1e-6,alpha_init=1 ,lambda_init=0.001, fit_intercept=False, compute_score=True)
#n_order = 3
#world_cases = np.array(world_cases).reshape(-1, 1)
#reshaped_X_train = np.array(X_train_confirmed).reshape(-1, 1)
#reshaped_X_test = np.array(X_test_confirmed).reshape(-1, 1)
#bayesian_X_train_confirmed = np.vander(reshaped_X_train, n_order + 1, increasing=True)
#bayesian_X_test_confirmed = np.vander(reshaped_X_train, n_order + 1, increasing=True)


#bayesian_poly_X_train_confirmed = bayesian_poly.fit(X_train_confirmed,y_train_confirmed)
#bayesian_poly_X_test_confirmed = bayesian_poly.fit(X_test_confirmed,y_test_confirmed)
#bayesian_poly_future_forcast = bayesian_poly.predict(future_forcast)


#plt.plot(y_test_confirmed)
#plt.plot(bayesian_poly_future_forcast)
#plt.legend(['Test Data', 'Bayesian Regression Predictions'])
#plt.show()
#print()

linear_model = LinearRegression(normalize=True, fit_intercept=False)
linear_model.fit(poly_X_train_confirmed, y_train_confirmed)
test_linear_pred = linear_model.predict(poly_X_test_confirmed)
linear_pred = linear_model.predict(poly_future_forcast)
print('MAE:', mean_absolute_error(test_linear_pred, y_test_confirmed))
print('MSE:', mean_squared_error(test_linear_pred, y_test_confirmed))

plt.plot(y_test_confirmed)
plt.plot(test_linear_pred)
plt.legend(['Test Data', 'Polynomial Regression Predictions'])
plt.show()
print()


#Plotting Number of Coronavirus Cases Over Time for SAARC

adjusted_dates = adjusted_dates.reshape(1, -1)[0]
# plt.figure(figsize=(8, 12))
plt.plot(adjusted_dates, world_cases)
plt.title('SAARC # of Coronavirus Cases Over Time')
plt.xlabel('Days Since 1/22/2020')
plt.ylabel('# of Cases')
# plt.xticks()
# plt.yticks()
plt.show()
print()

#Plotting Number of  Coronavirus DEATHS, RECOVERIES, ACTIVE CASES Over Time for SAARC

# plt.figure(figsize=(12, 6))
plt.plot(adjusted_dates, total_deaths)
plt.title('SAARC # of Coronavirus Deaths Over Time')
plt.xlabel('Days Since 1/22/2020')
plt.ylabel('# of Cases')
# plt.xticks(size=20)
# plt.yticks(size=20)
plt.show()
print()

# plt.figure(figsize=(12, 6))
plt.plot(adjusted_dates, total_recovered)
plt.title('SAARC # of Coronavirus Recoveries Over Time')
plt.xlabel('Days Since 1/22/2020')
plt.ylabel('# of Cases')
# plt.xticks(size=20)
# plt.yticks(size=20)
plt.show()
print()

# plt.figure(figsize=(12, 6))
plt.plot(adjusted_dates, total_active)
plt.title('SAARC # of Coronavirus Active Cases Over Time')
plt.xlabel('Days Since 1/22/2020')
plt.ylabel('# of Active Cases')
# plt.xticks(size=20)
# plt.yticks(size=20)
plt.show()
print()

# Plotting SAARC Daily Increases in Confirmed Cases

# plt.figure(figsize=(12, 6))
plt.bar(adjusted_dates, world_daily_increase)
plt.title('SAARC Daily Increases in Confirmed Cases')
plt.xlabel('Days Since 1/22/2020')
plt.ylabel('# of Cases')
# plt.xticks(size=20)
# plt.yticks(size=20)
plt.show()
print()

# Plotting SAARC Daily Increases in Confirmed Deaths

# plt.figure(figsize=(12, 6))
plt.bar(adjusted_dates, world_daily_death)
plt.title('SAARC Daily Increases in Confirmed Deaths')
plt.xlabel('Days Since 1/22/2020')
plt.ylabel('# of Cases')
# plt.xticks(size=20)
# plt.yticks(size=20)
plt.show()
print()

# Plotting SAARC Daily Increases in Confirmed Recoveries

# plt.figure(figsize=(12, 6))
plt.bar(adjusted_dates, world_daily_recovery)
plt.title('SAARC Daily Increases in Confirmed Recoveries')
plt.xlabel('Days Since 1/22/2020')
plt.ylabel('# of Cases')
# plt.xticks(size=20)
# plt.yticks(size=20)
plt.show()
print()

# Plotting Log of Number of Coronavirus Cases Over Time

# plt.figure(figsize=(12, 6))
plt.plot(adjusted_dates, np.log10(world_cases))
plt.title('SAARC Log of # of Coronavirus Cases Over Time')
plt.xlabel('Days Since 1/22/2020')
plt.ylabel('# of Cases')
# plt.xticks(size=20)
# plt.yticks(size=20)
plt.show()
print()

# Plotting Log of Number of Coronavirus Deaths Over Time

##plt.figure(figsize=(12, 6))
plt.plot(adjusted_dates, np.log10(total_deaths))
plt.title('SAARC Log of # of Coronavirus Deaths Over Time')
plt.xlabel('Days Since 1/22/2020')
plt.ylabel('# of Cases')
# plt.xticks(size=20)
# plt.yticks(size=20)
plt.show()
print()

# Plotting Log of Number of Coronavirus Recoveries Over Time

# plt.figure(figsize=(12, 6))
plt.plot(adjusted_dates, np.log10(total_recovered))
plt.title('SAARC Log of # of Coronavirus Recoveries Over Time')
plt.xlabel('Days Since 1/22/2020')
plt.ylabel('# of Cases')
# plt.xticks(size=20)
# plt.yticks(size=20)
plt.show()
print()

# Function of  plotting total confirmed cases, daily increasing (Confirmed cases, Death Cases, Recovery Cases)

def country_plot(x, y1, y2, y3, y4, country):
    # plt.figure(figsize=(12, 6))
    plt.plot(x, y1)
    plt.title('{} Confirmed Cases'.format(country))
    plt.xlabel('Days Since 1/22/2020')
    plt.ylabel('# of Cases')
    # plt.xticks(size=20)
    # plt.yticks(size=20)
    plt.show()
    print()

    # plt.figure(figsize=(12, 6))
    plt.bar(x, y2)
    plt.title('{} Daily Increases in Confirmed Cases'.format(country))
    plt.xlabel('Days Since 1/22/2020')
    plt.ylabel('# of Cases')
    # plt.xticks(size=20)
    # plt.yticks(size=20)
    plt.show()
    print()

    # plt.figure(figsize=(12, 6))
    plt.bar(x, y3)
    plt.title('{} Daily Increases in Deaths'.format(country))
    plt.xlabel('Days Since 1/22/2020')
    plt.ylabel('# of Cases')
    # plt.xticks(size=20)
    # plt.yticks(size=20)
    plt.show()
    print()

    # plt.figure(figsize=(12, 6))
    plt.bar(x, y4)
    plt.title('{} Daily Increases in Recoveries'.format(country))
    plt.xlabel('Days Since 1/22/2020')
    plt.ylabel('# of Cases')
    # plt.xticks(size=20)
    # plt.yticks(size=20)
    plt.show()
    print()


# Calling the function to plot the data of SAARC country

country_plot(adjusted_dates, bangladesh_cases, bangladesh_daily_increase, bangladesh_daily_death,bangladesh_daily_recovery, 'Bangladesh')
country_plot(adjusted_dates, india_cases, india_daily_increase, india_daily_death, india_daily_recovery, 'India')
country_plot(adjusted_dates, bhutan_cases, bhutan_daily_increase, bhutan_daily_death, bhutan_daily_recovery, 'Bhutan')
country_plot(adjusted_dates, nepal_cases, nepal_daily_increase, nepal_daily_death, nepal_daily_recovery, 'Nepal')
country_plot(adjusted_dates, srilanka_cases, srilanka_daily_increase, srilanka_daily_death, srilanka_daily_recovery,'Sri Lanka')
country_plot(adjusted_dates, maldives_cases, maldives_daily_increase, maldives_daily_death, maldives_daily_recovery,'Maldives')
country_plot(adjusted_dates, pakistan_cases, pakistan_daily_increase, pakistan_daily_death, pakistan_daily_recovery,'Pakistan')
country_plot(adjusted_dates, afghanistan_cases, afghanistan_daily_increase, afghanistan_daily_death,afghanistan_daily_recovery, 'Afghanistan')


# plotting confirmed cases of SAARC countires altogether

plt.plot(adjusted_dates, bangladesh_cases, 'g')
plt.plot(adjusted_dates, india_cases, 'b')
plt.plot(adjusted_dates, nepal_cases, 'r')
plt.plot(adjusted_dates, bhutan_cases, 'c')
plt.plot(adjusted_dates, maldives_cases, ':')
plt.plot(adjusted_dates, pakistan_cases, 'm')
plt.plot(adjusted_dates, afghanistan_cases, 'k')
plt.plot(adjusted_dates, srilanka_cases, color='orange')
plt.title('# of Coronavirus Cases')
plt.xlabel('Days Since 1/22/2020')
plt.ylabel('# of Cases')
plt.legend(['Bangladesh', 'India', 'Nepal', 'Bhutan', 'Maldives', 'Pakistan', 'Afghanistan', 'Sri Lanka'])
plt.show()
print()

# plotting death cases of SAARC countires altogether

plt.plot(adjusted_dates, bangladesh_deaths, 'g')
plt.plot(adjusted_dates, india_deaths, 'b')
plt.plot(adjusted_dates, nepal_deaths, 'r')
plt.plot(adjusted_dates, bhutan_deaths, 'c')
plt.plot(adjusted_dates, maldives_deaths, ':')
plt.plot(adjusted_dates, pakistan_deaths, 'm')
plt.plot(adjusted_dates, afghanistan_deaths, 'k')
plt.plot(adjusted_dates, srilanka_deaths, color='orange')

plt.title('# of Coronavirus Deaths')
plt.xlabel('Days Since 1/22/2020')
plt.ylabel('# of Cases')
plt.legend(['Bangladesh', 'India', 'Nepal', 'Bhutan', 'Maldives', 'Pakistan', 'Afghanistan', 'Sri Lanka'])
plt.show()
print()

# plotting recovery cases of SAARC countires altogether

plt.plot(adjusted_dates, bangladesh_recoveries, 'g')
plt.plot(adjusted_dates, india_recoveries, 'b')
plt.plot(adjusted_dates, nepal_recoveries, 'r')
plt.plot(adjusted_dates, bhutan_recoveries, 'c')
plt.plot(adjusted_dates, maldives_recoveries, ':')
plt.plot(adjusted_dates, pakistan_recoveries, 'm')
plt.plot(adjusted_dates, afghanistan_recoveries, 'k')
plt.plot(adjusted_dates, srilanka_recoveries, color='orange')

plt.title('# of Coronavirus Recoveries')
plt.xlabel('Days Since 1/22/2020')
plt.ylabel('# of Cases')
plt.legend(['Bangladesh', 'India', 'Nepal', 'Bhutan', 'Maldives', 'Pakistan', 'Afghanistan', 'Sri Lanka'])

plt.show()
print()


# plot predictions of SAARC via SVM, Polynomial Regression,

def plot_predictions(x, y, pred, algo_name, color):
    plt.plot(x, y)
    plt.plot(future_forcast, pred, linestyle='dashed', color=color)
    plt.title('# of Coronavirus Cases Over Time')
    plt.xlabel('Days Since 1/22/2020')
    plt.ylabel('# of Cases')
    plt.legend(['Confirmed Cases', algo_name])
    plt.show()


plot_predictions(adjusted_dates, world_cases, svm_pred, 'SVM Predictions', 'purple')
plot_predictions(adjusted_dates, world_cases, linear_pred, 'Polynomial Regression Predictions', 'orange')
plot_predictions(adjusted_dates, world_cases, bayesian_pred, 'Bayesian Ridge Regression Predictions','green')

# Future predictions using SVM
svm_df = pd.DataFrame({'Date': future_forcast_dates[-10:], 'SVM Predicted # of Confirmed Cases SAARC': np.round(svm_pred[-10:])})
print(svm_df)
# Future predictions using polynomial regression
linear_pred = linear_pred.reshape(1,-1)[0]
svm_df = pd.DataFrame({'Date': future_forcast_dates[-10:], 'Polynomial Predicted # of Confirmed Cases SAARC': np.round(linear_pred[-10:])})
print(svm_df)
# Future predictions using Bayesian Ridge
bayesian_poly_future_forcast=bayesian_poly_future_forcast.reshape(1,-1)[0]
svm_df = pd.DataFrame({'Date': future_forcast_dates[-10:], 'Bayesian Ridge Predicted # of Confirmed Cases SAARC': np.round(bayesian_pred[-10:])})
print(svm_df)


# Plotting Mortality Rate Of SAARC

mean_mortality_rate = np.mean(mortality_rate)
plt.plot(adjusted_dates, mortality_rate, color='orange')
plt.axhline(y=mean_mortality_rate, linestyle='--', color='black')
plt.title('Mortality Rate of Coronavirus Over Time')
plt.legend(['mortality rate', 'y=' + str(mean_mortality_rate)])
plt.xlabel('Days Since 1/22/2020')
plt.ylabel('Mortality Rate')
plt.show()
print()

#Plotting Recovery Rate of SAARC

mean_recovery_rate = np.mean(recovery_rate)
plt.plot(adjusted_dates, recovery_rate, color='blue')
plt.axhline(y=mean_recovery_rate, linestyle='--', color='black')
plt.title('Recovery Rate of Coronavirus Over Time')
plt.legend(['recovery rate', 'y=' + str(mean_recovery_rate)])
plt.xlabel('Days Since 1/22/2020')
plt.ylabel('Recovery Rate')
plt.show()
print()

#plotting total death and total recovere in one graph

plt.plot(adjusted_dates, total_deaths, color='r')
plt.plot(adjusted_dates, total_recovered, color='green')
plt.legend(['death', 'recoveries'], loc='best')
plt.title('# of Coronavirus Cases')
plt.xlabel('Days Since 1/22/2020')
plt.ylabel('# of Cases')
plt.show()

print()

#plotting Death vs Recoveries

plt.plot(total_recovered, total_deaths)
plt.title('# of Coronavirus Deaths vs. # of Coronavirus Recoveries')
plt.xlabel('# of Coronavirus Recoveries')
plt.ylabel('# of Coronavirus Deaths')

plt.show()