datafields = "../data/data/train.csv"

nav_image = "/images/prophet_output.png"
home_image = "/images/prophet_output.png"

quater_1 = "path to quater 1"
quater_2 = "path to quater 2"
quater_3 = "path to quater 3"
quater_4 = "path to quater 4"

quater_1_datasales = "path to quater_data_sales 1"
quater_2_datasales = "path to quater_data_sales 2"
quater_3_datasales = "path to quater_data_sales 3"
quater_4_datasales = "path to quater_data_sales 4"

features = ['Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday',
            'StoreType', 'Assortment', 'CompetitionDistance',
            'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2',
            'Promo2SinceWeek', 'Promo2SinceYear']

salesDependingFeatures = [1, 5, 1, '0', 1,
                          'c', 'a', 1270.0, 9.0, 2008.0, 0, 1, 1]

model_load = "/models/2022_09_10-01_58_24_PM_randomforest.pkl"
