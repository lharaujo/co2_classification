# Libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import median_absolute_error, explained_variance_score, classification_report

from xgboost import XGBRegressor

from tensorflow.keras.models import load_model

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.models import load_model

import requests
import io

# function to import data files from github
def get_csv(file_name):
   response = requests.get(f"https://raw.githubusercontent.com/lharaujo/co2_data/main/{file_name}")
   status = response.status_code
   return pd.read_csv(io.StringIO(response.text))

# Data

# cleaned data (whole dataset)
# this function will be used more than once in the code because
def load_df():
   df = get_csv("data_cleaned_final_1.csv")
   df = pd.concat([df, get_csv("data_cleaned_final_2.csv")])
   return df
df = load_df()

# categorical data (Italian dataset)
df_cat_it = get_csv("categorical_data_it.csv")

# categorical data (whole dataset)
df_cat_1 = get_csv('categorical_data_all_1.csv').set_index('ID')
df_cat_2 = get_csv('categorical_data_all_2.csv').set_index('ID')
df_all = pd.concat([df_cat_1, df_cat_2], axis = 0)

# Layout
st.title('CO2 Emissions by Vehicles')
st.sidebar.title("Table of contents")
pages=['Introduction', 'Data', 'Model', 'Competitor Models', 'Potential Improvements', 'Contributors']
page=st.sidebar.radio("Go to", pages)

# --- Page 1  (Martha) --- 

if page == pages[0] : 
   st.header("Introduction", divider="orange")
   st.markdown(
   """
   - Aim of Project: build a model that predicts expected carbon emissions of a car
   - Caveat: Only emissions while driving the car are taken into account (_**not**_ the emissions on the whole lifecycle of a car)
   - Input Data: set of technical features of a car
   - Dataset: final dataset on CO2 emissions of new passenger cars in the EU in 2022 provided by the European Environmental Agency (EEA)
   """
   )  

# --- Page 2 (Lena) ---

if page == pages[1] : 
   st.header("Data", divider="orange")
   st.markdown(
   """
   - Dataset on CO2 emissions of new passenger cars in the EU in
   2022 provided by the European Environmental Agency (EEA)
   - From 2009 EU legislation set CO2 emission standards for new vehicles that demand manufacturers to submit detailed
   information about their new fleets on a yearly basis in order to demonstrate their compliance
   with the standards
   """
   )

   st.divider()
   st.header("_1. Raw data_")
   st.markdown(
   """
   - 38 variables 
   - 9,479,544 records
   """
   )

   # Create the DataFrame
   data_raw = {
      'Variable Number': [
         0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 
         20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37
      ],
      'Variable': [
         'ID', 'Country', 'VFN', 'Mp', 'Mh', 'Man', 'MMS', 'Tan', 'T', 'Va', 'Ve', 'Mk', 'Cn', 
         'Ct', 'Cr', 'r', 'm (kg)', 'Mt', 'Enedc (g/km)', 'Ewltp (g/km)', 'W (mm)', 'At1 (mm)', 
         'At2 (mm)', 'Ft', 'Fm', 'ec (cm3)', 'ep (KW)', 'z (Wh/km)', 'IT', 'Ernedc (g/km)', 
         'Erwltp (g/km)', 'De', 'Vf', 'Status', 'year', 'Date of registration', 'Fuel consumption', 
         'Electric range (km)'
      ],
      'Description': [
         'Identification number', 'Country', 'Vehicle family identification number', 
         'Manufacturer pooling', 'Manufacturer name EU standard denomination', 
         'Manufacturer name OEM declaration', 'Manufacturer name MS registry denomination', 
         'Type approval number', 'Type', 'Variant', 'Version', 'Make', 'Commercial name', 
         'Category of the vehicle type approved', 'Category of the vehicle registered', 
         'Total new registrations', 'Mass in running order Completed/complete vehicle', 
         'WLTP test mass', 'Emmissions according to old testing standard (NEDC)', 
         'Specific CO2 Emissions (WLTP)', 'Wheel Base (distance between axes)', 
         'Axle width steering axle', 'Axle width other axle', 'Fuel type', 'Fuel mode', 
         'Engine capacity', 'Engine power', 'Electric energy consumption', 
         'Innovative technology or group of innovative technologies', 
         'Emissions reduction through innovative technologies (NEDC)', 
         'Emissions reduction through innovative technologies (WLTP)', 'No description provided', 
         'No description provided', 'P = Provisional data F = Final data', 'Reporting year', 
         'Registration date', 'Fuel consumption', 'Electric range'
      ],
      'Data Type': [
         'int64', 'object', 'object', 'object', 'object', 'object', 'float64', 'object', 'object', 
         'object', 'object', 'object', 'object', 'object', 'object', 'int64', 'float64', 
         'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'object', 'object', 
         'float64', 'float64', 'float64', 'object', 'float64', 'float64', 'float64', 'float64', 
         'object', 'int64', 'object', 'float64', 'float64'
      ],
      'Percentage of Missing Values': [
         '0%', '0%', '1.25%', '6.55%', '0%', '0%', '100%', '0.24%', '0.04%', '0.20%', 
         '0.42%', '0.01%', '0.87%', '0.15%', '0%', '0%', '0%', '17.8%', '82.86%', '0.09%', 
         '0.44%', '15.98%', '18.78%', '0%', '0%', '13.48%', '13.63%', '77.43%', '32.66%', 
         '100%', '34.55%', '100%', '100%', '0%', '0%', '17.02%', '14.74%', '79.72%'
      ]
   }

   df_raw = pd.DataFrame(data_raw)

   # Display the DataFrame in Streamlit
   st.title("Overview of the Variables in the Raw Dataset")
   st.table(df_raw)
   st.divider()

   #######################################################################################
   st.header("_2. Data Selection_")
   st.markdown(
   """
   - Deleted variables with over 75% missing values
   - Deleted variables that have the same value in all records
   - Variable “Ewltp (g/km)” (Specific CO2 Emissions (WLTP)) is target variable 
   - Selected variables that list technical details about the vehicles as feature variables
   - Additionally kept ID, country and manufacturer as potentially useful information
   """
   )
   # Create the DataFrame
   data_selected = {
      'Variable Number': [
         0, 1, 4, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 29, 30, 36
      ],
      'Variable': [
         'ID', 'Country', 'Mn', 'm (kg)', 'Mt', 'Ewltp (g/km)', 'W (mm)', 'At1 (mm)', 
         'At2 (mm)', 'Ft', 'Fm', 'ec (cm3)', 'ep (KW)', 'IT', 'Ewltp (g/km)', 'Fuel consumption'
      ],
      'Description': [
         'Identification number', 'Country', 'Manufacturer name EU standard denomination', 
         'Mass in running order Completed/complete vehicle', 'WLTP test mass', 
         'Specific CO2 Emissions (WLTP)', 'Wheel Base (distance between axles)', 
         'Axle width steering axle', 'Axle width other axle', 'Fuel type', 'Fuel mode', 
         'Engine capacity', 'Engine power', 'Innovative technology or group of innovative technologies', 
         'Emissions reduction through innovative technologies (WLTP)', 'Fuel consumption'
      ],
      'Data Type': [
         'int64', 'object', 'object', 'float64', 'float64', 'float64', 'float64', 'float64', 
         'float64', 'object', 'object', 'float64', 'float64', 'object', 'float64', 'float64'
      ],
      'Percentage of Missing Values': [
         '0%', '0%', '0%', '0%', '17.8%', '0.09%', '0.44%', '15.96%', '18.78%', '0%', '0%', 
         '13.48%', '13.63%', '32.66%', '34.55%', '14.74%'
      ]
   }

   df_selected = pd.DataFrame(data_selected)

   # Display the DataFrame in Streamlit
   st.title("Overview of the Selected Variables in the Dataset")
   st.table(df_selected)

   st.title("Descriptive Statistics of the Selected Variables in the Dataset")
   st.dataframe(df.describe())

   st.divider()

   st.header("_3. Data Pre-Processing_")
   st.markdown(
   """
   1. Defined the variable “ID” as index
   2. Saved variables “Country” and “Mh” (manufacturer name) for later, and deleted them 
   3. Deleted duplicates keeping the first record 

   4. iterative data cleaning (order in table below):
      - missing values
      - outliers  
   5. changed eco-innovation technology categories in variable "IT" 
   """
   )

   # Create the DataFrame
   data_cleaned = {
      'Variable': [
         'Ewltp (g/km)', 'm (kg)', 'Mt', 'W (mm)', 'At1 (mm)', 'At2 (mm)', 
         'Ft', 'Fm', 'ec (cm3)', 'ep (KW)', 'Fuel consumption', 'IT', 'Ewltp (g/km)'
      ],
      'Description': [
         'Specific CO2 Emissions (WLTP)', 'Mass in running order Completed/complete vehicle', 
         'WLTP test mass', 'Wheelbase (distance between axles)', 'Axle width steering axle', 
         'Axle width other axle', 'Fuel type', 'Fuel mode', 'Engine capacity', 
         'Engine power', 'Fuel consumption', 
         'Innovative technology or group of innovative technologies', 
         'Emissions reduction through innovative technologies (WLTP)'
      ],
      'Data Type': [
         'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 
         'object', 'object', 'float64', 'float64', 'float64', 'object', 'float64'
      ],
      'Treatment of Missing Values': [
         'deleted', 'deleted', 'deleted', 'none', 'deleted', 'deleted', 
         'none', 'none', 'deleted', 'deleted', 'deleted', 'replaced with category "none"', 
         'replaced with zero'
      ],
      'Treatment of Outliers': [
         'deleted records with zero emissions, kept outliers at the lower end, deleted outliers at the upper end', 
         'deleted', 'deleted', 'deleted outliers at the upper end', 'deleted', 'deleted outliers at the upper end', 
         'categorical variable, no outliers', 'categorical variable, no outliers', 
         'deleted outliers at the lower end, deleted outliers at the upper end', 
         'deleted outliers at the lower end, deleted outliers at the upper end', 
         'kept outliers at the lower end, deleted outliers at the upper end', 
         'see details in the text', 'no outliers'
      ]
   }

   df_cleaned = pd.DataFrame(data_cleaned)

   # Display the DataFrame in Streamlit
   st.title("Overview and Order of Data Cleaning")
   st.table(df_cleaned)

   st.divider()

   st.header("_4. Relationship Between Variables_")
   st.markdown(
   """
   We performed different analyses and provide different visualisations to show the relationship between 
   the target variable and the feature variables defined above.
   """
   )

   st.subheader("_4.1 Correlation heatmap of quantitative features_")
   st.markdown(
   """
   Overview of the relationships between the target variable, emissions (Ewltp (g/km)), and 
   quantitative features:  
   - Fuel consumption is most significant predictor of emissions due to its strong positive correlation
   - Other variables exhibit weak to moderate correlations with emissions
   - Strong correlations between certain pairs of independent variables, such as vehicle mass and
   test mass, and between axle widths.
   """
   )

   st.title("_Correlation heatmap of quantitative features_")
   #st.caption("The orange dashed lines denote the 20% percentiles (124, 134, 146, 159).")

   # separating qualitative and quantiative variables
   df_analysis_quant = df[['ewltp', 'm', 'mt', 'w', 'at1', 'at2', 'ec', 'ep', 'erwltp', 'fuel_consumption']]
   df_analysis_qual = df[['ewltp', 'ft', 'fm', 'only_it']]

   ########################################################################  
   # Calculate the correlation matrix
   corr = df_analysis_quant.corr()
   
   # Create the heatmap using Seaborn and Matplotlib
   fig, ax = plt.subplots()
   sns.heatmap(corr, cmap="Blues", annot=True, ax=ax)
   plt.xticks(rotation=45)
   plt.title("Correlation heatmap between the target (Ewltp (g/km)) and the quantitative features")

   # Display the heatmap in the Streamlit app
   st.pyplot(fig)

   #######################################################################
   st.subheader("_4.2 Analysis of CO2 Emissions in Relation to Fuel Consumption and Qualitative Variables_")

   st.title("Scatter Plots of CO2 Emissions")

   # Creating the subplot
   fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

   # Scatter plot for emissions based on fuel type
   sns.scatterplot(ax=axes[0], data=df, x='fuel_consumption', y='ewltp', hue='ft')
   axes[0].set_title('CO2 emissions based on fuel consumption and fuel type')

   # Scatter plot for emissions based on fuel mode
   sns.scatterplot(ax=axes[1], data=df, x='fuel_consumption', y='ewltp', hue='fm')
   axes[1].set_title('CO2 emissions based on fuel consumption and fuel mode')

   # Showing the plot in Streamlit
   st.pyplot(fig)

   st.markdown(
   """
   Fuel type:
   - Diesel and petrol vehicles generally exhibit high emissions and fuel consumption
   - LPG and natural gas (NG) vehicles have lower emissions and fuel consumption
   - Hybrid and electric vehicles lower emissions and fuel consumption

   Fuel mode:  
   - Mono-fuel vehicles show a broad range of emissions and fuel consumption 
   - Bi-fuel vehicles have slightly lower emissions compared to mono-fuel vehicles at similar fuel
   consumption levels
   - Hybrid electric tend to have lower emissions and are more fuel-efficient
   - Flex-fuel vehicles display varied emissions and fuel consumption but generally lower than mono-fuel vehicles
   """
   )

   st.divider()
   st.header("_5. Final Pre-Processed Dataset for Model Training_")
   st.markdown(
   """
   - 13 variables 
   - 643,463 records
   """
   )
# --- Page 3 (Martha) ---

if page == pages[2] : 
   st.header("XG Boost Classification Model", divider="orange")

   st.subheader("_1. Summary_")
   st.markdown(
   """
   - Predicts whether expected carbon emissions are very low, low, average, high or very high
   - Needs only 6 technical features as input data
   - Can be trained in less than 1.5 minutes (CPU time)
   - Performs exceptionally well in terms of accuracy, precision, recall and F1-score (96%)
   """
   )

   st.divider()
   
   # --- XG Boost Classification Model ---

   # new compound variable for axle widths
   df_all['at_sum'] = df_all['at1'] + df_all['at2']
   df_all = df_all.drop(columns = ['at1', 'at2'])

   # target variable
   target = df_all['emissions_cat']

   # features selected by the Extra Tree classifier
   features_etc = ['m', 'w', 'ep', 'fuel_consumption', 'at_sum']
   data_etc = df_all[features_etc]

   # train and test sets
   X_train_etc, X_test_etc, y_train_etc, y_test_etc  = train_test_split(data_etc, target, test_size = 0.2, random_state = 42)

   # scale
   scaler = StandardScaler()
   X_train_scaled_etc = scaler.fit_transform(X_train_etc)
   X_test_scaled_etc = scaler.transform(X_test_etc)
   st.write("### Introduction")

   st.subheader("_2. Target Variable: 5 Classes for CO2 Emissions_")

   # plot CO2 emissions
   fig = plt.figure()
   sns.kdeplot(df['ewltp'], fill = False, cut = 0)
   plt.axvline(x= 124.0, ymin = 0, ymax = 0.71, c = 'darkorange', ls = '--', lw = 1)
   plt.axvline(x= 134.0, ymin = 0, ymax = 0.95, c = 'darkorange', ls = '--', lw = 1)
   plt.axvline(x= 146.0, ymin = 0, ymax = 0.832, c = 'darkorange', ls = '--', lw = 1)
   plt.axvline(x= 159.0, ymin = 0, ymax = 0.54, c = 'darkorange', ls = '--', lw = 1)
   plt.title('Distribution of CO2 Emissions')
   plt.xlabel('CO2 Emissions')
   st.pyplot(fig)

   st.caption("The orange dashed lines denote the 20% percentiles (124, 134, 146, 159).")
   
   # --- Table with Categories ---
   a = [['very low', 0, '12 - 124'],
        ['low', 1, '124 - 134'],
        ['average', 2, '134 - 146'],
        ['high', 3, '146 - 159'],
        ['very high', 4, '159 - 233']]

   # create dataframe
   a_df = pd.DataFrame(a, columns = ['Description', 'Class', 'Value Range (in grams per kilometer)'])

   # display table
   st.write("CO2 Emissions Classes", a_df)
   # st.table(a_df)

   # st.write(":orange[---]")  

   st.divider()

   # ---- Italian Data --- 
   st.subheader("_3. The Italian Dataset (7.8% of the Data)_")

   # plot
   fig = plt.figure()
   sns.countplot(x = "emissions_cat", data = df_cat_it)
   plt.xlabel('CO2 Emissions Categories')
   # plt.title("CO2 Emissions by Category (Italian Cars)")
   st.pyplot(fig)

   st.divider()

   # ---- Feature Variables --- 
   st.subheader("_4. Feature Variables_")

   st.markdown(
   """
   1. One-hot encode categorical variables
   2. Split the data into a training and test set (20%)
   3. Standardize numerical data
   4. Train base models (with the Italian dataset and the whole dataset)
      - a. Grid search with KNN and Random Forest models 
      - b. Boosting algorithms: Ada Boost and XG Boost
   5. Select features
      - a. Drop one of the two variables for vehicle mass
      - b. Add the axle widths
      - c. Use RFECV (with the Italian dataset) to find optimal number of feature variables 
   """
   )

   # choose classifier 
   model_choice = ['Extra Tree', 'Random Forest', 'Decision Tree', 'XG Boost' ]
   option = st.selectbox('Choose the model for RFECV:', model_choice)

   def num_features(classifier):
      if classifier == 'Extra Tree':
         n = 6
      if classifier == 'Random Forest':
         n = 7
      elif classifier == 'Decision Tree':
         n = 8
      elif classifier == 'XG Boost':
         n = 9
      return n
   
   def num_feature_variables(classifier):
      if classifier == 'Extra Tree':
         n = 5
      if classifier == 'Random Forest':
         n = 6
      elif classifier == 'Decision Tree':
         n = 7
      elif classifier == 'XG Boost':
         n = 14
      return n
   
   def features(classifier):
      if classifier == 'Extra Tree':
         k = 'Fuel Consumption, Vehicle Mass, Engine Power, Wheelbase, Sum of the two Axle Widths'
      if classifier == 'Random Forest':
         k = 'Fuel Consumption, Vehicle Mass, Engine Power, Wheelbase, Sum of the two Axle Widths, Engine Capacity'
      elif classifier == 'Decision Tree':
         k = 'Fuel Consumption, Vehicle Mass, Engine Power, Wheelbase, Sum of the two Axle Widths, Engine Capacity, Fuel Type'
      elif classifier == 'XG Boost':
         k = 'Fuel Consumption, Vehicle Mass, Engine Power, Wheelbase, Sum of the two Axle Widths, Engine Capacity, Fuel Type, Fuel Mode'
      return k

   def show(clf, choice):
      if choice == 'Number of Features':
         return num_features(clf)
      elif choice == 'Features':
         return features(clf)
      elif choice == 'Number of Feature _Variables_':
         return num_feature_variables(clf)

   display = st.radio('RFECV Results:', ('Number of Features', 'Features', 'Number of Feature _Variables_'))
   if display == 'Number of Features':
      st.write(show(option, display))
   if display == 'Features':
      st.write(show(option, display)) 
   elif display == 'Number of Feature _Variables_':
      st.write(show(option, display))

   st.divider()

   # --- Model Specification---
   st.subheader("_5. Model Specification_")

   st.markdown(
   """
   - XG Boost Classification Model
   - Default Parameters
   - Size of Test Set: 20%
   - Input Data: Whole dataset with the following 5 feature _variables_:
      - Fuel Consumption 
      - Vehicle Mass 
      - Engine Power 
      - Wheelbase 
      - Sum of the two Axle Widths
   """
   )

   st.divider()

   # --- Model Performnace Evaluation---
   st.subheader("_6. Model Performance Evaluation_")

   # load model from file
   clf_loaded = joblib.load('xg_boost_clf_model')

   # predict
   y_pred = clf_loaded.predict(X_test_scaled_etc)

   # classification report
   r = pd.DataFrame(classification_report(y_test_etc, y_pred, output_dict=True))
   r = r.head(3)

   # confusion matrix
   cm = pd.crosstab(y_test_etc, y_pred, rownames = ['Real / Predicted'])

   display = st.radio('Choose:', ('Classification Report', 'Confusion Matrix'))
   if display == 'Classification Report':
      st.dataframe(r)
   elif display == 'Confusion Matrix':
      st.dataframe(cm)

   st.divider()

   # --- Feature Importances --- 
   st.subheader("_7. Feature Importances_")
   # data for feature importances
   data = [['Fuel Consumption', 0.586], 
           ['Vehicle Mass', 0.204], 
           ['Engine Power', 0.091],
           ['Wheelbase', 0.062],
           ['Sum of Axle Widhts', 0.057]]

   # create dataframe
   fi = pd.DataFrame(data, columns = ['Feature', 'Score'])

   # plot feature importances
   # plot
   fig = plt.figure()
   sns.barplot(fi, y = "Feature", x = "Score", color = 'steelblue')
   # plt.title("Feature Importances")
   plt.ylabel('')
   st.pyplot(fig)

# --- Page 4 (Marius and Leo) ---

if page == pages[3] : 
   st.header("Competitor Models", divider='orange')
   options = ["Coose a Model", "MLP Classification Model", "XG Boost Regression Model"]

   selected_option = st.selectbox('', options = options)
# --- Marius ---  
   if selected_option == "MLP Classification Model":
      # 2. Data Loading and Preprocessing
      st.header("Data Loading and Preprocessing")

      # Load the dataset
      df = load_df().set_index("ID")

      st.subheader("First few rows of the dataset:")
      st.write(df.head(10))

      # Calculate percentiles for 'ewltp' column
      p1, p2, p3, p4 = df['ewltp'].quantile(q=[0.2, 0.4, 0.6, 0.8])

      # Create 'emissions_cat' column based on percentiles
      df['emissions_cat'] = pd.cut(x=df['ewltp'],
                                 bins=[0, p1, p2, p3, p4, df['ewltp'].max()],
                                 labels=[1, 2, 3, 4, 5])

      st.subheader("Data after processing 'emissions_cat':")
      st.write(df.head(10))

      # Feature engineering
      df['at_sum'] = df['at1'] + df['at2']
      features = ['m', 'w', 'ep', 'fuel_consumption', 'at_sum']
      X = df[features]
      y = df['emissions_cat'].astype('int') - 1  # Adjust labels to start from 0

      # Normalize the features
      scaler = MinMaxScaler()
      X_scaled = scaler.fit_transform(X)

      # Split the data into training and testing sets
      X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            
      # 4. Model Evaluation
      st.subheader("Model Evaluation")

      # Load the saved model
      mlp_model = load_model("mlp_model.h5") 

      # Evaluate the model
      loss_mlp, accuracy_mlp = mlp_model.evaluate(X_test, y_test)
      st.write(f'MLP Accuracy on test set: {accuracy_mlp * 100:.2f}%')

      # Make predictions
      predictions_mlp = mlp_model.predict(X_test)
      predicted_classes_mlp = np.argmax(predictions_mlp, axis=1)

      # Classification report
      st.subheader("Classification Report")
      st.text(classification_report(y_test, predicted_classes_mlp))

      # Confusion matrix
      conf_matrix_mlp = confusion_matrix(y_test, predicted_classes_mlp)
      st.subheader("Confusion Matrix")
      st.write(conf_matrix_mlp)

      # Plot the confusion matrix
      st.subheader("Confusion Matrix Heatmap")
      plt.figure(figsize=(10, 7))
      sns.heatmap(conf_matrix_mlp, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1, 2, 3, 4], yticklabels=[0, 1, 2, 3, 4])
      plt.xlabel('Predicted')
      plt.ylabel('True')
      plt.title('MLP Confusion Matrix')
      st.pyplot(plt)

      # 5. Additional Metrics
      st.header("Additional Metrics")
      precision_mlp = precision_score(y_test, predicted_classes_mlp, average='weighted')
      recall_mlp = recall_score(y_test, predicted_classes_mlp, average='weighted')
      f1_mlp = f1_score(y_test, predicted_classes_mlp, average='weighted')
      roc_auc_mlp = roc_auc_score(y_test, predictions_mlp, multi_class='ovr', average='weighted')

      st.write(f'MLP Precision: {precision_mlp:.2f}')
      st.write(f'MLP Recall: {recall_mlp:.2f}')
      st.write(f'MLP F1 Score: {f1_mlp:.2f}')
      st.write(f'MLP ROC AUC: {roc_auc_mlp:.2f}')

      # 6. Visualizations
      st.header("### Model Training History")
      # Plot accuracy history
      plt.figure(figsize=(12, 6))
      plt.plot(history_mlp.history['accuracy'])
      plt.plot(history_mlp.history['val_accuracy'])
      plt.title('MLP Model accuracy')
      plt.ylabel('Accuracy')
      plt.xlabel('Epoch')
      plt.legend(['Train', 'Validation'], loc='upper left')
      st.pyplot(plt)

      # Plot loss history
      plt.figure(figsize=(12, 6))
      plt.plot(history_mlp.history['loss'])
      plt.plot(history_mlp.history['val_loss'])
      plt.title('MLP Model loss')
      plt.ylabel('Loss')
      plt.xlabel('Epoch')
      plt.legend(['Train', 'Validation'], loc='upper right')
      st.pyplot(plt)
      
# --- Leo ---
   if selected_option == "XG Boost Regression Model":
      st.markdown(
      """
      The XGBoost Regression model was optimized using GridSearchCV to 
      fine-tune its parameters. The best-performing configuration was 
      identified with the following parameters:
      - colsample_bytree = 0.8
      - learning_rate = 0.1
      - max_depth = 10
      - n_estimators = 200
      - subsample = 1.0
      """
   )
      st.divider()
      # Load model
      @st.cache_data
      def load_model(model_name):
         model = joblib.load(model_name)
         return model
      
      # loading model
      final_model = load_model("XG_Boost_regression_model.pkl")

      # load data
      df = load_df()
      
      # One-hot encoding of the three categorical variables Ft, Fm and only_IT
      df_encoded = pd.get_dummies(df, columns=['ft','fm','only_it'])

      # Features and target variable selection. We also drop "fuel_consumption" due
      # to its special linear relationship and high correlation to the target variable.
      # In a previous step of our analysis we showed that excluding this variable for model training 
      # does not reduce model performance.
      X = df_encoded.drop(['it', 'mh', 'ewltp', 'country','fuel_consumption'], axis=1)
      y = df_encoded['ewltp']

      # Feature reduction by summarizing redundant and highly correlated features measuring the axis width and the vehiacle mass
      X['m_total'] = X['m'] + X['mt']
      X['at_total'] = X['at1'] + X['at2']
      X = X.drop(['m', 'mt', 'at1', 'at2'], axis=1)

      # slitting into train and test set
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

      # Standardizing the data with a standard scaler
      scaler = StandardScaler()
      X_train[X_train.columns] = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index)
      X_test[X_test.columns] = pd.DataFrame(scaler.transform(X_test), index=X_test.index)
      
      # stating list of features chosen by rfecv
      columns_rfecv_et = ['w', 'ec', 'ep', 'erwltp', 'ft_diesel', 'ft_petrol', 'fm_H', 'fm_M',
         'fm_P', 'only_it_33 37', 'only_it_37', 'only_it_none', 'm_total', 'at_total']

      # removing the feature fm_P from list because it is too important
      # and can generate bias in the model
      columns_rfecv_et_noFmP = ['w', 'ec', 'ep', 'erwltp', 'ft_diesel', 'ft_petrol', 'fm_H', 'fm_M',
         'only_it_33 37', 'only_it_37', 'only_it_none', 'm_total', 'at_total']
      
      # selecting the features in the data datasets
      X_train_et_noFmP = X_train[columns_rfecv_et_noFmP]
      X_test_et_noFmP = X_test[columns_rfecv_et_noFmP]

      # Predict on the test set
      @st.cache_data
      def model_pred(_final_model_, X_test_):
         prediction = _final_model_.predict(X_test_)
         return prediction
      
      # generating predictions from the model
      #y_pred = model_pred(final_model, X_test_et_noFmP) # remove the first pound sign to run the model
      #np.save( "xg_boost_regression_y_pred.npy", y_pred) # remove the first pound sign to save the prediction
      y_pred = np.load("xg_boost_regression_y_pred.npy")
      
      # defininf a function to evaluate the model with several metrics
      @st.cache_data
      def evaluate_model(y_test, y_pred):
         mae = mean_absolute_error(y_test, y_pred)
         mse = mean_squared_error(y_test, y_pred)
         rmse = np.sqrt(mse)
         r2 = r2_score(y_test, y_pred)
         median_ae = median_absolute_error(y_test, y_pred)
         evs = explained_variance_score(y_test, y_pred)

         # creating a dictionary to display results
         data = {
            'Method': ["R^2 Score", "Mean Squared Error", "Root Mean Squared Error",
                        "Mean Absolute Error", "Median Absolute Error", "Explained Variance Score"],
            'Value': [r2, mse, rmse, mae, median_ae, evs]
         }
         # creating a dataframe with the evaluation metrics and values
         df_ = pd.DataFrame(data)

         # rounding the values to only 4 decimals
         df_['Value'] = df_.Value.round(4).astype(str)
         df_.to_csv("df_xgboost_model_evaluation.csv", index = False)
             
      # Evaluate the model using the function evaluate_model
      st.subheader('_1. MODEL PERFORMANCE_')
      #evaluate_model(y_test, y_pred) # Remove the first pund sign to run the function
      
      # displaying model evaluation dataframe
      df_ = pd.read_csv("df_xgboost_model_evaluation.csv")
      st.table(df_)

      # Plot residuals
      residuals = y_test - y_pred
      fig, ax = plt.subplots(figsize=(10,6))
      sns.scatterplot(x = y_pred, y = residuals, ax=ax)
      ax.axhline(y=0, color='r', linestyle='--')
      ax.set_xlabel('Predicted')
      ax.set_ylabel('Residuals')
      ax.set_title("Residuals vs Predicted")
      st.pyplot(fig)

      st.divider()
      # Get the feature importances
      xgb_feature_importances = final_model.feature_importances_
      feature_names = X_train_et_noFmP.columns

      # Create a DataFrame to display the feature importances
      importances_df = pd.DataFrame({
         'Feature': feature_names,
         'Importance': xgb_feature_importances
      }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
      
      st.subheader('_2. FEATURE IMPORTANCE_')
      st.markdown("""The feature variables, listed below, were identified through a 
   RFECV search and further model evaluations.""")

      # Plot the feature importances as a bar plot
      fig, ax = plt.subplots(figsize=(10, 6))
      sns.barplot(y = importances_df['Feature'], x = importances_df['Importance'], ax = ax, orient = 'h')
      ax.set_xlabel('Importance')
      ax.set_ylabel('Feature')
      ax.set_title('Feature Importances Final Model')
      st.pyplot(fig)

# --- Page 5 (Marius) ---

if page == pages[4] : 
   st.header('Potential Improvements', divider="orange")
   st.write("""
   Our classification model could be improved in two main ways:

   1. **Better evaluation metric scores** could be achieved with an XGBoost model using customized parameters.
   2. **Predictions at different levels of granularity** could be provided with more or fewer classes in the target variable, depending on the client's needs.
   """)

   # Discussion of XGBoost Model Improvement
   st.subheader("Improving the XGBoost Model with Grid Search")
   st.write("""
   The current XGBoost model uses default parameters, which provide a baseline performance. To achieve better evaluation metrics, we could perform a grid search to fine-tune the hyperparameters. This process would identify the optimal settings for the model, potentially leading to improved accuracy, precision, recall, and F1 scores.
   """)

   # Example of Grid Search for XGBoost (This is an illustrative example; in a real scenario, you'd need to run this with your data)

   st.write("Example of hyperparameters we might explore in a grid search:")
   params = {
      'n_estimators': [100, 200, 300],
      'max_depth': [3, 5, 7],
      'learning_rate': [0.01, 0.1, 0.2],
      'subsample': [0.8, 1.0],
      'colsample_bytree': [0.8, 1.0]
   }
   for key, values in params.items():
      st.write(f"{key}: {values}")

   st.write("""
   This grid search would evaluate different combinations of these hyperparameters to identify the best model configuration. The optimized model is likely to outperform the default XGBoost model, especially if the dataset is complex or imbalanced.
   """)

   # Discussion of Granularity in Predictions
   st.subheader("Adjusting Prediction Granularity")
   st.write("""
   Our current classification model predicts expected CO2 emissions at five levels: very low, low, average, high, and very high. However, the granularity of these predictions can be adjusted based on client needs.

   For example:
   - **More detailed predictions** could be made by creating ten classes (10% percentiles).
   - **Rougher predictions** could be made by reducing the number of classes to three (33.3% percentiles).

   This flexibility allows the model to be tailored to different business requirements, providing either more precision or more generalized predictions, depending on the use case.
   """)

# --- Page 6 (Leo) ---

if page == pages[5] :
   st.header('Contributors', divider="orange")
   st.markdown("""
                - Lena Andreessen
                - Leo Hollnagel de Araújo
                - Marius Höckel
                - Martha Loewe
   """)
   st.write("#### Mentor")
   st.write("* Romain Lesieur")

