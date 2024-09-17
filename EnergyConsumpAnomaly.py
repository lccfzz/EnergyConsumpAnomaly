# WORK IN PROGRESS # #  ANOMALY DETECTION IN ENERGY CONSUMPTION. # # WORK IN PROGRESS
############################################
# Objective: Create a model to detect unusual patterns 
# or anomalies in energy consumption data.
############################################
# Relevance: This is crucial for maintaining grid stability and identifying potential issues in energy distribution.
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn import metrics
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from xgboost import XGBClassifier
# import lightgbm as lgb
# import tensorflow as tf
# from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM,Dense, Dropout
# %%
# Step 1: Loading & treating training data
# hourly meter readings from 200 buildings.

train_data = pd.read_csv('train.csv')

train_data.dropna(inplace=True)
train_data['timestamp'] = pd.to_datetime(train_data['timestamp'])
train_data['hour'] = train_data['timestamp'].dt.hour
train_data['day_of_week'] = train_data['timestamp'].dt.dayofweek
train_data['month'] = train_data['timestamp'].dt.month
train_data.drop(columns=['timestamp'],inplace=True)

# Step 2: Building predictive model using TensorFlow
# Loading test data

test_data = pd.read_csv('test.csv')
test_data.dropna(inplace=True)
test_data['timestamp'] = pd.to_datetime(test_data['timestamp'])
test_data['hour'] = test_data['timestamp'].dt.hour
test_data['day_of_week'] = test_data['timestamp'].dt.dayofweek
test_data['month'] = test_data['timestamp'].dt.month
test_data.drop(columns=['timestamp'], inplace=True)

X_train = train_data.drop(columns=['anomaly','meter_reading'])
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
y_train = train_data['anomaly']

X_train, X_val, y_train, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
print(f'Validation Accuracy: {accuracy:.2f}')

print(classification_report(y_val, y_pred))

conf_matrix = confusion_matrix(y_val, y_pred)
print(f'Confusion Matrix:\n{conf_matrix}')

X_test = test_data.drop(columns=['row_id', 'meter_reading'])
X_test_scaled = scaler.transform(X_test)

test_predictions = rf_classifier.predict(X_test_scaled)

test_data['anomaly_prediction'] = test_predictions

anomalous_buildings = test_data[test_data['anomaly_prediction'] == 1]['building_id'].unique()
print(f'Anomalous Building IDs: {anomalous_buildings}')

train_data_visual = pd.DataFrame(X_train, columns=train_data.drop(columns=['anomaly', 'meter_reading']).columns)
train_data_visual['meter_reading'] = train_data['meter_reading']
train_data_visual['anomaly'] = y_train.reset_index(drop=True)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Non-Anomalous', 'Predicted Anomalous'], 
            yticklabels=['Actual Non-Anomalous', 'Actual Anomalous'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

buildings_to_plot = anomalous_buildings[:3]

plt.figure(figsize=(14, 7))
for building_id in buildings_to_plot:
    building_data = train_data[train_data['building_id'] == building_id]
    plt.plot(building_data['hour'], building_data['meter_reading'], label=f'Building ID {building_id}')

plt.title('Time Series of Meter Readings for Anomalous Buildings')
plt.xlabel('Hour of the Day')
plt.ylabel('Meter Reading')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=train_data_visual, x='anomaly', y='meter_reading', palette='viridis')
plt.title('Box Plot of Meter Reading by Anomaly Status')
plt.xlabel('Anomaly Status')
plt.ylabel('Meter Reading')
plt.xticks([0, 1], ['Non-Anomalous (0)', 'Anomalous (1)'])
plt.show()
# %%
# model = Sequential()
# model.add(LSTM(128, return_sequences=True, input_shape=(X_train_scaled.shape[1], 1)))
# model.add(Dropout(0.2))
# model.add(LSTM(64, return_sequences=False))
# model.add(Dropout(0.2))
# model.add(Dense(1))

# model = [XGBClassifier(), lgb.LGBMClassifier()]

# for mod in model:
#     # Train the model
#     mod.fit(X_train_scaled, Y_train)
    
#     # Evaluate the model
#     print(f'{mod} : ')
#     print('Training ROC-AUC Score: ', metrics.roc_auc_score(Y_train, model.predict_proba(X_train)[:,1]))
#     print('Validation ROC-AUC Score: ', metrics.roc_auc_score(Y_val, model.predict_proba(X_val)[:,1]))
#     print()
# %%
# model.compile(optimizer='adam', loss='huber')

# model.summary()
# def scheduler(epoch, lr):
#     return lr * 0.9 if epoch > 5 else lr

# lr_scheduler = LearningRateScheduler(scheduler)
# early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# history = model.fit(X_train_scaled, y_train, epochs=50, validation_split=0.2, callbacks=[lr_scheduler, early_stop])
# # %%
# print(history.history)
# # %%
# X_test = test_data.drop(columns=['meter_reading'])
# X_test_scaled = scaler.fit_transform(X_test)
# y_test = test_data['meter_reading'] if 'meter_reading' in test_data.columns else None

# X_test.shape, y_test.shape if y_test is not None else "No y_test"
# test_loss = model.evaluate(X_test_scaled, y_test)
# print(f"Test Loss: {test_loss}")
# # %%
# predictions = model.predict(X_test_scaled)
# # %%

# plt.figure(figsize=(10, 6))
# plt.plot(y_test.values, label="Actual Values")
# plt.plot(predictions, label="Predicted Values")
# plt.legend()
# plt.show()


# # %%
# mae = mean_absolute_error(y_test, predictions)
# rmse = np.sqrt(mean_squared_error(y_test, predictions))

# print(f"Mean Absolute Error: {mae}")
# print(f"Root Mean Squared Error: {rmse}")
# %%
# predictions = model.predict(X_test)

# Step 3: Output & Predictions
# Questions worth answering:
# 1. Accuracy of the model in relation to test.csv.
# 2. Building_id's that can show anomalous consumption.
# 3. Likellihood of previously anomalous consuptiom to have another anomaly.
# 4. Likellihood of no anomalous buildings_id to show anomalous behaviour.
# 5. Improvements and conclusions.

# %%
# y_test = test_data['meter_reading'] if 'meter_reading' in test_data.columns else None
# num_samples = len(y_test)
# sample_size = min(5000, num_samples)
# y_test = y_test.values.flatten() 
# sample_indices = np.random.choice(num_samples, size=sample_size, replace=False)
# y_test_sampled = y_test[sample_indices]
# predictions_sampled = predictions[sample_indices].flatten()
# residuals_sampled = y_test_sampled - predictions_sampled
# plt.figure(figsize=(10, 6))
# plt.plot(residuals_sampled)
# plt.plot(residuals_sampled)
# plt.title("Residuals (Sampled)")
# plt.show()

# %%
