# Import libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv

# Create the dataset
sun_hours = np.array([6.0, 7.0, 8.0, 5.0, 9.0, 7.0, 8.0, 6.0, 5.0, 4.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 11.0, 10.0, 9.0])
avg_humidity = np.array([50.0, 60.0, 55.0, 45.0, 65.0, 70.0, 75.0, 40.0, 55.0, 50.0, 60.0, 70.0, 65.0, 45.0, 60.0, 70.0, 75.0, 80.0, 85.0, 90.0])
avg_max_temp = np.array([30.0, 32.0, 33.0, 28.0, 34.0, 35.0, 31.0, 29.0, 27.0, 26.0, 36.0, 35.0, 33.0, 32.0, 31.0, 30.0, 29.0, 37.0, 36.0, 35.0])
errors = np.random.normal(0, 3, len(sun_hours))

real_max_temp = 0.5 * sun_hours + 0.2 * avg_humidity + 0.3 * avg_max_temp + errors

# Introduce randomly generated capture errors
for i in range(5):
    idx = np.random.randint(0, len(sun_hours))
    sun_hours[idx] = np.nan
    avg_humidity[idx] = np.nan
    avg_max_temp[idx] = np.nan

print("Step 1 completed: Data generated and errors introduced")

# Show the complete dataset before saving it to .csv
print("\nComplete Dataset:")
print("--------------------")
print("Sun Hours  | Avg Humidity | Real Max Temperature")
print("--------------------")
for i in range(len(sun_hours)):
    print(f"{sun_hours[i]:<14} | {avg_humidity[i]:<17} | {real_max_temp[i]}")

# Save the data in a .csv file
with open('test_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Sun Hours', 'Avg Humidity', 'Real Max Temperature'])
    for i in range(len(sun_hours)):
        writer.writerow([sun_hours[i], avg_humidity[i], real_max_temp[i]])

print("Step 2 completed: Data saved to test_data.csv")

# Load and clean the data
data = np.loadtxt('test_data.csv', delimiter=',', skiprows=1)
# Remove rows with null values
clean_data = data[~np.isnan(data).any(axis=1)]

print("Step 3 completed: Data loaded and cleaned")

# Create a neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=16, input_shape=(2,), activation='relu'),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

print("Step 4 completed: Model created and compiled")

# Train the model
X = clean_data[:, :2]
y = clean_data[:, 2]

training_history = model.fit(X, y, epochs=50, verbose=0)

print("Step 5 completed: Model trained")

# Make multiple predictions
new_data = np.array([[7, 60], [6, 70], [8, 55], [9, 45], [3, 80]])  # 5 data sets

predictions = model.predict(new_data)

print("Predictions:")
for i, data in enumerate(new_data):
    print(f"Sun Hours: {data[0]}, Humidity: {data[1]} => Expected max temperature: {predictions[i][0]:.2f}°C")

# Display the training plot
plt.plot(training_history.history['loss'])
plt.title('Loss during training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
