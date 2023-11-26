import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Smoothing factor (0 <= alpha <= 1)
alpha = 0.03
# Threshold for step detection
threshold = 10.5
threshold2 = 11.5

def smooth_ewma(data, a):
    '''
    Smooth the provided data using an exponential weighted moving average

    Params:
        data: The raw data to be smoothed
        a: alpha value for the EWMA
    '''
    assert 0 <= a <= 1
    # Apply the smoothing algorithm element-wise to the array, where:
    #   - s is the previous element, which has already been smoothed
    #   - x is the current element to be smoothed
    return np.frompyfunc(lambda s,x: a * x + (1 - a) * s, 2, 1).accumulate(data)

# FIXME This doesn't seem like it's the same as what was shown in lecture, but it provides better results than the EWMA
# Based off https://www.askpython.com/python/weighted-moving-average
def weightedmovingaverage(data, k):
    weighted = []
    # Extend the start of data backwards to ensure that there are enough values
    # to average at the start
    data = np.insert(data, 0, np.full(k - 1, data[0]))
    for i in range(k-1, len(data)):
        # Get weights and most recent period of values
        total = np.arange(1, k + 1, 1)
        sample = data[i-k+1:i+1]
        # Calculate the weighted moving average for the current period
        sample = total * sample
        wma = sample.sum() / total.sum()
        # Add the weighted moving average to the results
        weighted.append(wma)
    return weighted

if __name__ == '__main__':
    # Get data
    # usecols is specified to handle the trailing commas in the dataset
    df = pd.read_csv('datasets/WALKING.csv', usecols=['timestamp', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z'])
    accel_z = df['accel_z'].values

    # Smooth data
    # Exponential weighted moving average with alpha=0.03
    accel_z_filtered = smooth_ewma(accel_z, 0.03)
    # Weighted moving average with sample size of 50
    accel_z_filtered2 = weightedmovingaverage(accel_z, 50)

    df['timestamp'] -= df['timestamp'][0]
    df['timestamp'] /= 10**9

    # Count steps by counting intersections of the smoothed data with a threshold value
    num_steps = 0
    num_steps2 = 0
    intersection_times = []
    intersection_times2 = []
    for i in range(len(df['timestamp'].values) - 1):
        if accel_z_filtered[i] <= threshold and accel_z_filtered[i+1] >= threshold:
            num_steps += 1
            intersection_times += [df['timestamp'].values[i]]
        if accel_z_filtered2[i] <= threshold2 and accel_z_filtered2[i+1] >= threshold2:
            num_steps2 += 1
            intersection_times2 += [df['timestamp'].values[i]]

    print(num_steps) # REMOVE
    print(intersection_times) # REMOVE
    print(num_steps2) # REMOVE
    print(intersection_times2) # REMOVE


    # Plot results
    fig, ax = plt.subplots(1)
    fig.canvas.manager.set_window_title('Milestone 2')
    fig.tight_layout()
    ax.set_title('Acceleration')
    ax.set_ylabel('Acceleration (m / s²)')
    ax.set_xlabel('Time (s)')
    ax.plot(df['timestamp'].values, accel_z, label='Z', color='#f08000')
    ax.plot(df['timestamp'].values, accel_z_filtered, label='Z (EWMA)', color='#00f0ff')
    threshold_line = np.full(df['timestamp'].to_numpy().shape, threshold)
    ax.plot(df['timestamp'].values, threshold_line, label='Threshold (EWMA)', color='#0080ff')
    ax.fill_between(df['timestamp'].values, threshold_line-0.5, threshold_line+0.5, color='#0080ff', alpha=0.25)
    ax.plot(df['timestamp'].values, accel_z_filtered2, label='Z (WMA)', color='#d000d0')
    threshold_line2 = np.full(df['timestamp'].to_numpy().shape, threshold2)
    ax.plot(df['timestamp'].values, threshold_line2, label='Threshold (WMA)', color='#00ff80')
    ax.fill_between(df['timestamp'].values, threshold_line2-0.5, threshold_line2+0.5, color='#00ff80', alpha=0.25)
    ax.legend()

    plt.show()
