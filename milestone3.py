import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import milestone1 as ms1

def find_turns(data, time):
    tolerance = np.pi/64
    cw_turns = []
    ccw_turns = []
    last_half_pi_angle = 0
    idx = 0

    while True:
        # Find the next index of data where the angle changes by pi/2
        idx_tmp = np.argmax(data[idx:] >= last_half_pi_angle + np.pi/2 - tolerance)
        if idx_tmp != 0:
            # Found CCW turn
            last_half_pi_angle += np.pi/2
            ccw_turns.append(time[idx + idx_tmp])
        else:
            idx_tmp = np.argmax(data[idx:] <= last_half_pi_angle - np.pi/2 + tolerance)
            if idx_tmp != 0:
                # Found CW turn
                last_half_pi_angle -= np.pi/2
                cw_turns.append(time[idx + idx_tmp])
            else:
                # No turns found
                return cw_turns, ccw_turns
        # Move further along in the data
        idx += idx_tmp

if __name__ == '__main__':
    # Get data from csv
    df = pd.read_csv('datasets/TURNING.csv', usecols=['timestamp', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z'])

    # Change timestamps to be seconds since start
    df['timestamp'] -= df['timestamp'][0]
    df['timestamp'] /= 10**9
    # Integrate angular rate of change from gyro_z to get angular displacement
    theta_z = np.concatenate(([0], ms1.integrate(df['gyro_z'].values, df['timestamp'].values)))
    # Detect turns from angular displacement
    cw_turns, ccw_turns = find_turns(theta_z, df['timestamp'].values)
    # Mask theta_z to only show turns
    cw_turn_theta_z = np.ma.masked_array(theta_z, mask=~np.isin(df['timestamp'].values, cw_turns))
    ccw_turn_theta_z = np.ma.masked_array(theta_z, mask=~np.isin(df['timestamp'].values, ccw_turns))

    # Plot angular displacement and turns
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title('Milestone 3')
    fig.tight_layout()
    ax.set_title('Angular Offset')
    ax.set_ylabel('Angular Offset (rad)')
    ax.set_xlabel('Time (s)')
    ax.plot(df['timestamp'].values, theta_z, label='Theta_z', color='#0000ff')
    ax.plot(df['timestamp'].values, cw_turn_theta_z, label='CW Turns', color='#ff0000', marker='o', markersize=10)
    ax.plot(df['timestamp'].values, ccw_turn_theta_z, label='CCW Turns', color='#00ff00', marker='o', markersize=10)
    ax.legend()

    plt.show()