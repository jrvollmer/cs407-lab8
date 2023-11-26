import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import milestone1 as ms1
import milestone2 as ms2
import milestone3 as ms3


if __name__ == '__main__':
    acc_a = [-1, -1, 0.02]
    gyro_a = [-1, -1, 0.07]
    mag_a = [-1, -1, -1]
    show_acc = any(a >= 0 for a in acc_a)
    show_gyro = any(a >= 0 for a in gyro_a)
    show_mag = any(a >= 0 for a in mag_a)

    # Get data from csv
    df = pd.read_csv('datasets/WALKING_AND_TURNING.csv', usecols=['timestamp', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z'])

    # Change timestamps to be seconds since start
    df['timestamp'] -= df['timestamp'][0]
    df['timestamp'] /= 10**9
    time = df['timestamp'].to_numpy()

    # -------------------------------------------------------------
    # Turn Detection
    # -------------------------------------------------------------

    # Smooth angular rate of change using an EWMA with specified alpha
    smooth_gyro_z = ms2.smooth_ewma(df['gyro_z'].values, gyro_a[2])
    # Integrate angular rate of change from smoothed gyro_z to get angular displacement
    theta_z = np.concatenate(([0], ms1.integrate(smooth_gyro_z, time)))
    # Detect turns of specified increments from angular rate of change and angular displacement
    cw_turns, ccw_turns = ms3.find_turns(smooth_gyro_z, theta_z, np.pi/4, 0.125)
    cw_turns = np.array(cw_turns)
    ccw_turns = np.array(ccw_turns)
    print("CW:", cw_turns) # DEBUG
    print("CCW:", ccw_turns) # DEBUG

    # REMOVE
    # Mask timestamps for plotting turn points
    cw_tstamp = time[np.ravel(cw_turns)[::2].astype(int)]
    ccw_tstamp = time[np.ravel(ccw_turns)[::2].astype(int)]


    # -------------------------------------------------------------
    # Step Detection
    # -------------------------------------------------------------

    threshold = 9.75
    #threshold2 = 10
    accel_z = df['accel_z'].values

    # Smooth data
    # Exponential weighted moving average with alpha=0.03
    smooth_accel_z = ms2.smooth_ewma(accel_z, acc_a[2])
    # REMOVE Weighted moving average with sample size of 50
    #accel_z_filtered2 = ms2.weightedmovingaverage(accel_z, 75)

    # Count steps by counting intersections of the smoothed data with a threshold value
    num_steps = 0
    #num_steps2 = 0
    intersection_indices = []
    intersection_times = [] # REMOVE
    intersection_accs = [] # REMOVE
    #intersection_times2 = []
    #intersection_accs2 = []
    for i in range(len(time) - 1):
        if smooth_accel_z[i] <= threshold and smooth_accel_z[i+1] >= threshold:
            num_steps += 1
            intersection_indices += [i]
            intersection_times += [time[i]]
            intersection_accs += [smooth_accel_z[i]]
        #if accel_z_filtered2[i] <= threshold2 and accel_z_filtered2[i+1] >= threshold2:
        #    num_steps2 += 1
        #    intersection_times2 += [df['timestamp'].values[i]]
        #    intersection_accs2 += [accel_z_filtered2[i]]

    print(num_steps) # REMOVE
    print(intersection_times) # REMOVE
    #print(num_steps2) # REMOVE
    #print(intersection_times2) # REMOVE


    # -------------------------------------------------------------
    # Assemble Position Data
    # -------------------------------------------------------------
    
    # Current heading angle (start going north)
    heading = np.pi/2
    # Location info for plotting
    x_loc = [0]
    y_loc = [0]

    # Variables for tracking position in turn arrays
    cw_idx = 0
    ccw_idx = 0
    for i in range(len(time) - 1):
        step = 0
        # Check if we're stepping forward
        if i in intersection_indices:
            step = 1
        # Adjust heading according to turns
        if len(cw_turns) > 0 and i in cw_turns[:, 0]:
            heading += cw_turns[cw_idx, 1]
            cw_idx += 1
        elif len(ccw_turns) > 0 and i in ccw_turns[:, 0]:
            heading += ccw_turns[ccw_idx, 1]
            ccw_idx += 1
        # Move according to heading and step
        x_loc += [x_loc[-1] + step * np.cos(heading)]
        y_loc += [y_loc[-1] + step * np.sin(heading)]



    # TODO REVERT
    show_gyro = False

    # Plot results
    fig, axs = plt.subplots(
        np.sum([show_acc, show_gyro * 2, show_mag], dtype=int),
        sharex=False, sharey=False)
    fig.canvas.manager.set_window_title('Milestone 4')
    #fig.tight_layout()
    fig.canvas.manager.resize(750,750)
    plt.ylim(-5, 35)
    plt.xlim(-5, 35)
    axs.set_title('Position')
    axs.set_ylabel('Y (m)')
    axs.set_xlabel('X (m)')

    axs.plot(x_loc, y_loc, label='Position', color='#0000ff', marker='o', markersize=2, alpha=0.5)
    axs.legend()

    '''

    # Filter and plot data
    if show_acc:
        ax = axs if np.sum([show_acc, show_gyro * 2, show_mag], dtype=int) == 1 else axs[0]

        ax.set_title('Acceleration')
        ax.set_ylabel('Acceleration (m / s²)')
        ax.set_xlabel('Time (s)')

        if acc_a[0] >= 0:
            accel_x_filtered = ms2.smooth_ewma(df['accel_x'].values, acc_a[0])
            ax.plot(time, df['accel_x'].values, label='X', color='#800000')
            ax.plot(time, accel_x_filtered, label='X (filtered)', color='#ff0000')
        if acc_a[1] >= 0:
            accel_y_filtered = ms2.smooth_ewma(df['accel_y'].values, acc_a[1])
            ax.plot(time, df['accel_y'].values, label='Y', color='#008000')
            ax.plot(time, accel_y_filtered, label='Y (filtered)', color='#00ff00')
        if acc_a[2] >= 0:
            threshold_line = np.full(df['timestamp'].to_numpy().shape, threshold)
            ax.plot(df['timestamp'].values, threshold_line, label='Threshold', color='#0080ff')
            ax.fill_between(df['timestamp'].values, threshold_line-0.5, threshold_line+0.5, color='#0080ff', alpha=0.25)
            #threshold_line2 = np.full(df['timestamp'].to_numpy().shape, threshold2)
            #ax.plot(df['timestamp'].values, threshold_line2, label='Threshold 2', color='#00ff80')
            #ax.fill_between(df['timestamp'].values, threshold_line2-0.5, threshold_line2+0.5, color='#00ff80', alpha=0.25)
            ax.plot(time, df['accel_z'].values, label='Z', color='#f0f000')
            #ax.plot(time, accel_z_filtered2, label='Z (filtered 2)', color='#00ff00')
            ax.plot(time, smooth_accel_z, label='Z (filtered)', color='#ff00ff')
            #ax.vlines(intersection_times2, 8, intersection_accs2, label='Intersection', color='#ff0000')
            ax.vlines(intersection_times, 8, intersection_accs, label='Intersection', color='#0000ff')
        ax.legend()
    if show_gyro:
        idx = int(show_acc)
        axs[idx].set_title('Gyro')
        axs[idx].set_ylabel('Angular Rate of Change (rad / s)')
        axs[idx].set_xlabel('Time (s)')

        axs[idx+1].set_title('Gyro (Integrated)')
        axs[idx+1].set_ylabel('Angular Displacement (rad)')
        axs[idx+1].set_xlabel('Time (s)')

        if gyro_a[0] >= 0:
            gyro_x_filtered = ms2.smooth_ewma(df['gyro_x'].values, gyro_a[0])
            axs[idx].plot(time, df['gyro_x'].values, label='X', color='#00f080')
            axs[idx].plot(time, gyro_x_filtered, label='X (filtered)', color='#ff0000')
            theta_x = np.concatenate(([0], ms1.integrate(gyro_x_filtered, time)))
            axs[idx+1].plot(time, theta_x, label='X', color='#ff0000')
        if gyro_a[1] >= 0:
            gyro_y_filtered = ms2.smooth_ewma(df['gyro_y'].values, gyro_a[1])
            axs[idx].plot(time, df['gyro_y'].values, label='Y', color='#f00080')
            axs[idx].plot(time, gyro_y_filtered, label='Y (filtered)', color='#00ff00')
            theta_y = np.concatenate(([0], ms1.integrate(gyro_y_filtered, time)))
            axs[idx+1].plot(time, theta_y, label='Y', color='#00ff00')
        if gyro_a[2] >= 0:
            gyro_z_filtered = ms2.smooth_ewma(df['gyro_z'].values, gyro_a[2])
            axs[idx].plot(time, df['gyro_z'].values, label='Z', color='#f08000')
            axs[idx].plot(time, gyro_z_filtered, label='Z (filtered)', color='#0000ff')
            theta_z = np.concatenate(([0], ms1.integrate(gyro_z_filtered, time)))
            axs[idx+1].plot(time, theta_z, label='Z', color='#0000ff')
            axs[idx+1].vlines(np.concatenate((cw_tstamp, ccw_tstamp)), 0, np.array(cw_turns)[:,1] , label='Turns', color='#ff00ff')
        axs[idx].legend()
    if show_mag:
        idx = show_acc + show_gyro * 2
        axs[idx].set_title('Magnetometer')
        axs[idx].set_ylabel('Magnetic Flux Density (uT)')
        axs[idx].set_xlabel('Time (s)')

        if mag_a[0] >= 0:
            mag_x_filtered = np.concatenate(([0], ms2.smooth_ewma(df['mag_x'].values[1:], mag_a[0])))
            axs[idx].plot(time, df['mag_x'].values, label='X', color='#800000')
            axs[idx].plot(time, mag_x_filtered, label='X (filtered)', color='#ff0000')
        if mag_a[1] >= 0:
            mag_y_filtered = np.concatenate(([0], ms2.smooth_ewma(df['mag_y'].values[1:], mag_a[1])))
            axs[idx].plot(time, df['mag_y'].values, label='Y', color='#008000')
            axs[idx].plot(time, mag_y_filtered, label='Y (filtered)', color='#00ff00')
        if mag_a[2] >= 0:
            mag_z_filtered = np.concatenate(([0], ms2.smooth_ewma(df['mag_z'].values[1:], mag_a[2])))
            axs[idx].plot(time, df['mag_z'].values, label='Z', color='#000080')
            axs[idx].plot(time, mag_z_filtered, label='Z (filtered)', color='#0000ff')
        axs[idx].legend()
    '''

    plt.show()

'''
Idea:
    Detect turn bounds and directions by slightly smoothing gyro data and
    finding bases of peaks and troughs:
        trough bounds -> cw_turns = [(t1, t2), (t5, t6), ...]
        and
        peak bounds -> ccw_turns = [(t3, t4), (t7, t8), ...]
    Then, add a slight time margin to each bound and integrate gyro data across
    those bounds ot get turn angles. Finally, round each turn angle to the nearest
    multiple of whatever specified base angle to get the final turn angles.
'''