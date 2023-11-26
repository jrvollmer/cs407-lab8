import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import milestone1 as ms1
import milestone2 as ms2

def find_turns(gyro_data, data, turn_increment, threshold=0.25, tolerance=np.pi/32):
    # Limit the tolerance to at most half the turn increment 
    tolerance = (turn_increment / 2) if (turn_increment <= tolerance * 2) else tolerance
    cw_bounds, ccw_bounds = get_turn_bounds(gyro_data, threshold)
    cw_turns = []
    ccw_turns = []
    idx = 0

    # Record CW turns
    for i in range(len(cw_bounds)):
        # Bounds and initial angle of the current turn
        idx = cw_bounds[i][0]
        turn_end = cw_bounds[i][1]
        init_angle = data[idx]
        increments = 0
        '''
        print(init_angle, ":", data[idx], "-", data[turn_end - 1], "=", data[idx] - data[turn_end - 1])
        print(turn_increment, tolerance)
        print(data[idx:turn_end] <= init_angle - turn_increment + tolerance)
        '''
        # Find the next index of data where the angle changes by turn_increment
        idx_tmp = np.argmax(data[idx:turn_end] <= init_angle - turn_increment + tolerance)
        while idx_tmp != 0:
            print(init_angle, "-", data[turn_end - 1], "=", init_angle - data[turn_end - 1])
            idx += idx_tmp
            increments += 1
            idx_tmp = np.argmax(data[idx:turn_end] <= init_angle - (increments+1) * turn_increment + tolerance)
        if increments > 0:
            # Record middle index and turn angle
            cw_turns.append([np.floor((cw_bounds[i][0] + turn_end) / 2).astype(int), -increments * turn_increment])
    # Record CCW turns
    for i in range(len(ccw_bounds)):
        # Bounds and initial angle of the current turn
        idx = ccw_bounds[i][0]
        turn_end = ccw_bounds[i][1]
        init_angle = data[idx]
        increments = 0
        '''
        print(init_angle, ":", data[idx], "-", data[turn_end - 1], "=", data[idx] - data[turn_end - 1])
        print(turn_increment, tolerance)
        print(data[idx:turn_end] >= init_angle + turn_increment - tolerance)
        '''
        # Find the next index of data where the angle changes by turn_increment
        idx_tmp = np.argmax(data[idx:turn_end] >= init_angle + turn_increment - tolerance)
        while idx_tmp != 0:
            print(init_angle, "-", data[turn_end - 1], "=", init_angle - data[turn_end - 1])
            idx += idx_tmp
            increments += 1
            idx_tmp = np.argmax(data[idx:turn_end] >= init_angle + (increments+1) * turn_increment - tolerance)
        if increments > 0:
            # Record middle index and turn angle
            ccw_turns.append([np.floor((ccw_bounds[i][0] + turn_end) / 2).astype(int), increments * turn_increment])
    return cw_turns, ccw_turns


def get_turn_bounds(data, threshold=0.25):
    '''
    Returns two 2D arrays containing the start and end indices of each CW and CCW turn
 
    data: array containing angular rate of change data from which turn bounds will be determined
    threshold: the minimum angular rate of change (rad/s) to be considered the start/end of a turn
    '''
    cw_turn_bounds = []
    ccw_turn_bounds = []
    i = 0
    while i < len(data):
        # Step through the data until the start of a turn is reached
        if data[i] <= -threshold:
            # Record the start and (temporary) end bounds of the CW turn
            cw_turn_bounds.append([i,i])
            # Step through the data until the end of the CW turn is reached
            while (i < len(data)) and (data[i] <= -threshold):
                i += 1
            # Record the actual end index of the CW turn
            cw_turn_bounds[-1][1] = i
        if data[i] >= threshold:
            # Record the start and (temporary) end bounds of the CCW turn
            ccw_turn_bounds.append([i,i])
            # Step through the data until the end of the CCW turn is reached
            while (i < len(data)) and (data[i] >= threshold):
                i += 1
            # Record the actual end index of the CCW turn
            ccw_turn_bounds[-1][1] = i
        i += 1
    return cw_turn_bounds, ccw_turn_bounds


def do_the_thing(df, alpha, turn_inc, threshold=0.25, tolerance=np.pi/32):
    # Change timestamps to be seconds since start
    df['timestamp'] -= df['timestamp'][0]
    df['timestamp'] /= 10**9
    time = df['timestamp'].to_numpy()
    # Smooth angular rate of change using an EWMA with specified alpha
    smooth_gyro_z = ms2.smooth_ewma(df['gyro_z'].values, alpha)
    # Integrate angular rate of change from smoothed gyro_z to get angular displacement
    theta_z = np.concatenate(([0], ms1.integrate(smooth_gyro_z, time)))
    # Detect turns of specified increments from angular rate of change and angular displacement
    cw_turns, ccw_turns = find_turns(smooth_gyro_z, theta_z, turn_inc, threshold, tolerance)
    # Mask theta_z to only show turns REMOVE
    print("CW:", cw_turns) # DEBUG
    print("CCW:", ccw_turns) # DEBUG
    cw_turn_theta_z = theta_z[np.ravel(cw_turns)[::2].astype(int)]
    ccw_turn_theta_z = theta_z[np.ravel(ccw_turns)[::2].astype(int)]

    # Mask timestamps for plotting turn points
    cw_tstamp = time[np.ravel(cw_turns)[::2].astype(int)]
    ccw_tstamp = time[np.ravel(ccw_turns)[::2].astype(int)]
    
    # Plot results
    fig, axs = plt.subplots(2, sharex=False, sharey=False)
    fig.canvas.manager.set_window_title('Milestone 3')
    fig.tight_layout()

    axs[0].set_title('Angular Rate of Change')
    axs[0].set_ylabel('Angular Rate of Change (rad / s)')
    axs[0].set_xlabel('Time (s)')
    threshold_line = np.full(df['timestamp'].to_numpy().shape, threshold)
    axs[0].plot(df['timestamp'].values, threshold_line, label='Threshold (+)', color='#0080ff')
    axs[0].fill_between(df['timestamp'].values, threshold_line*0.5, threshold_line*1.5, color='#0080ff', alpha=0.25)
    axs[0].plot(df['timestamp'].values, -threshold_line, label='Threshold (-)', color='#00ff80')
    axs[0].fill_between(df['timestamp'].values, threshold_line*-1.5, threshold_line*-0.5, color='#00ff80', alpha=0.25)
    axs[0].plot(time, df['gyro_z'].values, label='Raw', color='#00f000')
    axs[0].plot(time, smooth_gyro_z, label='Smoothed', color='#f00000')
    axs[0].legend()

    axs[1].set_title('Angular Displacement and Turns')
    axs[1].set_ylabel('Angular Displacement (rad)')
    axs[1].set_xlabel('Time (s)')
    axs[1].axhline(color="#000000")
    axs[1].plot(time, np.concatenate(([0], ms1.integrate(df['gyro_z'].values, time))), label='Raw', color='#00b000')
    axs[1].plot(time, np.concatenate(([0], ms1.integrate(smooth_gyro_z, time))), label='Smoothed', color='#b00000')
    axs[1].vlines(np.concatenate((cw_tstamp, ccw_tstamp)), 0, np.concatenate((np.array(cw_turns)[:,1], np.array(ccw_turns)[:,1])) , label='Turns', color='#0000ff')
    axs[1].legend()
    

    '''
    # Plot angular displacement and turns
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title('Milestone 3')
    fig.tight_layout()
    ax.set_title('Angular Offset')
    ax.set_ylabel('Angular Offset (rad)')
    ax.set_xlabel('Time (s)')
    
    ax.plot(time, df['gyro_z'].values, label='Gyro Z (raw)', color='#ff0000')
    ax.plot(time, smooth_gyro_z, label='Gyro Z (smooth)', color='#f0f000')
    #threshold_line = np.full(df['timestamp'].to_numpy().shape, 0.25)
    #ax.plot(df['timestamp'].values, threshold_line, label='Threshold (+)', color='#0080ff')
    #ax.fill_between(df['timestamp'].values, threshold_line-0.125, threshold_line+0.125, color='#0080ff', alpha=0.25)
    #ax.plot(df['timestamp'].values, -threshold_line, label='Threshold (-)', color='#00ff80')
    #ax.fill_between(df['timestamp'].values, -threshold_line-0.125, -threshold_line+0.125, color='#00ff80', alpha=0.25)
    
    ax.plot(time, theta_z, label='Theta_z', color='#0000ff')
    # REMOVE ax.plot(df['timestamp'].values, cw_turn_theta_z, label='CW Turns', color='#ff0000', marker='o', markersize=10)
    # REMOVE ax.plot(df['timestamp'].values, ccw_turn_theta_z, label='CCW Turns', color='#00ff00', marker='o', markersize=10)
    ax.plot(cw_tstamp, cw_turn_theta_z, label='CW Turns', color='#ff0000', linestyle='None', marker='o', markersize=10)
    ax.plot(ccw_tstamp, ccw_turn_theta_z, label='CCW Turns', color='#00ff00', linestyle='None', marker='o', markersize=10)
    ax.legend()
    '''

    plt.show()



if __name__ == '__main__':
    # Get data from csv
    df = pd.read_csv('datasets/TURNING.csv', usecols=['timestamp', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z'])
    do_the_thing(df, 0.12, np.pi/2)