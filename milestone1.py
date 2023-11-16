import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def integrate(data, time):
    """Expecting data to be an array of shape (n, 2) and time to be an array of shape (n,)"""
    time = np.repeat(time, 2).reshape(-1,2)
    return (data[:-1,:] + ((data[1:,:] - data[:-1,:]) / 2)).cumsum(axis=0) * (time[1:,:] - time[:-1,:])

# Get data and calculate desired values
df = pd.read_csv('datasets/ACCELERATION.csv')
time = df['timestamp'].values
acc = df[['acceleration', 'noisyacceleration']].values
vel = np.vstack(([0,0], integrate(acc, time)))
dist = np.vstack(([0,0], integrate(vel, time)))

# Print result summaries
print("Final distances:\n\tReal: {}\tNoisy: {}\tError: {}".format(dist[-1,0], dist[-1,1], abs(dist[-1,1] - dist[-1,0])))

# Plot results
fig, axs = plt.subplots(3, sharex=False, sharey=False)
fig.canvas.manager.set_window_title('Milestone 1')
fig.tight_layout()

axs[0].set_title('Acceleration')
axs[0].set_ylabel('Acceleration (m / s²)')
axs[0].set_xlabel('Time (s)')
axs[0].plot(time, acc[:,0], label='Real', color='#00f000')
axs[0].plot(time, acc[:,1], label='Noisy', color='#f00000')
axs[0].legend()

axs[1].set_title('Speed')
axs[1].set_ylabel('Speed (m / s)')
axs[1].set_xlabel('Time (s)')
axs[1].plot(time, vel[:,0], label='Real', color='#00b000')
axs[1].plot(time, vel[:,1], label='Noisy', color='#b00000')
axs[1].legend()

axs[2].set_title('Distance')
axs[2].set_ylabel('Distance (m)')
axs[2].set_xlabel('Time (s)')
axs[2].plot(time, dist[:,0], label='Real', color='#007000')
axs[2].plot(time, dist[:,1], label='Noisy', color='#700000')
axs[2].legend()

plt.show()
