import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# dependencies
# python -m pip install PyQt5
# sudo apt-get install libqt5gui5

def plot3d(file, waypoints_ned, stateArray, env_step_dt):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot Waypoints:
    # Plot in enu frame
    wp_x = waypoints_ned[:, 0]
    wp_y = -waypoints_ned[:, 1]
    wp_z = -waypoints_ned[:, 2]
    for i in range(0, len(waypoints_ned)):
        wpText = ax.text(wp_x[i], wp_y[i], wp_z[i], 'wp-'+str(i))
    wp, = ax.plot(wp_x, wp_y, wp_z, 'bo')

    # Plot Quad
    point, = ax.plot([0], [0], [0], 'ro')
    pointText = ax.text(0, 0, 0, "Quad")

    def update(frame):
        # print(frame)
        x = stateArray[frame, 0]
        y = -stateArray[frame, 1]
        z = -stateArray[frame, 2]
        point.set_data(x, y)
        point.set_3d_properties(z)
        pointText.set_position_3d((x, y, z))

        return (point, pointText)
    
    ani = animation.FuncAnimation(fig, update, frames = len(stateArray), interval=env_step_dt)
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.set_zlim([-1, 10])
    writervideo = animation.FFMpegWriter(fps=60) 

    ani.save(file, writer=writervideo)

def plot3dFromFile(outfile, infile):
    statesfile = np.load(infile + '_runtime.npz')
    configfile = np.load(infile + '_config.npz')

    waypoints_ned = configfile['wps']
    stateArray = statesfile['state']
    env_step_dt = configfile['env_step_dt']

    plot3d(outfile, waypoints_ned, stateArray, env_step_dt)