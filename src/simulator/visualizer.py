import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# dependencies
# python -m pip install PyQt5
# sudo apt-get install libqt5gui5

def plot3d(file, waypoints_ned, stateArray, env_step_dt):
    fig = plt.figure(figsize=(12, 10))

    # 3D Plot
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    wp_x = waypoints_ned[:, 0]
    wp_y = -waypoints_ned[:, 1]
    wp_z = -waypoints_ned[:, 2]
    for i in range(0, len(waypoints_ned)):
        wpText = ax1.text(wp_x[i], wp_y[i], wp_z[i], 'wp-'+str(i))
    wp, = ax1.plot(wp_x, wp_y, wp_z, 'bo')
    point, = ax1.plot([0], [0], [0], 'ro')
    pointText = ax1.text(0, 0, 0, "Quad")
    
    # Set 3D plot ranges
    ax1.set_xlim(-7, 7)
    ax1.set_ylim(-7, 7)
    ax1.set_zlim(0, 7)
    
    # x vs z Plot
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')

    # Set x vs z plot ranges
    ax2.set_xlim(-7, 7)
    ax2.set_ylim(0, 7)

    # y vs z Plot
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_xlabel('Y')
    ax3.set_ylabel('Z')

    # Set y vs z plot ranges
    ax3.set_xlim(-7, 7)
    ax3.set_ylim(0, 7)

    # x, y Plot
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')

    # Set x, y plot ranges
    ax4.set_xlim(-7, 7)
    ax4.set_ylim(-7, 7)

    def update(frame):
        x = stateArray[frame, 0]
        y = -stateArray[frame, 1]
        z = -stateArray[frame, 2]

        # Update 3D plot
        point.set_data(x, y)
        point.set_3d_properties(z)
        pointText.set_position_3d((x, y, z))

        # Update x vs z plot
        ax2.scatter(x, z, color='r')

        # Update y vs z plot
        ax3.scatter(y, z, color='g')

        # Update x, y plot
        ax4.scatter(x, y, color='b')

        return (point, pointText)

    ani = animation.FuncAnimation(fig, update, frames=len(stateArray), interval=env_step_dt)

    writervideo = animation.FFMpegWriter(fps=60)
    ani.save(file, writer=writervideo)


def plot3dFromFile(outfile, infile):
    statesfile = np.load(infile + '_runtime.npz')
    configfile = np.load(infile + '_config.npz')

    waypoints_ned = configfile['wps']
    stateArray = statesfile['state']
    env_step_dt = configfile['env_step_dt']

    plot3d(outfile, waypoints_ned, stateArray, env_step_dt)