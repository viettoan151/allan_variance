#!/usr/bin/env python
import rospy
import sys
import allantools
import rosbag
import numpy as np
import csv
import rospkg
import os
import matplotlib.pyplot as plt  # only for plotting, not required for calculations
import math
import yaml


# https://www.mathworks.com/help/nav/ug/inertial-sensor-noise-analysis-using-allan-variance.html
def random_walk(tau, adev):
    # Find the index where the slope of the log-scaled Allan deviation is equal
    # to the slope specified.
    slope = 0.5  # slope of random walk
    """""""""""""""""""""""""""""""""
    " Find point where slope = -0.5 "
    """""""""""""""""""""""""""""""""
    logtau = np.array(list(map(lambda i: math.log10(i), tau)), dtype=np.float64)
    logadev = np.array(list(map(lambda i: math.log10(i), adev)), dtype=np.float64)
    dlogadev = np.diff(logadev) / np.diff(logtau)
    i = np.argmin(dlogadev)   

    #  Find the y-intercept of the line.
    log_b = logadev[i] - slope*logtau[i]
    # Determine the rate random walk coefficient from the line.
    log_k = slope*math.log10(3) + log_b

    K = pow(10, log_k)

    return K


def white_noise(tau, sigma):
    m = -0.5  # slope of random walk
    """""""""""""""""""""""""""""""""
    " Find point where slope = -0.5 "
    """""""""""""""""""""""""""""""""
    randomWalk = None
    i = 1
    idx = 1
    mindiff = 999
    logTau = -999
    while (logTau < 0):
        logTau = math.log(tau[i], 10)
        logSigma = math.log(sigma[i], 10)
        prevLogTau = math.log(tau[i-1], 10)
        prevLogSigma = math.log(sigma[i-1], 10)
        slope = (logSigma-prevLogSigma)/(logTau-prevLogTau)
        diff = abs(slope-m)
        if (diff < mindiff):
            mindiff = diff
            idx = i
        i = i + 1

    """"""""""""""""""""""""""""""
    " Project line to tau = 10^0 "
    """"""""""""""""""""""""""""""
    x1 = math.log(tau[idx], 10)
    y1 = math.log(sigma[idx], 10)
    x2 = 0
    y2 = m*(x2-x1)+y1

    return (pow(10, x1), pow(10, y1), pow(10, x2), pow(10, y2))


# axis info - list of tuples (taus_used, adev, rate_rand_walk, ang_rand_walk)
# res_type - string, either 'acceleration' or 'gyroscope'
# resultsPath - path of directory where files will be saved
def plot_n_save(axis_info, fig_name, results_path):
    # Plot Acceleration Result
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_xscale('log')

    avg_w = 0
    avg_b = 0

    for i in range(3):
        taus_used = axis_info[i][0]
        adev = axis_info[i][1]

        K = axis_info[i][2]
        white_noise = axis_info[i][3]
        
        avg_w = avg_w + white_noise[3]
        avg_b = avg_b + K
        
        lineK = np.array(list(map(lambda t: math.sqrt(t/3) * K, taus_used)), dtype=np.float64)

        # plot Allan Deviation
        plt.plot(taus_used, adev)
        # plot bias function
        plt.plot(taus_used, lineK, color='b', linewidth=2, alpha=0.5)  
        plt.plot(3, K, 'o', color='r')
        # plot white noise
        plt.plot([white_noise[0], white_noise[2]],
                 [white_noise[1], white_noise[3]], 
                 'r', linewidth=2, alpha=0.5)
    
    avg_w = avg_w / 3
    avg_b = avg_b / 3
    
    with open(os.path.join(results_path, fig_name+'.yaml'), 'w') as f:
        coeffs = {'White Noise': float(avg_w), 'Random Walk': float(avg_b)}
        yaml.dump(coeffs, f)
    
    title = 'Allan deviation ('+fig_name+')'
    plt.grid(True, which="both")
    plt.title(title)
    plt.xlabel('Tau (s)')
    plt.ylabel('ADEV')

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)

    plt.show(block=False)
    plt.savefig(os.path.join(results_path, fig_name))


def main(args):

    rospy.init_node('allan_variance_node')

    t0 = rospy.get_time()

    """"""""""""""
    " Parameters "
    """"""""""""""
    bagfile = rospy.get_param('~bagfile_path', '~/data/static.bag')
    topic = rospy.get_param('~imu_topic_name', '/imu')
    sample_rate = rospy.get_param('~sample_rate', 100)
    is_delta_type = rospy.get_param('~delta_measurement', False)
    num_tau = rospy.get_param('~number_of_lags', 1000)
    results_path = rospy.get_param('~results_directory_path', None)

    """"""""""""""""""""""""""
    " Results Directory Path "
    """"""""""""""""""""""""""
    if results_path is None:
        paths = rospkg.get_ros_paths()
        path = paths[1]  # path to workspace's devel
        idx = path.find("ws/")
        workspace_path = path[0:(idx+3)]
        results_path = workspace_path + 'av_results/'

        if not os.path.isdir(results_path):
            os.mkdir(results_path)

    print "\nResults will be saved in the following directory: \n\n\t %s\n" % results_path

    """"""""""""""""""
    " Form Tau Array "
    """"""""""""""""""
    taus = np.logspace(-2.0, 5.0, num=num_tau)

    """""""""""""""""
    " Parse Bagfile "
    """""""""""""""""
    bag = rosbag.Bag(bagfile)

    N = bag.get_message_count(topic)  # number of measurement samples

    data = np.zeros((6, N))  # preallocate vector of measurements

    if is_delta_type:
        scale = sampleRate
    else:
        scale = 1.0

    cnt = 0
    for topic, msg, t in bag.read_messages(topics=[topic]):
        data[0, cnt] = msg.linear_acceleration.x * scale
        data[1, cnt] = msg.linear_acceleration.y * scale
        data[2, cnt] = msg.linear_acceleration.z * scale
        data[3, cnt] = msg.angular_velocity.x * scale
        data[4, cnt] = msg.angular_velocity.y * scale
        data[5, cnt] = msg.angular_velocity.z * scale
        cnt = cnt + 1

    bag.close()

    print "[%0.2f seconds] Bagfile parsed\n" % (rospy.get_time()-t0)

    accel_axis_info = []
    gyro_axis_info = []

    # Acceleration
    for axis in range(3):
        (taus_used, adev, adev_err, adev_n) = allantools.adev(
            data[axis], data_type='freq', rate=float(sample_rate), taus=taus)

        rand_walk = random_walk(taus_used, adev)
        w_noise = white_noise(taus_used, adev)

        accel_axis_info.append(
            (taus_used, adev, rand_walk, w_noise))

        print "[%0.2f seconds] Finished calculating allan variance for acceleration axis %d" % (rospy.get_time()-t0, axis)

    # Gyro
    for axis in range(3, 6):
        (taus_used, adev, adev_err, adev_n) = allantools.adev(
            data[axis], data_type='freq', rate=float(sample_rate), taus=np.array(taus))

        rand_walk = random_walk(taus_used, adev)
        w_noise = white_noise(taus_used, adev)

        gyro_axis_info.append(
            (taus_used, adev, rand_walk, w_noise))

        print "[%0.2f seconds] Finished calculating allan variance for gyro axis %d" % (rospy.get_time()-t0, axis-3)

    plot_n_save(accel_axis_info, 'acceleration', results_path)
    plot_n_save(gyro_axis_info, 'gyroscope', results_path)

    inp = raw_input("Press Enter key to close figures and end program\n")


if __name__ == '__main__':
    main(sys.argv)
