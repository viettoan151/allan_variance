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


def getRRWSegment(tau, sigma, bias_idx):

    m = 0.5  # slope of random walk
    """""""""""""""""""""""""""""""""
    " Find point where slope = -0.5 "
    """""""""""""""""""""""""""""""""
    # randomWalk = None
    i = len(tau)-2  # start idx
    stopTauPoint = math.log(tau[bias_idx])
    idx = 1  # idx of intersection point btw tangent line and curve
    mindiff = -1

    while (math.log(tau[i]) > stopTauPoint):
        leftLogTau = math.log(tau[i], 10)
        leftLogSigma = math.log(sigma[i], 10)
        rightLogTau = math.log(tau[i+1], 10)
        rightLogSigma = math.log(sigma[i+1], 10)
        slope = (rightLogSigma-leftLogSigma)/(rightLogTau-leftLogTau)

        diff = abs(slope-m)
        if (diff < mindiff or mindiff == -1):
            mindiff = diff
            idx = i
        i = i - 1

    """"""""""""""""""""""""""""""
    " Project line to tau = 10^0 "
    """"""""""""""""""""""""""""""
    x2 = math.log(tau[idx], 10)
    y2 = math.log(sigma[idx], 10)

    x1 = math.log(3, 10)
    y1 = y2-m*(x2-x1)

    return (pow(10, x1), pow(10, y1), pow(10, x2), pow(10, y2))


def getARWSegment(tau, sigma):

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


def get_bias_inst_idx(tau, sigma):
    i = 1
    while (i < tau.size):
        if (tau[i] > 1) and ((sigma[i]-sigma[i-1]) > 0):  # only check for tau > 10^0
            break
        i = i + 1
    return i

# axis info - list of tuples (taus_used, adev, rate_rand_walk, ang_rand_walk)
# res_type - string, either 'acceleration' or 'gyroscope'
# resultsPath - path of directory where files will be saved
def plot_n_save(axis_info, res_type, resultsPath):
    # Plot Acceleration Result
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_xscale('log')

    av_rrt = 0
    av_art = 0

    fname = res_type + '_results'
    f = open(resultsPath + fname + '.csv', 'wt')
    try:
        writer = csv.writer(f)

        for x in range(0, 3):
            taus_used = axis_info[x][0]
            adev = axis_info[x][1]
            rate_rand_walk = axis_info[x][2]
            ang_rand_walk = axis_info[x][3]

            # deviation value in point tau = 3, NoiseBias
            av_rrt = av_rrt + rate_rand_walk[1]
            # deviation value in point tau = 1, NoiseDensity
            av_art = av_art + ang_rand_walk[3]
            if x == 0:
                writer.writerow(('Rate Random Walk X', rate_rand_walk[1]))
                writer.writerow(('Angular Random Walk X', ang_rand_walk[1]))
                plt.plot([ang_rand_walk[0], ang_rand_walk[2]],
                         [ang_rand_walk[1], ang_rand_walk[3]],
                         label='Angular random walk X',
                         linestyle='dotted',
                         linewidth=5)
                plt.plot([rate_rand_walk[0], rate_rand_walk[2]],
                         [rate_rand_walk[1], rate_rand_walk[3]],
                         label='Rate random walk X',
                         linestyle='dashed',
                         linewidth=5)
                plt.plot(taus_used, adev)
            elif x == 1:
                writer.writerow(('Rate Random Walk Y', rate_rand_walk[1]))
                writer.writerow(('Angular Random Walk Y', ang_rand_walk[1]))
                plt.plot([ang_rand_walk[0], ang_rand_walk[2]],
                         [ang_rand_walk[1], ang_rand_walk[3]],
                         label='Angular random walk Y',
                         linestyle='dotted',
                         linewidth=5)
                plt.plot([rate_rand_walk[0], rate_rand_walk[2]],
                         [rate_rand_walk[1], rate_rand_walk[3]],
                         label='Rate random walk Y',
                         linestyle='dashed',
                         linewidth=5)
                plt.plot(taus_used, adev)
            elif x == 2:
                writer.writerow(('Rate Random Walk Z', rate_rand_walk[1]))
                writer.writerow(('Angular Random Walk Z', ang_rand_walk[1]))
                plt.plot([ang_rand_walk[0], ang_rand_walk[2]],
                         [ang_rand_walk[1], ang_rand_walk[3]],
                         label='Angular random walk Z',
                         linestyle='dotted',
                         linewidth=5)
                plt.plot([rate_rand_walk[0], rate_rand_walk[2]],
                         [rate_rand_walk[1], rate_rand_walk[3]],
                         label='Rate random walk Z',
                         linestyle='dashed',
                         linewidth=5)
                plt.plot(taus_used, adev)

        av_rrt = av_rrt / 3
        av_art = av_art / 3
        writer.writerow(('Rate Random Walk Avg', av_rrt))
        writer.writerow(('Angular Random Walk Avg', av_art))
    finally:
        f.close()

    title = 'Allan Deviation for '+res_type
    fig_name = 'allan_'+res_type

    plt.grid(True, which="both")
    plt.title(title)
    plt.xlabel('Tau (s)')
    plt.ylabel('ADEV')
    plt.legend()

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    plt.show(block=False)
    plt.savefig(resultsPath + fig_name)


def main(args):

    rospy.init_node('allan_variance_node')

    t0 = rospy.get_time()

    """"""""""""""
    " Parameters "
    """"""""""""""
    bagfile = rospy.get_param('~bagfile_path', '~/data/static.bag')
    topic = rospy.get_param('~imu_topic_name', '/imu')
    sampleRate = rospy.get_param('~sample_rate', 100)
    isDeltaType = rospy.get_param('~delta_measurement', False)
    numTau = rospy.get_param('~number_of_lags', 1000)
    resultsPath = rospy.get_param('~results_directory_path', None)

    """"""""""""""""""""""""""
    " Results Directory Path "
    """"""""""""""""""""""""""
    if resultsPath is None:
        paths = rospkg.get_ros_paths()
        path = paths[1]  # path to workspace's devel
        idx = path.find("ws/")
        workspacePath = path[0:(idx+3)]
        resultsPath = workspacePath + 'av_results/'

        if not os.path.isdir(resultsPath):
            os.mkdir(resultsPath)

    print "\nResults will be saved in the following directory: \n\n\t %s\n" % resultsPath

    """"""""""""""""""
    " Form Tau Array "
    """"""""""""""""""
    taus = [None]*numTau

    cnt = 0
    # lags will span from 10^-2 to 10^5, log spaced
    for i in np.linspace(-2.0, 5.0, num=numTau):
        taus[cnt] = pow(10, i)
        cnt = cnt + 1

    """""""""""""""""
    " Parse Bagfile "
    """""""""""""""""
    bag = rosbag.Bag(bagfile)

    N = bag.get_message_count(topic)  # number of measurement samples

    data = np.zeros((6, N))  # preallocate vector of measurements

    if isDeltaType:
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
        (taus_used, adev, adev_err, adev_n) = allantools.oadev(
            data[axis], data_type='freq', rate=float(sampleRate), taus=np.array(taus))

        bias_inst_idx = get_bias_inst_idx(
            taus_used, adev)   # bias instability point idx
        rate_rand_walk = getRRWSegment(taus_used, adev, bias_inst_idx)
        ang_rand_walk = getARWSegment(taus_used, adev)

        accel_axis_info.append(
            (taus_used, adev, rate_rand_walk, ang_rand_walk))

        print "[%0.2f seconds] Finished calculating allan variance for acceleration axis %d" % (rospy.get_time()-t0, axis)

    # Gyro
    for axis in range(3, 6):
        (taus_used, adev, adev_err, adev_n) = allantools.oadev(
            data[axis], data_type='freq', rate=float(sampleRate), taus=np.array(taus))

        bias_inst_idx = get_bias_inst_idx(
            taus_used, adev)   # bias instability point idx
        rate_rand_walk = getRRWSegment(taus_used, adev, bias_inst_idx)
        ang_rand_walk = getARWSegment(taus_used, adev)

        gyro_axis_info.append(
            (taus_used, adev, rate_rand_walk, ang_rand_walk))

        print "[%0.2f seconds] Finished calculating allan variance for gyro axis %d" % (rospy.get_time()-t0, axis)

    plot_n_save(accel_axis_info, 'acceleration', resultsPath)
    plot_n_save(gyro_axis_info, 'gyroscope', resultsPath)

    inp = raw_input("Press Enter key to close figures and end program\n")


if __name__ == '__main__':
    main(sys.argv)
