import datetime
import subprocess


def slice_file(path_to_file, target_path, prediction_intervals, extension=".wav"):
    """Cut file using ffmpeg into pieces described in the intervals get by prediction. Can be used to cut sound or
    video

    Args:
        path_to_file: path to file to be cut
        target_path: directory with name prefix where file pieces will be saved
        prediction_intervals: the list of intervals where interval is the list of the beginning and the end time
        stored as string in format 'hh:mm:ss'
        extension: target extension (e. g., ".wav" for sound)"""
    i = 1  # counter for video piece
    for interval in prediction_intervals:
        timedelta_interval = __time_to_seconds(interval)  # convert to timedelta interval
        begin = timedelta_interval[0].total_seconds()  # convert to total seconds as ffmpeg gets seconds
        end = timedelta_interval[1].total_seconds()
        duration = end - begin
        piece_name = target_path + "_piece_" + str(i) + extension  # create name of the file to save piece

        # create ffmpeg command
        command = "ffmpeg -i {} -ss {} -t {} -acodec copy {}".format(path_to_file, begin, duration, piece_name)
        subprocess.call(command, shell=True)  # run command
        i += 1


def __time_to_seconds(interval):
    """Translate time stored in string interval to the timedelta interval

    Args:
        interval: the list of the beginning and the end time stored as string in format 'hh:mm:ss'

    Returns:
        the list of the beginning and the end time stored as timedelta objects"""
    timedelta_interval = []  # list for the answer
    for part in interval:
        date_time_obj = datetime.datetime.strptime(part, '%H:%M:%S')  # parse time to datetime
        timedelta_obj = datetime.timedelta(hours=date_time_obj.hour, minutes=date_time_obj.minute,
                                           seconds=date_time_obj.second)  # convert to timedelta
        timedelta_interval.append(timedelta_obj)
    return timedelta_interval

