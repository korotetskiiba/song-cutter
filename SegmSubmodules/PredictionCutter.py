import datetime
import subprocess


def cut_file(path_to_file, target_path, prediction_intervals, extension=".wav"):
    """Cut file using ffmpeg into pieces described in the intervals get by prediction. Can be used to cut sound or
    video

    Args:
        path_to_file: path to file to be cut

        target_path: directory with name prefix where file pieces will be saved

        prediction_intervals: the list of intervals where interval is the list of the beginning and the end time
        stored as string in format 'hh:mm:ss'

        extension: target extension (e. g., ".wav" for sound)"""
    i = 1
    for interval in prediction_intervals:
        timedelta_interval = __time_to_seconds(interval)
        begin = timedelta_interval[0].total_seconds()
        end = timedelta_interval[1].total_seconds()
        end = end - begin  # translate end from absolute bias to bias from begin
        piece_name = target_path + "_piece_" + str(i) + extension

        command = "ffmpeg -i {} -ss {} -t {} -acodec copy {}".format(path_to_file, begin, end, piece_name)
        subprocess.call(command, shell=True)
        i += 1


def __time_to_seconds(interval):
    """Translate time stored in string interval to the timedelta interval

    Args:
        interval: the list of the beginning and the end time stored as string in format 'hh:mm:ss'

    Returns:
        the list of the beginning and the end time stored as timedelta objects"""
    timedelta_interval = []
    for part in interval:
        date_time_obj = datetime.datetime.strptime(part, '%H:%M:%S')
        timedelta_obj = datetime.timedelta(hours=date_time_obj.hour, minutes=date_time_obj.minute,
                                           seconds=date_time_obj.second)
        timedelta_interval.append(timedelta_obj)
    return timedelta_interval

