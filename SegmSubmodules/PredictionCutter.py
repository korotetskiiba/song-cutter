import datetime
import subprocess


def cut_file(path_to_file, target_path, prediction_intervals, extension=".wav"):
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
    timedelta_interval = []
    for part in interval:
        date_time_obj = datetime.datetime.strptime(part, '%H:%M:%S')
        timedelta_obj = datetime.timedelta(hours=date_time_obj.hour, minutes=date_time_obj.minute,
                                           seconds=date_time_obj.second)
        timedelta_interval.append(timedelta_obj)
    return timedelta_interval

