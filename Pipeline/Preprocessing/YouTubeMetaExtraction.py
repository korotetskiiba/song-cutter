import pytube
import argparse
import csv
import re
from datetime import datetime

KNOWN_CHANNELS = ['Вечерний Ургант', 'НТВ']


class YouTubeMetaExtraction:

    def __init__(self, url):
        self.__video = pytube.YouTube(url)
        self.__url = url

    def get_description(self):
        """
        Getter for the description of the video
        :return: description (str)
        """
        return self.__video.description

    def get_title(self):
        """
        Getter for the title of the video
        :return: title (str)
        """
        return self.__video.title

    def get_channel_name(self):
        """
        Getter for the author of the video
        :return: author (str)
        """
        return self.__video.author

    def get_html(self):
        """
        Getter for the html code of a page with a video
        :return: html code (str)
        """
        return pytube.request.get(self.__url)

    def get_time_codes_list_from_description(self):
        """
        Getter for a list with time codes and names from description
        :return: list with time codes and names (list)
        """
        pattern = '[0-9]+:[0-9]+[^\n\r]+'
        codes = re.findall(pattern, self.__video.description)
        return codes

    def get_songs_list_from_description(self):
        """
        Getter for a list with the names of parts from the description
        :return: list with names (list)
        """
        songs = self.get_time_codes_list_from_description()
        for i in range(len(songs)):
            songs[i] = re.sub('[0-9:]+[\W]+', '', songs[i])
        return songs

    def get_caption(self, language_code):
        """
        Getter for the caption to the video
        :param language_code: language code
        :return: caption (str)
        """
        assert isinstance(language_code, str), "Language code must be a str"

        caption_types = self.get_captions_type_list()

        assert language_code in [caption_types[i].code for i in range(len(caption_types))], "Invalid language code"

        cap = self.__video.captions.get_by_language_code(language_code)
        assert cap is not None, "Invalid language code"

        return cap.generate_srt_captions()

    def get_captions_type_list(self):
        """
        Getter for the types of caption to the video
        :return: types (list)
        """
        return self.__video.captions.all()

    def get_music_parts_from_caption(self, language_code):
        """
        Getter for the parts from caption with a music tag
        :param language_code: language code
        :return: times of parts (list)
        """
        lang = {'en': 'music', 'ru': 'музыка'}
        if language_code not in lang:
            return None

        caption = self.get_caption(language_code)

        mus = re.findall('[0-9]+[,:0-9]+ --> [0-9]+[,:0-9]+\n\['+lang[language_code]+']', caption)
        for i in range(len(mus)):
            mus[i] = re.search('[0-9]+[,:0-9]+ --> [0-9]+[,:0-9]+', mus[i]).group().replace('->', '')
        return mus

    def get_proper_name_list(self):
        """
        Getter for the proper name from description
        :return: proper names (list)
        """
        names = re.findall('[\"«][^»\"]+[»\"]', self.__video.description)
        for i in range(len(names)):
            names[i] = re.search('[^«»\"]+', names[i]).group()
        return names

    def get_specialized_information(self):
        """
        Getter for the information from known channels
        :return: dict with information (dict)
        """
        channel = self.get_channel_name()

        if channel == KNOWN_CHANNELS[0]:
            return self.__get_urgant_info()
        elif channel == KNOWN_CHANNELS[1]:
            return self.__get_ntv_info()
        return {}

    def __get_urgant_info(self):
        """
        Getter for the information from 'Вечерний Ургант'
        :return: dict with information (dict)
        """
        info = {}
        title = re.search('\w[^.]+[-–][^.\n]+', self.get_title())
        if title is None:
            return {}
        title = title.group()
        info['artist'] = re.sub(' [-–][^.\n]+','', title)
        info['song_title'] = re.sub('\w[^.]+[-–] ','', title)
        return info

    def __get_ntv_info(self):
        """
        Getter for the information from 'НТВ'(Квартирник НТВ у Маргулиса)
        :return: dict with information (dict)
        """
        artist = re.search(': [^\n]+', self.get_title())
        if artist is None:
            return {}
        artist = artist.group()
        artist = re.sub(': ', '', artist)
        return {'artist': artist}

    def get_text(self, start_time, end_time, language_code):
        assert isinstance(start_time, int) and isinstance(end_time, int), 'time is expected in whole seconds'
        assert start_time < end_time, 'end is less than start'
        assert end_time <= self.get_length(), 'out of bounds'

        caption = self.get_caption(language_code)

        captions_parts = re.split('\n\n', caption)

        parts = []
        for i in range(len(captions_parts)):
            time = re.search('[0-9]+:[0-9]+:[0-9]+', captions_parts[i]).group()
            time = datetime.strptime(time, '%H:%M:%S')
            time_sec = time.second + time.minute * 60 + time.hour * 3600
            text = re.split('\n', captions_parts[i])[2]
            if re.search('\[[\w]+\]', text) is None:
                parts.append({'time': time_sec, 'text': text})

        parts = [parts[i] for i in range(len(parts)) if (parts[i]['time'] >= start_time and parts[i]['time'] <= end_time)]

        text = ''
        for i in range(len(parts)):
            text += parts[i]['text']+' '

        return text

    def get_length(self):
        return self.__video.length





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocessing')

    parser.add_argument('-u', type=str, action="store", dest="url", help='url')
    parser.add_argument('-o', type=str, action="store", dest="path_to_save_csv",
                        help='path to save the output file')
    args = parser.parse_args()

    yt = YouTubeMetaExtraction(args.url)

    data = {}

    data['channel'] = yt.get_channel_name()
    data['title'] = yt.get_title()
    data['proper_names'] = yt.get_proper_name_list()
    data['songs'] = yt.get_songs_list_from_description()
    data['description_codes'] = yt.get_time_codes_list_from_description()

    data.update(yt.get_specialized_information())


    captions = yt.get_captions_type_list()
    if len(captions) > 0:
        lang_code = captions[0].code
        data['caption'] = yt.get_caption(lang_code).replace('\n', ' ').replace(';', ' ')
        data['text'] = yt.get_text(0, yt.get_length(), lang_code)
        data['mus_caption'] = yt.get_music_parts_from_caption(lang_code)
    else:
        data['captions'] = None
        data['text'] = None
        data['mus_caption'] = None

    data['description'] = yt.get_description().replace('\n', ' ').replace(';', ' ')

    data['html'] = yt.get_html().replace('\n', ' ').replace(';', ' ')

    with open(args.path_to_save_csv, "a", newline='', encoding='utf-16') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for x in data:
            writer.writerow([x, data[x]])
