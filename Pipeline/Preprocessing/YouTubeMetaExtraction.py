import pytube
import argparse
import csv
import re

KNOWN_CHANNELS = ['Вечерний Ургант', 'НТВ']


class YouTubeMetaExtraction:

    def __init__(self, url):
        self.__video = pytube.YouTube(url)
        self.__url = url

    def get_description(self):
        return self.__video.description

    def get_title(self):
        return self.__video.title

    def get_channel_name(self):
        return self.__video.author

    def get_html(self):
        return pytube.request.get(self.__url)

    def get_time_codes_list_from_description(self):
        pattern = '[0-9]+:[0-9]+[^\n\r]+'
        codes = re.findall(pattern, self.__video.description)
        return codes

    def get_songs_list_from_description(self):
        songs = self.get_time_codes_list_from_description()
        for i in range(len(songs)):
            songs[i] = re.sub('[0-9:]+[\W]+', '', songs[i])
        return songs

    def get_caption(self, language_code):
        cap = self.__video.captions.get_by_language_code(language_code)
        assert cap is not None, "Invalid language code"
        return cap.generate_srt_captions()

    def get_captions_type_list(self):
        return self.__video.captions.all()

    def get_music_parts_from_caption(self, language_code):
        lang = {'en': 'music', 'ru': 'музыка'}
        if language_code not in lang:
            return None

        cap = self.__video.captions.get_by_language_code(language_code)
        assert cap is not None, "Invalid language code"

        caption = cap.generate_srt_captions()
        mus = re.findall('[0-9]+[,:0-9]+ --> [0-9]+[,:0-9]+\n\['+lang[language_code]+']', caption)
        for i in range(len(mus)):
            mus[i] = re.search('[0-9]+[,:0-9]+ --> [0-9]+[,:0-9]+', mus[i]).group().replace('->', '')
        return mus

    def get_proper_name_list(self):
        names = re.findall('[\"«][^»\"]+[»\"]', self.__video.description)
        for i in range(len(names)):
            names[i] = re.search('[^«»\"]+', names[i]).group()
        return names

    def get_specialized_information(self):
        channel = self.get_channel_name()

        if channel == KNOWN_CHANNELS[0]:
            return self.__get_urgant_info()
        elif channel == KNOWN_CHANNELS[1]:
            return self.__get_ntv_info()
        return {}

    def __get_urgant_info(self):
        info = {}
        title = re.search('\w[^.]+[-–][^.\n]+', self.get_title())
        if title is None:
            return {}
        title = title.group()
        info['artist'] = re.sub(' [-–][^.\n]+','', title)
        info['song_title'] = re.sub('\w[^.]+[-–] ','', title)
        return info

    def __get_ntv_info(self):
        artist = re.search(': [^\n]+', self.get_title())
        if artist is None:
            return {}
        artist = artist.group()
        artist = re.sub(': ', '', artist)
        return {'artist': artist}


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
        data['mus_caption'] = yt.get_music_parts_from_caption(lang_code)
    else:
        data['captions'] = None
        data['mus_caption'] = None

    data['description'] = yt.get_description().replace('\n', ' ').replace(';', ' ')

    data['html'] = yt.get_html().replace('\n', ' ').replace(';', ' ')

    with open(args.path_to_save_csv, "a", newline='', encoding='utf-16') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for x in data:
            writer.writerow([x, data[x]])
