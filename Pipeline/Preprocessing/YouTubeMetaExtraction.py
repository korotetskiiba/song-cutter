import pytube
import argparse
import csv
import re

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

    def get_time_codes(self):
        pattern = '[0-9]+:[0-9]+[^\n\r]+'
        codes = re.findall(pattern, self.__video.description)
        return codes

    def get_caption(self, language_code):
        cap = self.__video.captions.get_by_language_code(language_code)
        assert cap is not None, "Invalid language code"
        return cap.generate_srt_captions()

    def get_captions_type(self):
        return self.__video.captions.all()

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
    data['codes'] = yt.get_time_codes()
    data['captions'] = yt.get_caption('ru').replace('\n', ' ').replace(';', ' ')
    data['description'] = yt.get_description().replace('\n', ' ').replace(';', ' ')
    data['html'] = yt.get_html().replace('\n', ' ').replace(';', ' ')

    with open(args.path_to_save_csv, "a", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow([data[x] for x in data])
