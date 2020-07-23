import pytube


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