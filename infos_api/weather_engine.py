import bs4 as bs
import urllib.request
import re


class WeatherEngine:

    def __init__(self, country, city):

        self.source = urllib.request.urlopen('https://www.weather-atlas.com/en/'+ country + "/" + city + '#daily').read()
        self.base_url = "https://www.weather-atlas.com/en/"
        self.country = country
        self.city = city

    def get_today_weather(self):
        soup = bs.BeautifulSoup(self.source, "lxml")

        # Get place and date
        h22 = soup.find_all("h2", {"id": "daily"})[0]
        place_date = h22.get_text()

        # Get informations
        div = soup.find_all("div", {"class": "panel"})[1]
        spans = div.findChildren("span", recursive=True)
        weather_description = spans[1].get_text()
        temperature = spans[2].get_text()

        if spans[3].get_text() == '/':
            wind = "not available !"
        else:
            wind = spans[3].get_text()

        # Create weather complete description text
        weather = place_date + ". Weather description : " + weather_description + ". Temperature : " + temperature + ". Wind : " + wind

        return weather



