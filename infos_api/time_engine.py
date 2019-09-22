from datetime import date
from datetime import time
from datetime import datetime


class TimeEngine():

    def __init__(self):

        self.today = datetime.now()

    def date(self):

        days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        week_day = date.weekday(self.today)
        day = self.today.day
        month = self.today.month
        year = self.today.year

        date_today = "Today is : " + str(days[week_day]) + ", " + str(day) + ", " + str(months[month]) + ", " + str(year) + "."

        return date_today

    def time(self):

        hour = self.today.strftime("%I")
        minute = self.today.strftime("%M")
        am_pm = self.today.strftime("%p")

        current_time = "Time now is : " + hour + ", " + minute + ", " + am_pm + "."

        return current_time
