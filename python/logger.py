import datetime

class Logger:

    @staticmethod
    def log(message):
        file = open("log.txt", "a")
        current_time = datetime.datetime.now()
        file.write(str(current_time) + " -> "+ message + "\n")
        file.close()