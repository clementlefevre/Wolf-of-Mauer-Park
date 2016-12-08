class Prediction(object):

    def __init__(self, df, serie_name, date, shift):
        self.df = df
        self.serie_name = serie_name
        self.shift = shift
        self.date = date
