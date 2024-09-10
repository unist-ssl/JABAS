class ElasticTrainReStartTimer(object):
    def __init__(self, start_time):
        self.start_time = start_time
        self.elapsed_time = 0

    def update(self, measured_time):
        self.elapsed_time = measured_time - self.start_time