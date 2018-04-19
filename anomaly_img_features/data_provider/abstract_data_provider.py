class AbstractDataProvider(object):

    def get_training_data(self):
        raise NotImplementedError("Cannot use abstract data provider")
        pass
