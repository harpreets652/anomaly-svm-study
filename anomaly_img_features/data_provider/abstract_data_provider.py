class AbstractDataProvider(object):

    def get_training_data(self):
        raise NotImplementedError("Cannot use abstract data provider")
        pass

    def get_image_descriptor(self, image_path):
        raise NotImplementedError("Cannot use abstract data provider")
        pass
