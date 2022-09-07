class BaseInferer:
    """ Base inferer class

    """
    def infer(self, *args, **kwargs):
        """ Perform an inference on test data.

        """
        raise NotImplementedError

    def fusion(self, submissions_dir, preds):
        """ Ensamble predictions.

        """
        raise NotImplementedError        

        



