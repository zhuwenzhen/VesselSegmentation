import PIL
import exceptions
class Adv_Image(PIL.Image):
    def __init__(self, *args, **argv):
        super().__init__(*args, **argv)
    def interpolate(x, y):
        raise exceptions.NotImplementedError()
