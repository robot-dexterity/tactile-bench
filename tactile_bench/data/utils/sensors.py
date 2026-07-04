import cv2

from tactile_bench.data.utils.transform_image import transform_image


class BaseSensor:
    def __init__(self, sensor_params={}):
        self.sensor_params = sensor_params

    def read(self):
        raise NotImplementedError

    def process(self, outfile=None):
        img = transform_image(self.read(), **self.sensor_params)
        if outfile:
            cv2.imwrite(outfile, img)
        return img


class SimSensor(BaseSensor):
    def __init__(self, sensor_params={}, embodiment={}):
        super().__init__(sensor_params)
        self.embodiment = embodiment

    def read(self):
        return self.embodiment.get_tactile_observation()


class RealSensor(BaseSensor):
    def __init__(self, sensor_params={}):
        super().__init__(sensor_params)
        source = sensor_params.get('source', 0)
        exposure = sensor_params.get('exposure', -7)

        self.cam = cv2.VideoCapture(source)
        self.cam.set(cv2.CAP_PROP_EXPOSURE, exposure)
        for _ in range(5):
            self.cam.read()

    def read(self):
        _, img = self.cam.read()
        return img


class ReplaySensor(BaseSensor):
    def read(self, outfile):
        return cv2.imread(outfile)

    def process(self, outfile):
        return transform_image(self.read(outfile), **self.sensor_params)


class DummySensor(BaseSensor):
    def read(self, outfile=None):
        return None

    def process(self, outfile=None):
        return None
