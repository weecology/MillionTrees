from torchdata.datapipes.iter import FileLister, FileOpener
from PIL import Image
import io
import numpy as np
from deepforest import main
from matplotlib import pyplot as plt

datapipe1 = FileLister("/Users/benweinstein/Downloads/charlottesville", "*")
datapipe2 = FileOpener(datapipe1, mode="b")
tfrecord_loader_dp = iter(datapipe2.load_from_tfrecord())

m = main.deepforest()
m.use_release()

for x in range(10):
    example = next(tfrecord_loader_dp)
    print(example.keys())

    # Read bytes image and view
    image_data = np.array(example["streetlevel/encoded"])
    image = Image.open(io.BytesIO(image_data))
    image.show()

    # Create numpy array from image
    image_array = np.array(image)
    plt.imshow(image_array)
    plt.show()

    predicted_image = m.predict_image(image_array, return_plot=True)
    plt.imshow(predicted_image[:,:,::-1])
    plt.show()


# Calgary orthophoto links
https://www.arcgis.com/apps/mapviewer/index.html?webmap=823b8c06c5544c1b825c7dd5da96d35a