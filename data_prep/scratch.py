from deepforest import main
import comet_ml
import PIL.Image
from pytorch_lightning.loggers import CometLogger


# Log comet experiment

experiment = CometLogger()

PIL.Image.MAX_IMAGE_PIXELS = 3000145511

tile_path = "/blue/ewhite/everglades/projected_mosaics/2022/Joule/Joule_02_11_2022_projected.tif"
model = main.deepforest(
    config_args={
        "num_classes": 7,
        "label_dict": {
            "Anhinga": 6,
            "Great Blue Heron": 3,
            "Great Egret": 0,
            "Roseate Spoonbill": 1,
            "Snowy Egret": 5,
            "White Ibis": 2,
            "Wood Stork": 4,
        },
    }
)
model.load_model("weecology/everglades-bird-species-detector")
#model.create_trainer(logger=experiment)
boxes = model.predict_tile(path=tile_path, patch_overlap=0, patch_size=1500)