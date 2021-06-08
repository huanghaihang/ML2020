from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.THRESHOLD = 0.5
_C.MODEL.NUM_CLASSES = 21
_C.MODEL.NEG_POS_RATIO = 3
_C.MODEL.CENTER_VARIANCE = 0.1
_C.MODEL.SIZE_VARIANCE = 0.2

_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = 'vgg'
_C.MODEL.BACKBONE.OUT_CHANNELS = (512, 1024, 512, 256, 256, 256)
_C.MODEL.BACKBONE.PRETRAINED = True

# -----------------------------------------------------------------------------
# PRIORS
# -----------------------------------------------------------------------------
_C.MODEL.PRIORS = CN()
_C.MODEL.PRIORS.FEATURE_MAPS = [38, 19, 10, 5, 3, 1]
_C.MODEL.PRIORS.STRIDES = [8, 16, 32, 64, 100, 300]
_C.MODEL.PRIORS.MIN_SIZES = [30, 60, 111, 162, 213, 264]
_C.MODEL.PRIORS.MAX_SIZES = [60, 111, 162, 213, 264, 315]
_C.MODEL.PRIORS.ASPECT_RATIOS = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
_C.MODEL.PRIORS.BOXES_PER_LOCATION = [4, 6, 6, 6, 4, 4]
_C.MODEL.PRIORS.CLIP = True

_C.MODEL.BOX_HEAD = CN()
_C.MODEL.BOX_HEAD.NAME = 'SSDBoxHead'
_C.MODEL.BOX_HEAD.PREDICTOR = 'SSDBoxPredictor'

_C.INPUT = CN()
_C.INPUT.IMAGE_SIZE = 300
_C.INPUT.PIXEL_MEAN = [123, 117, 104]

_C.DATASETS = CN()
_C.DATASETS.TRAIN = ()
_C.DATASETS.TARGET = ()
_C.DATASETS.TEST = ()

_C.DATA_LOADER = CN()
_C.DATA_LOADER.NUM_WORKERS = 8
_C.DATA_LOADER.PIN_MEMORY = True

_C.SOLVER = CN()
_C.SOLVER.MAX_ITER = 120000
_C.SOLVER.LR_STEPS = [80000, 100000]
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.BATCH_SIZE = 32
_C.SOLVER.BACKBONELR = 1e-3
_C.SOLVER.BOXHEADLR = 1e-3
_C.SOLVER.DOMAINDISCRIMINATORLR = 1e-3
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 5e-4
_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 500
_C.SOLVER.LAMBDA = 0.5

_C.TEST = CN()
_C.TEST.NMS_THRESHOLD = 0.45
_C.TEST.CONFIDENCE_THRESHOLD = 0.01
_C.TEST.MAX_PER_CLASS = -1
_C.TEST.MAX_PER_IMAGE = 100
_C.TEST.BATCH_SIZE = 10

_C.OUTPUT_DIR = 'outputs'
