from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
    "caricature": {
        "transforms": transforms_config.CaricatureTransforms,
        "train_source_root": "../../datasets/lfw_real_sep/a-p",
        "train_target_root": "../../datasets/lfw_cari_sep/a-p",
        "test_source_root": "../../datasets/lfw_real_sep/q-z",
        "test_target_root": "../../datasets/lfw_cari_sep/q-z",
    },
    "ffhq_frontalize": {
        "transforms": transforms_config.FrontalizationTransforms,
        "train_source_root": dataset_paths["ffhq"],
        "train_target_root": dataset_paths["ffhq"],
        "test_source_root": dataset_paths["celeba_test"],
        "test_target_root": dataset_paths["celeba_test"],
    },
    "celebs_sketch_to_face": {
        "transforms": transforms_config.SketchToImageTransforms,
        "train_source_root": dataset_paths["celeba_train_sketch"],
        "train_target_root": dataset_paths["celeba_train"],
        "test_source_root": dataset_paths["celeba_test_sketch"],
        "test_target_root": dataset_paths["celeba_test"],
    },
    "celebs_seg_to_face": {
        "transforms": transforms_config.SegToImageTransforms,
        "train_source_root": dataset_paths["celeba_train_segmentation"],
        "train_target_root": dataset_paths["celeba_train"],
        "test_source_root": dataset_paths["celeba_test_segmentation"],
        "test_target_root": dataset_paths["celeba_test"],
    },
    "celebs_super_resolution": {
        "transforms": transforms_config.SuperResTransforms,
        "train_source_root": dataset_paths["celeba_train"],
        "train_target_root": dataset_paths["celeba_train"],
        "test_source_root": dataset_paths["celeba_test"],
        "test_target_root": dataset_paths["celeba_test"],
    },
}
