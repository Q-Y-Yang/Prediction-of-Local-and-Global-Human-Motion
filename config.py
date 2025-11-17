from dataclasses import dataclass


@dataclass
class Human36MConfig:
    DATA_DIR = "./Human3.6M"


@dataclass
class HA4MConfig:
    DATA_DIR = "FILEPATH/TO/HA4MMDATA/HERE"


@dataclass
class AMASSConfig:
    DATA_DIR = "./AMASS"
    MODEL_DIR = "./SMPL-X models/models"
