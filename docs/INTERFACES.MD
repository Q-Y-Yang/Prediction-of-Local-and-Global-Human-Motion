# Interfaces

This repository defines interfaces for Datasets, Metrics, Pose Representations and Training Loggers.

Find out how to add your own Datasets -> [here](./DATASETS.MD)

**Metrics:** New Metrics that inherit from the [``Metric``](../metrics/metrics.py) base class can be used interchangeably with those used in the training and evaluation scripts.

**Pose Representations:** New Pose Representations that inherit from the [``PoseRepresentation``](../data/pose_representation/base.py) base class can be used interchangeably with those defined in this repository.

They can be used flawlessly in the [``main``](training.py) training function if further the enum ``RepresentationType`` and util function ``get_pose_representation`` located in [util/pose_representation.py](../util/pose_representation.py) are extended for the new Pose Representation.

**Training Loggers:** Custom Training Loggers can be integrated into the training pipeline in this repository. The custom Logger needs to implement the [``Logger``](../loggers/logger.py) base class.

The new Logger can be used together with the existing [``TensorboardLogger``](../loggers/tensorboard.py) by adding the new Logger to the ``MultiLogger`` defined in the [``main``](training.py) training routine as follows:

```Python
logger = MultiLogger(
    log_dir=log_dir,
    loggers=[
        TensorboardLogger(
            log_dir=log_dir, hyperparameters=hyperparameters
        ),
        YourLogger(...)
    ]
)
```

Your added logger will be supplied with the following information during training:
- ``_add_scalar(..)`` - Metric values calculated for every batch
- ``_add_scalar_epoch(..)`` - Mean Metric values of an epoch
- ``_add_figure(..)`` - visualizations created during training
- ``_add_hparams(..)`` - dictionary of model hyperparameter settings + dictionary of metrics calculated during the best validation epoch
