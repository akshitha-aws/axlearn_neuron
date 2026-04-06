# Copyright © 2023 Apple Inc.

"""Main function for launching the trainer."""

from absl import app, flags
from jax.sharding import PartitionSpec

from axlearn.common import launch, launch_trainer, measurement
from axlearn.common.config import config_for_function, register_validator

register_validator(
    match_fn=lambda v: isinstance(v, PartitionSpec),
    validate_fn=lambda _: None,
)


def main(_):
    measurement.initialize(flags.FLAGS)
    launch.setup()
    trainer_config = launch_trainer.get_trainer_config()
    trainer_config.set(recorder=config_for_function(lambda: measurement.global_recorder))
    measurement.start_monitoring()
    launch_trainer.run_trainer(trainer_config)


if __name__ == "__main__":
    measurement.define_flags()
    app.run(main)
