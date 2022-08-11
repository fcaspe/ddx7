# Dataset generation

1. Please populate the `data_config.yaml` file with the path to the `Dataset` directory present in the URMP dataset.

2. If you wish to generate a testset using audio from a different source, you can populate the secion `testset` of the `data_config.yaml` file
with a path to a location that contains directories with instrument names. Each directory should contain `.wav` files of the corresponding instrument.
Also, make sure you run `create_data.py` with the option `process_testset=True`.

3. You can also use the `testset` section to process external files into continuous test sets, i.e. not chopped. Just make sure the option `testset.contiguous`
is set to `True`. Then, load the additional testset when training with the option `load_additional_testset: True` located in `recipes/config.yaml`.


