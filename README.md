# almanac

## Environment Setup
To install environment, run:
```
conda env create  --file conda.yml
conda activate almanac-env
python -m ipykernel install --user --name almanac-kernel --display-name "almanac kernel"`
pre-commit install
```