from config import AMASSConfig
from data.amass import AMASSData

if __name__ == "__main__":
    # Pre-Process the Dataset
    amass = AMASSData(
        data_dir=AMASSConfig.DATA_DIR,
        model_dir=AMASSConfig.MODEL_DIR,
        sampling_rate=30,
        simple_model=True,
        preload_sequences=False,
        preprocess=True
    )
