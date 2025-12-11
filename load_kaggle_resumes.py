# pip install kagglehub[pandas-datasets]

import kagglehub
from kagglehub import KaggleDatasetAdapter

file_path = "UpdatedResumeDataSet.csv"

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "gauravduttakiit/resume-dataset",
    file_path,
)

print(df.head())
print(df.columns)
