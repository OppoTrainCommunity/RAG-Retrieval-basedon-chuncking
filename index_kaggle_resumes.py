import pandas as pd
from cv_store import index_resume_text_in_chroma

df = pd.read_csv("data/UpdatedResumeDataSet.csv")
df = df.head(20)

print("Loaded dataset with shape:", df.shape)

resume_ids = []

for i, row in df.iterrows():
    resume_text = row["Resume"]
    category = row["Category"]
    resume_id = f"kaggle_row_{i}"
    rid = index_resume_text_in_chroma(resume_text, category, resume_id=resume_id)
    resume_ids.append(rid)
    print("Indexed", i + 1)

print("Indexed", len(resume_ids), "resumes from dataset.")
