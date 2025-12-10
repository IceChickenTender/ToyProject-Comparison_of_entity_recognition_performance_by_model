from huggingface_hub import login
from huggingface_hub import HfApi

login(token='hf_token')

api = HfApi()
repo_id = "klue-bert-base-klue-ner-finetuned"
api.create_repo(repo_id=repo_id)

api.upload_folder(
    folder_path="./final_ner_model_paper_ver",
    repo_id=f"본인의 허깅페이스 사용자 이름/{repo_id}",
    repo_type="model",
)