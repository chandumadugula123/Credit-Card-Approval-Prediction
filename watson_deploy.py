from __future__ import annotations

import os
from pathlib import Path

from ibm_watson_machine_learning import APIClient


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "credit_card_approval_model.joblib"


def deploy_to_watson() -> None:
    required = ["IBM_WML_API_KEY", "IBM_WML_URL", "IBM_WML_SPACE_ID"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing IBM Watson environment variables: {', '.join(missing)}")

    client = APIClient(
        {
            "apikey": os.environ["IBM_WML_API_KEY"],
            "url": os.environ["IBM_WML_URL"],
        }
    )
    client.set.default_space(os.environ["IBM_WML_SPACE_ID"])

    metadata = {
        client.repository.ModelMetaNames.NAME: "Credit Card Approval Prediction",
        client.repository.ModelMetaNames.TYPE: "scikit-learn_1.3",
        client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: client.software_specifications.get_id_by_name(
            "runtime-23.1-py3.10"
        ),
    }

    stored_model = client.repository.store_model(str(MODEL_PATH), meta_props=metadata)
    model_id = client.repository.get_model_id(stored_model)
    deployment = client.deployments.create(
        artifact_uid=model_id,
        meta_props={
            client.deployments.ConfigurationMetaNames.NAME: "Credit Card Approval API",
            client.deployments.ConfigurationMetaNames.ONLINE: {},
        },
    )
    print("Deployment ID:", client.deployments.get_id(deployment))


if __name__ == "__main__":
    deploy_to_watson()
