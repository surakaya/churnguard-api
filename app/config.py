from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Model versiyonunu ortam değişkeniyle yönetmek için tek giriş noktası.
    MODEL_VERSION: str = "churn_lr_v1"

    model_config = {
        # Ortam değişkenleri CHURNGUARD_ prefix'i ile okunur.
        "env_prefix": "CHURNGUARD_",
    }


settings = Settings()
