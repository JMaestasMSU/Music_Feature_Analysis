from typing import TYPE_CHECKING


if TYPE_CHECKING:
    # for static type checkers
    from pydantic import BaseSettings as _BaseSettings  # type: ignore
else:
    try:
        # pydantic v1 or the compatibility API
        from pydantic import BaseSettings as _BaseSettings
    except Exception:
        try:
            # pydantic v2 moved BaseSettings to pydantic-settings
            from pydantic_settings import BaseSettings as _BaseSettings
        except Exception:
            # fallback stub for runtime-less type checking
            class _BaseSettings:  # type: ignore
                pass


class Settings(_BaseSettings):
    app_name: str = "music-feature-analysis"
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000
    model_path: str = ""
    model_backend: str = "sklearn"  # or 'torch'

    class Config:
        env_file = ".env"


settings = Settings()

