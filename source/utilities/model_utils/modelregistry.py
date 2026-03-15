import lightning

AVAILABLE_MODELS = {}
def register(
        cls: type[lightning.LightningModule]
    ) -> type[lightning.LightningModule]:
    AVAILABLE_MODELS[cls.__name__] = cls
    return cls