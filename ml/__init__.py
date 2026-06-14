try:
    from .predictor import MLPredictor
    __all__ = ["MLPredictor"]
except Exception:
    __all__ = []
