"""
Scikit-learn instrumentation for automatic lineage capture.
"""
from __future__ import annotations

import functools
from typing import Any, Callable, Dict, List, Optional, Type

from tracepipe.core import (
    OperationType,
    get_code_location,
    get_context,
)

_ORIGINAL_METHODS: Dict[str, Callable] = {}
_INSTRUMENTED_CLASSES: List[Type] = []

try:
    import sklearn
    from sklearn.base import BaseEstimator, TransformerMixin
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    BaseEstimator = object
    TransformerMixin = object


def _wrap_fit(cls_name: str, original: Callable) -> Callable:
    @functools.wraps(original)
    def wrapper(self, X, y=None, **kwargs):
        ctx = get_context()
        
        if not ctx.enabled:
            return original(self, X, y, **kwargs)
        
        result = original(self, X, y, **kwargs)
        
        params = _extract_estimator_params(self)
        params["estimator"] = cls_name
        
        import numpy as np
        import pandas as pd
        
        input_data = None
        if isinstance(X, (np.ndarray, pd.DataFrame)):
            input_data = X
        
        ctx.graph.add_node(
            operation=OperationType.SKLEARN_FIT,
            operation_name=f"{cls_name}.fit",
            input_data=input_data,
            output_data=None,
            parameters=params,
            code_location=get_code_location(depth=3),
            metadata={"fitted_estimator_id": id(self)},
        )
        
        return result
    
    return wrapper


def _wrap_transform(cls_name: str, original: Callable) -> Callable:
    @functools.wraps(original)
    def wrapper(self, X, **kwargs):
        ctx = get_context()
        
        if not ctx.enabled:
            return original(self, X, **kwargs)
        
        result = original(self, X, **kwargs)
        
        import numpy as np
        import pandas as pd
        
        input_data = None
        if isinstance(X, (np.ndarray, pd.DataFrame)):
            input_data = X
        
        output_data = None
        if isinstance(result, (np.ndarray, pd.DataFrame)):
            output_data = result
        
        params = {"estimator": cls_name}
        
        ctx.graph.add_node(
            operation=OperationType.SKLEARN_TRANSFORM,
            operation_name=f"{cls_name}.transform",
            input_data=input_data,
            output_data=output_data,
            parameters=params,
            code_location=get_code_location(depth=3),
        )
        
        return result
    
    return wrapper


def _wrap_fit_transform(cls_name: str, original: Callable) -> Callable:
    @functools.wraps(original)
    def wrapper(self, X, y=None, **kwargs):
        ctx = get_context()
        
        if not ctx.enabled:
            return original(self, X, y, **kwargs)
        
        result = original(self, X, y, **kwargs)
        
        import numpy as np
        import pandas as pd
        
        input_data = None
        if isinstance(X, (np.ndarray, pd.DataFrame)):
            input_data = X
        
        output_data = None
        if isinstance(result, (np.ndarray, pd.DataFrame)):
            output_data = result
        
        params = _extract_estimator_params(self)
        params["estimator"] = cls_name
        
        ctx.graph.add_node(
            operation=OperationType.SKLEARN_TRANSFORM,
            operation_name=f"{cls_name}.fit_transform",
            input_data=input_data,
            output_data=output_data,
            parameters=params,
            code_location=get_code_location(depth=3),
            metadata={"fitted_estimator_id": id(self)},
        )
        
        return result
    
    return wrapper


def _wrap_predict(cls_name: str, original: Callable) -> Callable:
    @functools.wraps(original)
    def wrapper(self, X, **kwargs):
        ctx = get_context()
        
        if not ctx.enabled:
            return original(self, X, **kwargs)
        
        result = original(self, X, **kwargs)
        
        import numpy as np
        import pandas as pd
        
        input_data = None
        if isinstance(X, (np.ndarray, pd.DataFrame)):
            input_data = X
        
        output_data = None
        if isinstance(result, (np.ndarray, pd.DataFrame, pd.Series)):
            output_data = result if isinstance(result, np.ndarray) else result
        
        params = {"estimator": cls_name}
        
        ctx.graph.add_node(
            operation=OperationType.SKLEARN_PREDICT,
            operation_name=f"{cls_name}.predict",
            input_data=input_data,
            output_data=output_data,
            parameters=params,
            code_location=get_code_location(depth=3),
        )
        
        return result
    
    return wrapper


def _wrap_predict_proba(cls_name: str, original: Callable) -> Callable:
    @functools.wraps(original)
    def wrapper(self, X, **kwargs):
        ctx = get_context()
        
        if not ctx.enabled:
            return original(self, X, **kwargs)
        
        result = original(self, X, **kwargs)
        
        import numpy as np
        import pandas as pd
        
        input_data = None
        if isinstance(X, (np.ndarray, pd.DataFrame)):
            input_data = X
        
        output_data = None
        if isinstance(result, np.ndarray):
            output_data = result
        
        params = {"estimator": cls_name}
        
        ctx.graph.add_node(
            operation=OperationType.SKLEARN_PREDICT,
            operation_name=f"{cls_name}.predict_proba",
            input_data=input_data,
            output_data=output_data,
            parameters=params,
            code_location=get_code_location(depth=3),
        )
        
        return result
    
    return wrapper


def _extract_estimator_params(estimator: Any) -> Dict[str, Any]:
    params = {}
    try:
        est_params = estimator.get_params(deep=False)
        for key, value in est_params.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                params[key] = value
            elif isinstance(value, (list, tuple)) and len(value) <= 5:
                params[key] = list(value)
            else:
                params[key] = str(type(value).__name__)
    except Exception:
        pass
    return params


def _instrument_class(cls: Type) -> None:
    cls_name = cls.__name__
    
    if hasattr(cls, "fit"):
        key = f"{cls_name}.fit"
        if key not in _ORIGINAL_METHODS:
            original = cls.fit
            _ORIGINAL_METHODS[key] = original
            cls.fit = _wrap_fit(cls_name, original)
    
    if hasattr(cls, "transform"):
        key = f"{cls_name}.transform"
        if key not in _ORIGINAL_METHODS:
            original = cls.transform
            _ORIGINAL_METHODS[key] = original
            cls.transform = _wrap_transform(cls_name, original)
    
    if hasattr(cls, "fit_transform"):
        key = f"{cls_name}.fit_transform"
        if key not in _ORIGINAL_METHODS:
            original = cls.fit_transform
            _ORIGINAL_METHODS[key] = original
            cls.fit_transform = _wrap_fit_transform(cls_name, original)
    
    if hasattr(cls, "predict"):
        key = f"{cls_name}.predict"
        if key not in _ORIGINAL_METHODS:
            original = cls.predict
            _ORIGINAL_METHODS[key] = original
            cls.predict = _wrap_predict(cls_name, original)
    
    if hasattr(cls, "predict_proba"):
        key = f"{cls_name}.predict_proba"
        if key not in _ORIGINAL_METHODS:
            original = cls.predict_proba
            _ORIGINAL_METHODS[key] = original
            cls.predict_proba = _wrap_predict_proba(cls_name, original)
    
    _INSTRUMENTED_CLASSES.append(cls)


def instrument_sklearn():
    if not HAS_SKLEARN:
        return
    
    ctx = get_context()
    if ctx.is_instrumented("sklearn"):
        return
    
    from sklearn.preprocessing import (
        StandardScaler,
        MinMaxScaler,
        MaxAbsScaler,
        RobustScaler,
        Normalizer,
        Binarizer,
        LabelEncoder,
        OneHotEncoder,
        OrdinalEncoder,
        LabelBinarizer,
        PolynomialFeatures,
        PowerTransformer,
        QuantileTransformer,
    )
    from sklearn.impute import SimpleImputer
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.decomposition import PCA, TruncatedSVD, NMF
    from sklearn.feature_selection import (
        SelectKBest,
        SelectPercentile,
        VarianceThreshold,
    )
    
    preprocessing_classes = [
        StandardScaler,
        MinMaxScaler,
        MaxAbsScaler,
        RobustScaler,
        Normalizer,
        Binarizer,
        LabelEncoder,
        OneHotEncoder,
        OrdinalEncoder,
        LabelBinarizer,
        PolynomialFeatures,
        PowerTransformer,
        QuantileTransformer,
        SimpleImputer,
        CountVectorizer,
        TfidfVectorizer,
        PCA,
        TruncatedSVD,
        NMF,
        SelectKBest,
        SelectPercentile,
        VarianceThreshold,
    ]
    
    for cls in preprocessing_classes:
        _instrument_class(cls)
    
    try:
        from sklearn.linear_model import (
            LinearRegression,
            LogisticRegression,
            Ridge,
            Lasso,
            ElasticNet,
            SGDClassifier,
            SGDRegressor,
        )
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        from sklearn.ensemble import (
            RandomForestClassifier,
            RandomForestRegressor,
            GradientBoostingClassifier,
            GradientBoostingRegressor,
            AdaBoostClassifier,
            AdaBoostRegressor,
        )
        from sklearn.svm import SVC, SVR
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        from sklearn.naive_bayes import GaussianNB, MultinomialNB
        from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
        
        model_classes = [
            LinearRegression,
            LogisticRegression,
            Ridge,
            Lasso,
            ElasticNet,
            SGDClassifier,
            SGDRegressor,
            DecisionTreeClassifier,
            DecisionTreeRegressor,
            RandomForestClassifier,
            RandomForestRegressor,
            GradientBoostingClassifier,
            GradientBoostingRegressor,
            AdaBoostClassifier,
            AdaBoostRegressor,
            SVC,
            SVR,
            KNeighborsClassifier,
            KNeighborsRegressor,
            GaussianNB,
            MultinomialNB,
            KMeans,
            DBSCAN,
            AgglomerativeClustering,
        ]
        
        for cls in model_classes:
            _instrument_class(cls)
    except ImportError:
        pass
    
    ctx.mark_instrumented("sklearn")


def uninstrument_sklearn():
    if not HAS_SKLEARN:
        return
    
    for key, original in _ORIGINAL_METHODS.items():
        parts = key.split(".")
        if len(parts) == 2:
            cls_name, method_name = parts
            for cls in _INSTRUMENTED_CLASSES:
                if cls.__name__ == cls_name:
                    setattr(cls, method_name, original)
                    break
    
    _ORIGINAL_METHODS.clear()
    _INSTRUMENTED_CLASSES.clear()
    get_context()._instrumented_modules.discard("sklearn")
