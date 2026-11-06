"""Small fixed scikit-learn catalogue; all scaling is fitted on eligible rows."""
from dataclasses import dataclass
import warnings
import numpy as np
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from .supervised import recursive_rollout,FeatureSpec


@dataclass(frozen=True)
class TabularSpec:
    name: str
    family: str
    alpha: float = 1.
    strategy: str = 'direct'
    seed: int = 1731

    def __post_init__(self):
        if self.family not in {'ridge','elastic','svr','forest','boost'} or self.strategy not in {'direct','recursive'}:
            raise ValueError("Unknown tabular family or strategy")
        if not np.isfinite(self.alpha) or self.alpha<=0:
            raise ValueError("Regularization must be positive")


def tabular_models():
    return [TabularSpec('Ridge_0.1','ridge',.1),TabularSpec('Ridge_10','ridge',10.),
            TabularSpec('ElasticNet','elastic',.03),TabularSpec('SVR_RBF','svr'),
            TabularSpec('RandomForest','forest'),TabularSpec('HistBoost','boost'),
            TabularSpec('Ridge_recursive','ridge',10.,strategy='recursive')]


class TabularFit:
    def __init__(self, model, xscale, yscale, spec):
        self.model,self.xscale,self.yscale,self.spec=model,xscale,yscale,spec

    def predict(self, sequence, track='returns', feature_spec=FeatureSpec()):
        def native(seq):
            flat=np.asarray(seq).reshape(1,-1)
            with threadpool_limits(limits=1):
                result=np.asarray(self.model.predict(self.xscale.transform(flat))).reshape(1,-1)
            return self.yscale.inverse_transform(result)[0]
        return (native(sequence) if self.spec.strategy=='direct' else
                recursive_rollout(lambda seq:native(seq)[0],sequence,track,feature_spec))


def fit_tabular(X,y,spec):
    X,y=np.asarray(X),np.asarray(y)
    if X.ndim!=3 or y.shape!=(len(X),5) or len(X)<40 or not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Need finite sequence examples and five mature targets")
    flat=X.reshape(len(X),-1)
    target=y if spec.strategy=='direct' else y[:,:1]
    xscale,yscale=StandardScaler().fit(flat),StandardScaler().fit(target)
    features,labels=xscale.transform(flat),yscale.transform(target)
    if spec.family=='ridge':
        model=Ridge(alpha=spec.alpha)
    elif spec.family=='elastic':
        model=MultiOutputRegressor(ElasticNet(alpha=spec.alpha,l1_ratio=.5,max_iter=5000))
    elif spec.family=='svr':
        model=MultiOutputRegressor(SVR(C=1.,epsilon=.1,kernel='rbf'))
    elif spec.family=='forest':
        model=RandomForestRegressor(n_estimators=48,max_depth=4,min_samples_leaf=10,
                                    random_state=spec.seed,n_jobs=1)
    else:
        model=MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=60,max_depth=2,
            learning_rate=.05,l2_regularization=1.,early_stopping=False,random_state=spec.seed))
    with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        model.fit(features,labels if labels.shape[1]>1 or spec.family not in {'forest'} else labels[:,0])
    if any(issubclass(w.category,ConvergenceWarning) for w in captured):
        raise RuntimeError('Nonconverged estimator: '+' | '.join(str(w.message) for w in captured))
    return TabularFit(model,xscale,yscale,spec),{'warnings':' | '.join(str(w.message) for w in captured),
        'n_train':len(X),'feature_mean':xscale.mean_.tolist(),'target_mean':yscale.mean_.tolist()}
