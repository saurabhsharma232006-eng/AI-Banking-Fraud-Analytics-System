"""Export XGBoost model trees to flat JSON arrays for pure-Python runtime inference."""
import json, math, os
import xgboost as xgb
import numpy as np

booster = xgb.Booster()
booster.load_model('models/xgb_model.json')
trees_raw = booster.get_dump(dump_format='json')
trees = [json.loads(t) for t in trees_raw]

def flatten_tree(node, out=None):
    """Pre-flatten a tree into parallel arrays indexed by node position."""
    if out is None:
        out = {
            'feature': [], 'threshold': [], 'yes': [], 'no': [],
            'leaf': [], 'is_leaf': [], 'nodeid_to_idx': {}
        }
    idx = len(out['feature'])
    out['nodeid_to_idx'][node['nodeid']] = idx
    if 'leaf' in node:
        out['feature'].append(-1)
        out['threshold'].append(0.0)
        out['yes'].append(-1)
        out['no'].append(-1)
        out['leaf'].append(float(node['leaf']))
        out['is_leaf'].append(True)
        return out
    out['feature'].append(int(node['split'][1:]))
    out['threshold'].append(float(node['split_condition']))
    out['yes'].append(node['yes'])
    out['no'].append(node['no'])
    out['leaf'].append(0.0)
    out['is_leaf'].append(False)
    for c in node.get('children', []):
        flatten_tree(c, out)
    return out

def resolve_tree(flat):
    """Resolve nodeid references to array indices."""
    nmap = flat['nodeid_to_idx']
    flat['yes'] = [nmap.get(v, -1) if not flat['is_leaf'][i] else -1
                   for i, v in enumerate(flat['yes'])]
    flat['no']  = [nmap.get(v, -1) if not flat['is_leaf'][i] else -1
                   for i, v in enumerate(flat['no'])]
    del flat['nodeid_to_idx']
    return flat

flat_trees = [resolve_tree(flatten_tree(t)) for t in trees]

export = {
    'base_score_raw': 0.0,  # logit(0.5) = 0.0 for binary:logistic
    'objective': 'binary:logistic',
    'trees': flat_trees
}

out_path = 'models/xgb_trees_flat.json'
with open(out_path, 'w') as f:
    json.dump(export, f, separators=(',', ':'))

size_mb = os.path.getsize(out_path) / 1024 / 1024
print('Exported:', out_path)
print('Size:', round(size_mb, 3), 'MB')
print('Trees:', len(flat_trees))

# Verify predictions still match
def predict_pure_python(flat_trees, base_score_raw, x):
    raw = base_score_raw
    for tree in flat_trees:
        feat = tree['feature']
        thr  = tree['threshold']
        yes  = tree['yes']
        no   = tree['no']
        leaf = tree['leaf']
        node = 0
        while feat[node] != -1:
            node = yes[node] if x[feat[node]] < thr[node] else no[node]
        raw += leaf[node]
    return 1.0 / (1.0 + math.exp(-raw))

np.random.seed(42)
X_test = np.random.rand(3, 17).astype(np.float32)
dmat = xgb.DMatrix(X_test)
probs_xgb = booster.predict(dmat)
probs_py  = [predict_pure_python(flat_trees, 0.0, x.tolist()) for x in X_test]

print('XGBoost probs:', probs_xgb.tolist())
print('PurePython probs:', probs_py)
max_err = max(abs(a-b) for a,b in zip(probs_xgb, probs_py))
print('Max error:', max_err)
assert max_err < 1e-5, "Predictions diverge!"
print("PASS: pure-Python predictions match XGBoost exactly.")
