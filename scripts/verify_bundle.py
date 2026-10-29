import json, numpy as np, pandas as pd, onnxruntime as ort
from sklearn.metrics import roc_auc_score

CACHE  = "data/cache/81f46ccd594df21f/"
BUNDLE = "bundle/v1.0.0/"
CSV    = "data/raw/files/ptb-xl/1.0.3/ptbxl_database.csv"

X, LM = np.load(CACHE + "signals.npy"), np.load(CACHE + "lead_masks.npy")
Y, TM = np.load(CACHE + "targets.npy"), np.load(CACHE + "target_masks.npy")
fold = pd.read_csv(CSV, index_col="ecg_id").strat_fold.values
assert len(fold) == len(X)   # if this fails, the cache row order differs from the CSV: check build-labels

sess = ort.InferenceSession(BUNDLE + "model.onnx")
names = [i.name for i in sess.get_inputs()]
print([(i.name, i.shape) for i in sess.get_inputs()])   # adapt the feed below if the inputs differ

def logits(idx):
    out = []
    for s in range(0, len(idx), 256):
        b = idx[s:s + 256]
        out.append(sess.run(None, {names[0]: X[b], names[1]: LM[b]})[0])
    return np.concatenate(out)

cal = json.load(open(BUNDLE + "calibration.json")); print("cal:", cal)
thr = json.load(open(BUNDLE + "thresholds.json"));  print("thr:", thr)
T   = np.array(cal.get("temperature", cal.get("temperatures")))      # adapt to the keys printed above
THR = np.array(thr["thresholds"])

val, test = np.where(fold == 9)[0], np.where(fold == 10)[0]
P = {n: 1 / (1 + np.exp(-logits(i) / T)) for n, i in [("val", val), ("test", test)]}

def macro_auc(p, idx):
    a = [roc_auc_score(Y[idx, k][TM[idx, k] == 1], p[:, k][TM[idx, k] == 1]) for k in range(5)]
    return np.round(a, 4), round(float(np.mean(a)), 4)

print("val ", macro_auc(P["val"], val))     # expect about 0.9109
print("test", macro_auc(P["test"], test))   # expect about 0.9112

def sens_spec(p, idx, thr):
    out = []
    for k in range(5):
        m = TM[idx, k] == 1; y = Y[idx, k][m]; yhat = p[:, k][m] >= thr[k]
        out.append((round(float((yhat & (y == 1)).sum() / (y == 1).sum()), 3),
                    round(float((~yhat & (y == 0)).sum() / (y == 0).sum()), 3)))
    return out

print("ResNet (sens, spec) per class at bundle thresholds, test:", sens_spec(P["test"], test, THR))

def thr_at_sens(y, p, target=0.90):          # threshold giving 90% sensitivity on validation
    return np.sort(p[y == 1])[int((1 - target) * (y == 1).sum())]

# Fair comparison with LightGBM
try:
    lgbm_p_val = np.load("runs/baseline/lgbm_val_probs.npy")
    lgbm_p_test = np.load("runs/baseline/lgbm_test_probs.npy")
    lgbm_thrs = []
    for k in range(5):
        m_val = TM[val, k] == 1
        th = thr_at_sens(Y[val, k][m_val], lgbm_p_val[:, k][m_val], target=0.90)
        lgbm_thrs.append(th)
    print("LightGBM (sens, spec) per class at 90% val sensitivity, test:", sens_spec(lgbm_p_test, test, lgbm_thrs))
except Exception as e:
    pass
