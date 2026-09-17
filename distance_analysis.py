import numpy as np
import pandas as pd
import re, glob

def parse_ph4(filename, features_dat=None):
  feature_rows = []

  n_features = None
  reading_features = False
  features_read = 0

  with open(filename, 'r') as f:
    for line in f:
      stripped = line.strip()

      # all the initial lines before #feature is read are part of the header
      if stripped.startswith("#feature"):
        parts = stripped.split()
        n_features = int(parts[1])
        reading_features = True
        continue

      if reading_features and features_read < n_features:
        if not stripped or stripped.startswith("#"): continue

        parts = stripped.split()
        # based on example format
        feature_rows.append({
          "type": parts[0],
          "color": parts[1],
          "x": float(parts[2]),
          "y": float(parts[3]),
          "z": float(parts[4]),
          "radius": float(parts[5]),
          "ebits": int(parts[6]),
          "gbits": int(parts[7]),
        })
        features_read += 1

      if reading_features and features_read == n_features: break

  features_df = pd.DataFrame(feature_rows)

  if features_dat:
    # Read in <prot>.features.dat file with the FGFE for each feature
    # Type | Sphere Center X | Y | Z | Sphere radius R | FGFE

    feat_dat = pd.read_csv(features_dat, sep='\\s+', skiprows=2, names=['type','x','y','z','radius','FGFE'])
    for c in ['x','y','z']: 
      feat_dat[c] = feat_dat[c].round(4) 
      features_df[c] = features_df[c].round(4)
    cols = ['x', 'y', 'z']

    merged = features_df.merge(feat_dat[cols + ['FGFE']], on=cols, how='left')
    features_df['FGFE'] = merged['FGFE'].round(4)
 
  return features_df

def load_expt_hypotheses(filename):
  lists = []
  with open(filename) as f:
    for line in f:
      lists.append(eval(line.strip()))
  return lists

def parse_ph4gen_out(filename):
  results = {}
  pattern = re.compile( r'^\s*\d+\s+([0-9.]+)\s+\[([0-9,\s]+)\]')
  with open(filename) as f:
    rank = 0
    for line in f:
      rank+=1
      match = pattern.match(line)
      if match:
        score = float(match.group(1))
        hypothesis = [int(x) for x in match.group(2).split(',')]
        #results[score] = [hypothesis]
        results[rank] = [score, hypothesis]
  df = pd.DataFrame(results).transpose().reset_index(drop=True).reset_index()
  df.columns = ['rank', 'score','hypothesis']
  #df['rank'] += 1
  return df

def rmsd(c1, c2):
  # assumes coordinates match
  return np.sqrt(((c2-c1)**2).sum(-1).mean())

def diffcen(c1, c2):
  # distance between centers
  cen1 = c1.mean()
  cen2 = c2.mean()
  return np.sqrt(((cen2-cen1)**2).sum())

from scipy.optimize import linear_sum_assignment

def matched_rmsd(coords1, coords2, indices1=None, indices2=None):
  coords1 = np.asarray(coords1)
  coords2 = np.asarray(coords2)

  if indices1 is None:
    indices1 = np.arange(len(coords1))
  if indices2 is None:
    indices2 = np.arange(len(coords2))

  indices1 = np.asarray(indices1)
  indices2 = np.asarray(indices2)

  # First match identical indices
  common = np.intersect1d(indices1, indices2)

  matched1 = []
  matched2 = []

  for idx in common:
    matched1.append(np.where(indices1 == idx)[0][0])
    matched2.append(np.where(indices2 == idx)[0][0])

  # Remaining points
  rem1 = np.array([i for i in range(len(coords1))
                   if i not in matched1])
  rem2 = np.array([i for i in range(len(coords2))
                   if i not in matched2])

  # Match remaining points by minimum total spatial distance
  n = min(len(rem1), len(rem2))

  if n > 0:
    dists = np.linalg.norm(
      coords1[rem1, None, :] - coords2[None, rem2, :],
      axis=2
    )

    rows, cols = linear_sum_assignment(dists)

    # Only use as many matches as possible
    rows = rows[:n]
    cols = cols[:n]

    matched1.extend(rem1[rows])
    matched2.extend(rem2[cols])

  # RMSD over matched points
  matched1 = np.asarray(matched1)
  matched2 = np.asarray(matched2)

  diff = coords1[matched1] - coords2[matched2]
  rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

  return rmsd, len(matched1)

systems_list = ['bace','cdk2','jnk1','p38','ptp1b','thrombin','tyk2', 'fxr','hdm2','hsp90','trmd']
types        = ['training','training','training','training','training','training','training','validation','validation','validation','validation']

results = pd.DataFrame(
  index=range(1, 21),
  columns=pd.MultiIndex.from_product(
    #[systems_list, ['Ne', 'Nh', 'Nm', 'dCen', 'rmsd','hit']]
    [systems_list, ['hit', 'rmsd', 'dCen', 'Nm']]
  )
)
for sys,ftype in zip(systems_list,types):
  expt=[]
  for i in [4,5]:
    expt += load_expt_hypotheses(f"{ftype}/{sys}/both_hypotheses_{i}.txt")
  dfpred = parse_ph4gen_out(f"{ftype}/{sys}/output/ph4gen.out")
  dfph4 = parse_ph4(glob.glob(f"{ftype}/{sys}/*.keyf_*.ph4")[0])
  # NOTE: the feature indices in dfpred (eg hyp=[0,1,2]) can be used as direct indices of dfph4 since both start from 0
  
  for i,row in dfpred.iterrows():
    seth = set(row['hypothesis'])
    hcoords = dfph4.loc[row['hypothesis'], ['x','y','z']].to_numpy()
    max_n_common = 0
    maxlist = []
    maxset = set()
    subset=False
    #results.loc[i + 1, sys] = maxi
    for listj in expt:
      sete = set(listj)
      n_common = len(seth & sete) # intersection 
      if n_common > max_n_common: 
        max_n_common = n_common
        maxlist = listj
      if seth <= sete: subset=True

    matchcoords = dfph4.loc[maxlist, ['x','y','z']].to_numpy()
    #if len(hcoords) == len(matchcoords): print(maxlist, row['hypothesis'])
    rmsd, nmatch = matched_rmsd(hcoords, matchcoords, row['hypothesis'], maxlist)
    deltacen = diffcen(hcoords,matchcoords)
    #results.loc[i + 1, (sys, 'Ne')] = len(listj)
    #results.loc[i + 1, (sys, 'Nh')] = len(row['hypothesis'])
    results.loc[i + 1, (sys, 'Nm')] = max_n_common
    results.loc[i + 1, (sys, 'dCen')] = deltacen
    results.loc[i + 1, (sys, 'rmsd')] = rmsd
    results.loc[i + 1, (sys, 'hit')] = subset


training_systems = [ sys for sys, ftype in zip(systems_list, types) if ftype == 'training' ]
validation_systems = [ sys for sys, ftype in zip(systems_list, types) if ftype == 'validation' ]

pd.set_option('display.precision', 1)
print(results[training_systems].head(10))
print(results[validation_systems].head(10))

results[training_systems].head(10).to_csv('training.csv')
results[validation_systems].head(10).to_csv('validation.csv')
