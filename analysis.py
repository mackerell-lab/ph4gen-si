import ph4gen
import pandas as pd
import numpy as np
import glob
import itertools
from scipy.spatial.distance import cdist
from scipy.special import expit
from sklearn.metrics import precision_recall_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GroupKFold, KFold
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
import argparse


# https://arxiv.org/abs/2107.02270 Petroff, Accessible Color Sequences for Data Visualization
c = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]

# for subsequent training; nPGFE is normed Pharmacophore GFE, (pgfe-m)/s
x_features = ['nPGFE', 'aro_count', 'acc_count', 'don_count', 'ani_count', 'cat_count', 'max_dist', 'min_dist', 'mean_dist', 'centroid_dist', 'mean_separation'] 

def load_expt_hypotheses(filename):
  lists = []
  with open(filename) as f:
    for line in f:
      lists.append(eval(line.strip()))
  return lists

def analyze_hypotheses(features_df, center, n_feat, centroid_cutoff, separation_cutoff, min_aro, min_hb, max_ani, min_dist_cutoff, max_dist_cutoff, mean_dist_cutoff):
  indices = features_df.index
  # NOTE: $m is & or "AND"; $m$h is &! or "AND NOT"
  aro_hyd_indices = features_df[(features_df['type'].str.contains('Aro|Hyd', regex=True))].index
  don_indices = features_df[(features_df['type'].str.contains('Don')) & ~ (features_df['type'].str.contains('$hDon'))].index
  acc_indices = features_df[(features_df['type'].str.contains('Acc')) & ~ (features_df['type'].str.contains('$hAcc'))].index
  cat_indices = features_df[(features_df['type'].str.contains('Cat')) & ~ (features_df['type'].str.contains('$hCat'))].index
  ani_indices = features_df[(features_df['type'].str.contains('Ani')) & ~ (features_df['type'].str.contains('$hAni'))].index

  # Enumeration of combinations
  hypotheses = np.array(list(itertools.combinations(indices, n_feat)))

  # Properties
  points = features_df[['x','y','z']].to_numpy()
  dist = np.linalg.norm(points-center, axis=1)
  mutual_dist = cdist(points, points)
  # Counts
  aro_count = np.isin(hypotheses, aro_hyd_indices).sum(axis=1)
  don_count = np.isin(hypotheses, don_indices).sum(axis=1)
  acc_count = np.isin(hypotheses, acc_indices).sum(axis=1)
  hb_count  = np.isin(hypotheses, don_indices).sum(axis=1) + np.isin(hypotheses, acc_indices).sum(axis=1)
  ani_count = np.isin(hypotheses, ani_indices).sum(axis=1)
  cat_count = np.isin(hypotheses, cat_indices).sum(axis=1)
  # Distances
  centroid_dist = np.array([np.linalg.norm(points[idx_list].mean(axis=0)-center) for idx_list in hypotheses])
  mean_separation = np.array([mutual_dist[idx_list].mean() for idx_list in hypotheses])
  max_dist = np.array([dist[idx_list].max() for idx_list in hypotheses])
  min_dist = np.array([dist[idx_list].min() for idx_list in hypotheses])
  mean_dist = np.array([dist[idx_list].mean() for idx_list in hypotheses])
  # P(harm)GFE = sum of F(eature)GFE
  fgfe = features_df['FGFE'].to_numpy()
  pgfe = np.array([fgfe[idx_list].sum() for idx_list in hypotheses])

  # Create masks from rules: (default= keep all)
  mask_aro = aro_count >= min_aro                 # at least <min_aro> aro|hyd
  mask_hb = hb_count >= min_hb                    # at least 1 don|acc
  mask_ani = ani_count <= max_ani                 # at most <max_ani> ani
  mask_centroid = centroid_dist < centroid_cutoff # centroid distance from center less than 
  mask_sep = mean_separation < separation_cutoff  # separation/mean_separation maximum
  mask_maxdist = max_dist <  max_dist_cutoff      # max distance of any feature to center no more than
  mask_mindist = min_dist < min_dist_cutoff       # min distance of any feature to center no more than
  mask_meandist = mean_dist < mean_dist_cutoff    # mean distance of any feature to center no more than
  
  # Consolidate masks: (default = keep all)
  keep = mask_aro & mask_hb & mask_centroid & mask_sep & mask_maxdist & mask_mindist & mask_meandist
  
  df = pd.DataFrame({
  "hypothesis": list(hypotheses),
  "keep": keep,
  "PGFE": pgfe,
  "aro_count": aro_count,
  "acc_count":acc_count,
  "don_count":don_count,
  "hb_count": hb_count,
  "ani_count": ani_count,
  "cat_count": cat_count,
  "max_dist": max_dist,
  "min_dist": min_dist,
  "mean_dist": mean_dist,
  "mean_separation": mean_separation,
  "centroid_dist": centroid_dist
  })
  df["hypothesis_list"] = df["hypothesis"].apply(lambda h: [int(hi) for hi in h])
  df["hypothesis_set"]  = df["hypothesis"].apply(lambda h: set([int(hi) for hi in h]))
  return df

parser = argparse.ArgumentParser()
parser.add_argument("--n_feat",            type=int,   default=None,   help="Number of features; default=%(default)s")
parser.add_argument("--train_val",         type=str,   default="training", choices=['training','validation','both'], help="Training or Validation; default=%(default)s")
parser.add_argument("--min_aro",           type=int,   default=0,      help="Minimum aromatic features; default=%(default)s")
parser.add_argument("--max_ani",           type=int,   default=4,      help="Maximum anionic features; default=%(default)s")
parser.add_argument("--min_hb",            type=int,   default=0,      help="Minimum hydrogen bond donor/acceptor features; default=%(default)s")
parser.add_argument("--centroid_cutoff",   type=float, default=np.inf, help="Maximum for centroid distance to center; default=%(default)s")
parser.add_argument("--max_dist_cutoff",   type=float, default=np.inf, help="Maximum for Max. feature distance to center; default=%(default)s")
parser.add_argument("--min_dist_cutoff",   type=float, default=np.inf, help="Maximum for Min. feature distance to center; default=%(default)s")
parser.add_argument("--mean_dist_cutoff",  type=float, default=np.inf, help="Maximum for Mean feature distance to center; default=%(default)s")
parser.add_argument("--separation_cutoff", type=float, default=np.inf, help="Maximum of Mean separation between features; default=%(default)s")
parser.add_argument("--ktop",              type=int,   default=20,     help="Top k hypotheses to be selected; default=%(default)s")
parser.add_argument("--system",            type=str,   default=None,   help="Individual system, loops over all if not supplied; default=%(default)s")
parser.add_argument("--quiet",             action="store_true",        help="Turn off plotting; default=%(default)s")
parser.add_argument("--justpr",            action="store_true",        help="Plot only Prec-Recall; default=%(default)s")
parser.add_argument("--no-log_flag",action="store_false",dest='log_flag',help="Use logarithmic y-scale histograms")
parser.add_argument("--expt",              action="store_true",        help="Just show experimental distributions")
parser.add_argument("--lig_crds", choices=['smc','xtal','both'],       default='both', help="Use SILCS-MC or X-tal coords of ligands to identify hypotheses")
parser.add_argument("--factor",            type=float,  default=1,     help="Round model weights to nearest 1/f; f=0 will result in no rounding; default=%(default)s")
parser.add_argument("--final",             action='store_true',        help="Use final production model rather than fitting")
parser.add_argument("--rfe",               action='store_true',        help="Recursive feature elimination")
parser.add_argument("--prefer",            default='small',            help="Prefer a large or small hypothesis size")
args = parser.parse_args()

len_training = 7
if args.train_val == 'training':
  systems_list = ['bace','cdk2','jnk1','p38','ptp1b','thrombin','tyk2'] if args.system is None else [args.system]
  pretty_system = ['BACE1','CDK2','JNK1','P38','PTP1B','Thrombin','TYK2']
elif args.train_val == 'validation':
  systems_list = ['hsp90','hdm2','fxr','trmd'] if args.system is None else [args.system]
  pretty_system = ['HSP90','HDM2','FXR','TRMD']
elif args.train_val == 'both':
  systems_list = ['bace','cdk2','jnk1','p38','ptp1b','thrombin','tyk2','hsp90','hdm2','fxr','trmd']
  pretty_system = ['BACE1','CDK2','JNK1','P38','PTP1B','Thrombin','TYK2','HSP90','HDM2','FXR','TRMD']
else:
  print('not an option for train_val')
 
n_hypotheses_system = []
hypotheses_system = {}
all_df = pd.DataFrame()
for system in systems_list:
  # use ph4gen.py to parse the ph4 files
  features_df = ph4gen.parse_ph4(glob.glob(f"{args.train_val}/{system}/*keyf*ph4")[0], features_dat=glob.glob(f"{args.train_val}/{system}/*.features.dat")[0])
  center_str = open(glob.glob(f"{args.train_val}/{system}/center.txt")[0]).read().strip()
  center = np.array([float(i) for i in center_str.split(',')])

  ####################################
  # load core features
  ###########
 
  # Analyze hypotheses, generating information
  # Loop over N_Features
  expt_sets = []
  for nfeat in [4,5] if not args.n_feat else [args.n_feat]:
    df = analyze_hypotheses(features_df, center, centroid_cutoff=args.centroid_cutoff, n_feat=nfeat, min_aro=args.min_aro, 
                            min_hb=args.min_hb, max_ani=args.max_ani, max_dist_cutoff=args.max_dist_cutoff, min_dist_cutoff=args.min_dist_cutoff,
                            separation_cutoff=args.separation_cutoff, mean_dist_cutoff=args.mean_dist_cutoff)
    features_df['dist'] = np.linalg.norm(features_df[['x','y','z']].to_numpy()-center, axis=1)
    # Consoliate to all_df
    df['system'] = system

    # load experimental hypotheses indices
    if args.lig_crds in ['smc','xtal','both']: 
      prefix=f"{args.lig_crds}_"
      filename=glob.glob(f"{args.train_val}/{system}/{prefix}hypotheses_{nfeat}.txt")[0]
    expt = load_expt_hypotheses(filename)
    for e in expt:
      expt_sets.append(set(e))

    # concatenate all (mcl1 is removed)
    all_df = pd.concat([all_df, df])
 
  print(system, expt_sets)
  # Loop over enumerated hypotheses (list of indices of feature_df), compare with set of expt (true) hypotheses
  # set to 1 (true) if the enumerated hypothesis is in the subset which is actually in expt
  mask = all_df['system'] == system
  
  all_df.loc[mask, "expt"] = (
    all_df.loc[mask, "hypothesis_set"]
          .apply(lambda h: any(h <= E for E in expt_sets))
  )
  
  all_df.loc[mask, "expt_support"] = (
    all_df.loc[mask, "hypothesis_set"]
          .apply(lambda h: [E for E in expt_sets if h <= E])
  )

all_df["expt"] = all_df["expt"].astype(bool)
#exit()
    
#############
# Cross-Validate Logistic Regression scoring function
#############
def precision_at_k_scorer(estimator, X, y, k=20):
  scores = estimator.predict_proba(X)[:, 1]
  top_k = np.argsort(scores)[-k:]
  return y[top_k].mean()

def recall_at_k_scorer(estimator, X, y, k=20):
  scores = estimator.predict_proba(X)[:, 1]
  top_k = np.argsort(scores)[-k:]
  return y[top_k].sum()/y.sum()

# l1_ratio = 0 is l2_norm, l1_ratio = 1 is l1_norm, 0<l1_ratio>1 is elastic net
model = LogisticRegression(l1_ratio=0.0, fit_intercept=True, class_weight='balanced', solver='lbfgs', C=1e2, max_iter=10000) 

## Standardize PGFE; mean=-53, std=35
all_df['nPGFE'] = (all_df['PGFE'] + 53)/35
X = all_df[x_features].to_numpy()
# Standardize all (doesn't seem totally necessary - model very similar except for PGFE score) #X = (X - X.mean(axis=0))/X.std(axis=0) 
y = all_df['expt'].to_numpy()
systems = all_df['system'].to_numpy()
cv = GroupKFold(n_splits=4, shuffle=True, random_state=10101)
#cv = GroupKFold(n_splits=2, shuffle=True)#, random_state=10101)

if not args.final:
  if args.rfe:
    # RFE CV to select features:
    # and plot feature elimination curve
    rfecv = RFECV(estimator=model, step=1, cv=cv, scoring=lambda est, X, y: recall_at_k_scorer(est, X, y, k=args.ktop), n_jobs=-1)
    rfecv.fit(X, y, groups=systems)
    selected = np.array(x_features)[rfecv.support_]
    print('features selected:', selected)
    data = { key: value for key, value in rfecv.cv_results_.items() if key in ["n_features", "mean_test_score", "std_test_score"] }
    cv_results = pd.DataFrame(data)
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Mean test accuracy")
    plt.errorbar(x=cv_results["n_features"], y=cv_results["mean_test_score"], yerr=cv_results["std_test_score"])
  else:
    selected = ['nPGFE','aro_count', 'acc_count', 'don_count', 'ani_count', 'cat_count', 'max_dist']

  # Fit final model
  X = all_df[selected].to_numpy()
  scores = cross_val_score(model, X, y, groups=systems, cv=cv, scoring=lambda est, X, y: recall_at_k_scorer(est, X, y, k=args.ktop))
  print(f'Fitting results CV spilt and mean recall@top{args.ktop}:')
  print(scores, f'{scores.mean():.2f}','\n')
  model.fit(X, y)
  intercept = model.intercept_[0]
  coef = model.coef_[0]

  # Rounding model
  factor=args.factor # round to nearest 1/factor
  model.coef_[0] = np.round(factor*coef, 0)/factor
  model.intercept_[0] = np.round(factor*intercept,0)/factor
  all_df['score'] = model.predict_proba(X)[:, 1]

  intercept = model.intercept_[0]
  coef = model.coef_[0]

if args.final:
  selected = ['nPGFE','aro_count', 'acc_count', 'don_count', 'ani_count', 'cat_count', 'max_dist']
  X = all_df[selected].to_numpy()
  coef = np.array([-2.00, 4.00, 6.00, -5.00, -4.00, 7.00, -2.00])
  intercept = -1
  all_df['score'] = expit(X @ coef + intercept)

for ni,ci in zip(x_features, coef):
  print(f"{ni}:{ci:.2f}")
print(f'Intercept: {intercept:.2f}')

## 
## Construct what is kept
#####

## remove redundancy
all_df = ( all_df .groupby('system', group_keys=True) .apply(ph4gen.select_top_k, k=args.ktop, prefer=args.prefer))
all_df = all_df.reset_index()

## no redundancy removal scheme
#all_df['keep'] = (all_df.groupby('system')['score'].rank(method='first', ascending=False).le(args.ktop))

#all_df.sort_values(by=['keep', 'system', 'score'], ascending=[False, True, False], inplace=True)

all_df = all_df.sort_values(["system", "score"], ascending=[True, False])
all_df["rank"] = all_df.groupby("system").cumcount() + 1
all_df["01rank"] = ( (all_df["rank"] - 1) / (all_df.groupby("system")["rank"].transform("max") - 1))

pd.set_option('display.max_rows', 1000)
#print(all_df[(all_df['keep']) & (all_df['system']=='bace')][['system','score','hypothesis','expt','expt_support']+selected])
#print(all_df[(all_df['keep'])][['system','score','hypothesis_set','expt','expt_support']+selected])
#exit()

##############################
# Plot Score Ranking
##############################

fig, (ax0,ax1) = plt.subplots(1,2, figsize=(8,4))

for i,sys in enumerate(systems_list):
  sys_data = all_df[(all_df['system'] == sys)]
  hits  = sys_data[( sys_data['expt'])]
  if hits.shape[0] > 0:
    nohit = sys_data[(~sys_data['expt'])]
    color = c[i%len(c)]
    if args.train_val == 'validation': color = c[(i+len_training)%len(c)]
    ax0.plot(sys_data['01rank'], sys_data['score'], color=color, zorder=1)#, label=pretty_system[i])
 
    ax0.scatter(hits['01rank'], hits['score'], color=color, label=f"{pretty_system[i]} ({hits.shape[0]}, {nohit.shape[0]})", zorder=10)
 
ax0.legend(fontsize=9)
ax0.grid()
ax0.set_ylabel('Score', fontweight='bold')
ax0.set_xlabel('Hypothesis Rank', fontweight='bold')

##############################
# Precision/Recall
##############################

def prcurve(data, label, ax, c):
  precision, recall, _ = precision_recall_curve(data['expt'], data['score'])
  area_under_curve=auc(recall, precision)
  ax1.plot(recall, precision, label=f'{label} (%.3f)'%area_under_curve, color=c)

prcurve(all_df, 'Score', ax1, 'k')
for i,sys in enumerate(systems_list):
  data = all_df[all_df['system'] == sys]
  color = c[i%len(c)]
  if args.train_val =='validation': color = c[(i+len_training)%len(c)]
  if data.shape[0] > 0: prcurve(data, pretty_system[i], ax1, color)

if all_df['keep'].sum() > 10: 
  all_ratio = all_df['expt'].sum()/all_df.shape[0]
  ax1.plot([0,1],[all_ratio,all_ratio], linestyle='--', linewidth=2, color='r', label='Data (%.3f)'%all_ratio)
ax1.legend(fontsize=9)
ax1.set_ylabel('Precision', fontweight='bold')
ax1.set_xlabel('Recall', fontweight='bold')
ax1.grid()
ax1.set_ylim([-0.05,1.05])
plt.tight_layout()

##############
# Summary, N. hypotheses bar plot
##############

if args.expt: all_df = all_df[all_df['expt']]

summary = (
  all_df
    .groupby(['system', 'keep', 'expt'])
    .size()
    .unstack(['keep', 'expt'], fill_value=0)
    .reindex(
      columns=pd.MultiIndex.from_product(
        [[False, True], [False, True]],
        names=['keep', 'expt']
      ),
      fill_value=0
    )
)
tp = summary[(True, True)]
fp = summary[(True, False)]
fn = summary[(False, True)]

summary['Precision'] = tp / (tp + fp)
summary['Recall'] = tp / (tp + fn)
print('***'*5)
summary.reset_index(inplace=True)

# drop column name levels entirely
summary.columns = ['system', 'TN', 'FN', 'FP', 'TP', 'Precision', 'Recall']

# add the minimum rank (best) of the expt=true in score-based ranking
min_rank = (
  all_df[all_df["expt"] == True]
  .groupby("system")["rank"]
  .min()
  .reset_index()
  .rename(columns={"rank": "min_expt_rank"})
)
summary = summary.merge(min_rank, on="system", how="left")
#if args.train_val == 'training':
#  summary = summary.merge(ligands_results[['system','n_match','n','Ligand Recall']], on='system')
pd.set_option('display.precision',3)
print(summary)

##############################
# Bar Plots
##############################

letter=['A','B','C','D','E','F','G']
# Feature counts
count_cols = ['aro_count','don_count','acc_count','cat_count','ani_count']
label2 = ['Aromatic or Hydrophobic', 'H-bond Donor','H-bond Acceptor','Cation','Anion']
fig2, axs2 = plt.subplots(len(label2),1, sharex=True,sharey=True, figsize=(6,9))

# Distance metrics
metric_cols = ['min_dist','max_dist','mean_dist', 'centroid_dist', 'mean_separation']
label3 = [r'Min. Dist.', r'Max. Dist.', r'Mean Dist.', 'Cen. Dist.', r'Mean Feature Sep.']
fig3, axs3 = plt.subplots(len(label3),1, sharex=True, figsize=(6,9))

# PGFE
fig4, axs4 = plt.subplots(1,1)  

groups = [
  ("All",    np.ones(len(all_df), dtype=bool), 1.00, c[0]),
  ("Expt",   all_df["expt"],                   1.00, c[1]),
  ("Top 20", all_df["keep"],                   1.00, c[2]),
]

# Loop over both
for group_name, mask, alpha, color in groups:

  # feature counts
  for i, col in enumerate(count_cols):
    axs2[i].hist(
      all_df.loc[mask, col],
      bins=np.arange(0, 7),
      color=color,
      alpha=alpha,
      log=args.log_flag,
      #histtype="stepfilled",
      linewidth=2,
      histtype="step",
      label=group_name if i == 0 else None,
    )
    axs2[i].grid()
    axs2[i].annotate(letter[i], fontweight='bold', fontsize='14', xy=(-0.10, 0.93), xycoords='axes fraction', zorder=10)

  axs2[0].legend()
  axs2[i].set_xlabel('# Features', fontweight='bold')
  fig2.tight_layout()

  # distances
  for i, col in enumerate(metric_cols):
    axs3[i].hist(
      all_df.loc[mask, col],
      bins=np.arange(0, 12),
      color=color,
      alpha=alpha,
      log=args.log_flag,
      #histtype="stepfilled",
      linewidth=2,
      histtype="step",
      label=group_name if i == 0 else None,
    )
    axs3[i].grid()
    axs3[i].annotate(letter[i], fontweight='bold', fontsize='14', xy=(-0.10, 0.93), xycoords='axes fraction', zorder=10)
  axs3[0].legend()
  axs3[i].set_xlabel(r'Distance $\bf{(\AA)}$', fontweight='bold')
  fig3.tight_layout()
  
  # PGFE
  axs4.hist(
  all_df.loc[mask, 'PGFE'],
  color=color,
  alpha=alpha,
  log=args.log_flag,
  histtype="stepfilled",
  linewidth=2,
  #histtype="step",
  label=group_name,
  )
  axs4.legend()
  axs4.grid()
  axs4.set_xlabel('PGFE', fontweight='bold')

##################
# Final Model histograms
fig5_cols = [ 'nPGFE', 'aro_count', 'don_count', 'acc_count', 'cat_count', 'ani_count', 'max_dist' ]

fig5 = plt.figure(figsize=(6, 9))

# Create an outer GridSpec with three sections:
#   1. Top histogram (Norm. PGFE)
#   2. Middle block of five count histograms
#   3. Bottom histogram (Max. Dist.)
outer = fig5.add_gridspec(nrows=5, ncols=1, height_ratios=[1, 0.35, 5, 0.35, 1], hspace=0.05)

# Create axes
# Top axis (independent x-axis)
ax_top = fig5.add_subplot(outer[0])

# Middle block
middle = outer[2].subgridspec(nrows=5, ncols=1, hspace=0.02)

# First middle axis
axs_mid = [fig5.add_subplot(middle[0])]

# Remaining middle axes share the same x-axis
for i in range(1, 5):
  axs_mid.append(fig5.add_subplot(middle[i], sharex=axs_mid[0]))

# Bottom axis (independent x-axis)
ax_bottom = fig5.add_subplot(outer[4])

# Collect all axes in the same order as fig5_cols
axs5 = [ ax_top, *axs_mid, ax_bottom ]

# Plot histograms
for group_name, mask, alpha, color in groups:
  for i, col in enumerate(fig5_cols):
    # Use integer bins for the count descriptors
    bins = np.arange(0, 7) if "count" in col else None
    axs5[i].hist(
      all_df.loc[mask, col],
      bins=bins,
      color=color,
      alpha=alpha,
      log=args.log_flag,
      histtype="step",
      linewidth=2,
      label=group_name if i == 0 else None
    )

    axs5[i].grid()

    # Panel letter
    axs5[i].annotate(
      letter[i],
      xy=(-0.08, 0.93),
      xycoords="axes fraction",
      fontsize=14,
      fontweight="bold"
    )

# Formatting
# Only the bottom axis of the shared block gets x tick labels
limits=[1+0.1,10**5-0.1]
for ax in axs_mid[:-1]:
  ax.tick_params(labelbottom=False)
  ax.set_ylim(limits)
axs_mid[-1].set_ylim(limits)
ax_bottom.set_ylim(limits)

# Axis labels
ax_top.set_xlabel("Normalized PGFE", fontweight='bold', labelpad=1)
axs_mid[-1].set_xlabel("Feature Count", fontweight='bold', labelpad=1)
ax_bottom.set_xlabel("Maximum Distance (Å)", fontweight='bold', labelpad=1)

# Legend only once
ax_top.set_ylim(limits)
ax_top.legend(loc='best')

# Optional close figures to clean up
#plt.close(fig)
plt.close(fig2)
plt.close(fig3)
plt.close(fig4)
plt.close(fig5)

if not args.quiet: plt.show()
