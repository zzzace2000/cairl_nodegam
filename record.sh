


# Under batch-apprenticeship-learning/
python main.py mountaincar --model_id mma.2 --e_filename demo.deterministic.expert.npz

# Annoying. Not runable :(

python sepsis_sim_learnmdp.py --mdp gam
python sepsis_sim_learnmdp.py --mdp linear

python sepsis_sim.py --mdp gam
python sepsis_sim.py --mdp linear

python sepsis_sim.py --mdp gam --fold 0


python sepsis_expert_gen.py --mdp linear --fold 0
python sepsis_expert_gen.py --mdp gam --fold 0

python main.py --mdp gam
python main.py --mdp linear --opt_method max_margin
python main.py --mdp linear --opt_method projection

# cache expert data
for gamma in '0.9' '0.99'; do
  for mdp in 'linear' 'gam'; do
    python sepsis_expert_gen.py --mdp ${mdp} --gamma ${gamma} &
  done
done

# Change to uniform policy at first!
#pi_init='uniform' # vws23
pi_init='bc' # vws39
for gamma in '0.9' '0.99'; do
  opt_method='max_margin'
  mdp='gam'
  name="0704_${pi_init}_${mdp}_${opt_method}_g${gamma}"
  python -u main.py --name ${name} --pi_init ${pi_init} --mdp ${mdp} --gamma ${gamma} --opt_method ${opt_method} &> logs/${name} &
  mdp='linear'
  for opt_method in 'max_margin' 'projection'; do
    name="0704_${pi_init}_${mdp}_${opt_method}_g${gamma}"
    python -u main.py --name ${name} --pi_init ${pi_init} --mdp ${mdp} --gamma ${gamma} --opt_method ${opt_method} &> logs/${name} &
  done
done


for pi_init in 'bc' 'uniform'; do
  for gamma in '0.9' '0.99'; do
    mdp='gam'
    for opt_method in 'projection'; do
      name="0704_${pi_init}_${mdp}_${opt_method}_g${gamma}"
      python -u main.py --name ${name} --pi_init ${pi_init} --mdp ${mdp} --gamma ${gamma} --opt_method ${opt_method} &> logs/${name} &
    done
  done
done


for gamma in '0.9' '0.99'; do
  for mdp in 'linear' 'gam'; do
    python sepsis_expert_gen.py --mdp ${mdp} --gamma ${gamma} &
  done
done

#gamma='0.9'
#gamma='0.99'
for gamma in '0.9' '0.99'; do
for pi_init in 'uniform'; do
  for mdp in 'gam' 'linear'; do
    for opt_method in 'projection' 'max_margin'; do
      name="0707_${pi_init}_${mdp}_${opt_method}_g${gamma}"
      python -u run_mma.py --name ${name} --pi_init ${pi_init} --mdp ${mdp} --gamma ${gamma} --opt_method ${opt_method} &> logs/${name} &
    done
  done
done
done

# Now we have val/test splits! Store to mma_new2.csv
gamma='0.9'
pi_init='uniform'
opt_method='max_margin'
for mdp in 'gam' 'linear'; do
  name="0708_${pi_init}_${mdp}_${opt_method}_g${gamma}"
  python -u run_mma.py --name ${name} --pi_init ${pi_init} --mdp ${mdp} --gamma ${gamma} --opt_method ${opt_method} --n_iter 30 &> logs/${name} &
done


python -u run_airl.py --name 0708 --epochs 100

# Add a val/test set for the a_match metrics
# Record the hyperparameters for lightning: csvrecording...

python -u run_airl.py --name 0710 --random_search 15 --epochs 50

python -u run_airl.py --name 0714 --random_search 15 --epochs 50

#
python -u run_airl.py --name 0715_linear --random_search 15 --epochs 50 --arch AIRLLightning

# NODE GAM hyperparam search
python -u run_airl.py --name 0715_linear_node --random_search 30 --epochs 50 --mdp linear --arch AIRL_NODEGAM_Lightning

for expert_pol in 'eps0.07' 'eps0.14'; do
  python -u run_airl.py --name 0715_linear_${expert_pol} --random_search 15 --epochs 50 --expert_pol ${expert_pol}
  python -u run_airl.py --name 0715_linear_node_${expert_pol} --random_search 30 --epochs 50 --mdp linear --arch AIRL_NODEGAM_Lightning --expert_pol ${expert_pol}
done

# RERUN LINEAR: it's wrong lol
python -u run_airl.py --name 0716_linear --random_search 15 --epochs 50 --arch AIRLLightning
for expert_pol in 'eps0.07' 'eps0.14'; do
  python -u run_airl.py --name 0716_linear_${expert_pol} --random_search 15 --epochs 50 --expert_pol ${expert_pol} --arch AIRLLightning
done


#### Since we do normalization, retrain! Run two environments and see which can succeed
python -u run_airl.py --name 0717_lmdp_nodegam --random_search 30 --epochs 50 --mdp linear --arch AIRL_NODEGAM_Lightning
python -u run_airl.py --name 0717_gammdp_nodegam --random_search 30 --epochs 50 --mdp gam --arch AIRL_NODEGAM_Lightning
python -u run_airl.py --name 0717_lmdp_linear --random_search 20 --epochs 50 --mdp linear --arch AIRLLightning
python -u run_airl.py --name 0717_gammdp_linear --random_search 20 --epochs 50 --mdp gam --arch AIRLLightning

## Run eps0.07 with linear....
#for expert_pol in 'eps0.07' 'eps0.14'; do
for expert_pol in 'eps0.07'; do
  python -u run_airl.py --name 0718_lmdp_nodegam_${expert_pol} --random_search 30 --epochs 50 --mdp linear --arch AIRL_NODEGAM_Lightning --expert_pol ${expert_pol}
  python -u run_airl.py --name 0718_gammdp_nodegam_${expert_pol} --random_search 30 --epochs 50 --mdp gam --arch AIRL_NODEGAM_Lightning --expert_pol ${expert_pol}
  python -u run_airl.py --name 0718_lmdp_linear_${expert_pol} --random_search 20 --epochs 50 --mdp linear --arch AIRLLightning --expert_pol ${expert_pol}
  python -u run_airl.py --name 0718_gammdp_linear_${expert_pol} --random_search 20 --epochs 50 --mdp gam --arch AIRLLightning --expert_pol ${expert_pol}
done


## Regenerate reward and expert since having new definition of reward
for gamma in '0.9' '0.99'; do
  for mdp in 'linear' 'gam'; do
    python sepsis_expert_gen.py --mdp ${mdp} --gamma ${gamma} &
  done
done


# (R) the baseline apprenticeship learning
gamma='0.9'
pi_init='uniform'
for mdp in 'gam' 'linear'; do
  for opt_method in 'projection' 'max_margin'; do
    name="0719_${mdp}mdp_${pi_init}_${opt_method}_g${gamma}"
    ./my_sbatch -p cpu --gpu 0 --cpu 8 python -u run_mma.py --name ${name} --pi_init ${pi_init} --mdp ${mdp} --gamma ${gamma} --opt_method ${opt_method}
  done
done

# (R) What if the gamma chosen is different? Choose the best hparams in linear
# Choose based on best val_a
#
# 0719_gammdp_nodegam_s321_lr0.0005_bs512_ud30_dn0.2_dns10_nl3_nt20_ad0_td2_od0.0_ld0.0_cs1.0_an3000
for model_gamma in '0.5' '0.8' '0.99'; do
  load_from_hparams='0719_gammdp_nodegam_s321_lr0.0005_bs512_ud30_dn0.2_dns10_nl3_nt20_ad0_td2_od0.0_ld0.0_cs1.0_an3000'
  postfix=${load_from_hparams:4}
  ./my_sbatch python -u run_airl.py --name 0720_mg${model_gamma}${postfix} --model_gamma ${model_gamma} --load_from_hparams ${load_from_hparams}
done

## (R) taking the best epoch with later metric...
## Rerun with val_a_matched as stochastic matching!
gamma='0.9'
pi_init='uniform'
for mdp in 'gam' 'linear'; do
  for opt_method in 'max_margin'; do
    name="0720_${mdp}mdp_${pi_init}_${opt_method}_g${gamma}"
    ./my_sbatch -p cpu --gpu 0 --cpu 8 python -u run_mma.py --name ${name} --pi_init ${pi_init} --mdp ${mdp} --gamma ${gamma} --opt_method ${opt_method}
  done
done



## Run the transition model random search!
python -u main.py --name 0729_gru --random_search 1 --epochs 80 --arch HypotensionGRULightning

# Make sure it runs over
python -u main.py --name 0730_gru --random_search 30 --epochs 200 --arch HypotensionGRULightning


python -u main.py --name 0731_gru --random_search 50 --epochs 100 --arch HypotensionGRULightning

# Design new range and fix val/test eval error. Trust this number
python -u main.py --name 0732_gru --random_search 50 --epochs 100 --arch HypotensionGRULightning
python -u main.py --name 0732_lr --random_search 10 --epochs 100 --arch HypotensionLRLightning


# What to do?
#(R) (1) Run the new best for the IRL NODE-GAM. (0.9 is the best)
for model_gamma in '0.5' '0.8' '0.9' '0.99'; do
  load_from_hparams='0719_gammdp_nodegam_s321_lr0.0005_bs512_ud30_dn0.2_dns10_nl3_nt20_ad0_td2_od0.0_ld0.0_cs1.0_an3000'
  postfix=${load_from_hparams:4}
  ./my_sbatch python -u main.py --name 0730_mg${model_gamma}${postfix} --model_gamma ${model_gamma} --load_from_hparams ${load_from_hparams}
done
# ReRun the best linear model IRL
'0719_lmdp_linear_s321_lr0.001_bs512_ud30_dn0.05_dns20' \
'0719_gammdp_linear_s321_lr0.0003_bs512_ud20_dn0.0_dns10' \
for d in \
'0717_lmdp_nodegam_s321_lr0.001_bs512_ud50_dn0.1_dns20_nl3_nt100_ad0_td3_od0.0_ld0.0_cs0.5_an5000' \
; do
  postfix=${d:4}
  echo ${postfix}
  ./my_sbatch python -u main.py --name 0730${postfix} --load_from_hparams ${d} --epochs 100
done


# (R) for more gamma
for model_gamma in '0' '0.1' '0.3'; do
  load_from_hparams='0719_gammdp_nodegam_s321_lr0.0005_bs512_ud30_dn0.2_dns10_nl3_nt20_ad0_td2_od0.0_ld0.0_cs1.0_an3000'
  postfix=${load_from_hparams:4}
  ./my_sbatch python -u main.py --name 0730_mg${model_gamma}${postfix} --model_gamma ${model_gamma} --load_from_hparams ${load_from_hparams}
done
# (R) Run again for more epochs for NODE-GAM on linear MDP
for d in \
'0717_lmdp_nodegam_s321_lr0.001_bs512_ud50_dn0.1_dns20_nl3_nt100_ad0_td3_od0.0_ld0.0_cs0.5_an5000' \
; do
  postfix=${d:4}
  echo ${postfix}
  ./my_sbatch python -u main.py --name 0730${postfix} --load_from_hparams ${d} --epochs 100
done
# (R) run for gammdp with shaping
load_from_hparams='0719_gammdp_nodegam_s321_lr0.0005_bs512_ud30_dn0.2_dns10_nl3_nt20_ad0_td2_od0.0_ld0.0_cs1.0_an3000'
postfix=${load_from_hparams:4}
./my_sbatch python -u main.py --name 0730_shaping${postfix} --load_from_hparams ${load_from_hparams} --shaping 1

# (R) run for new transition model that has indicator variables.
# TODO: do the teacher forcing
python -u main.py --name 0732_gru --random_search 50 --epochs 100 --arch HypotensionGRULightning
python -u main.py --name 0732_lr --random_search 10 --epochs 100 --arch HypotensionLRLightning




# random search for AIRL with shaping? Would that change? Yes!
python -u main.py --name 0804_lmdp_nodegam --random_search 50 --epochs 100 --mdp linear --arch AIRL_NODEGAM_Lightning

# Do a random search on gru with new hparams. To see:
# (1) Change to student forcing: is it better? Should I search the annealing?
# (2) tf_epochs
# (3) Alpha for balance between reg loss and ind loss
python -u main.py --name 0811_gru --random_search 50 --epochs 100 --arch HypotensionGRULightning

# Redo random search
python -u main.py --name 0813_gru --random_search 50 --epochs 200 --arch HypotensionGRULightning

# Debug: if the for loop same as the common gru?
python -u main.py --name 0813_gru --random_search 50 --epochs 500 --arch HypotensionGRULightning

# Retrain since we add current action and action embeddings!
python -u main.py --name 0825_gru --random_search 50 --epochs 500 --arch HypotensionGRULightning

python -u main.py --name 0825_gru --random_search 50 --epochs 300 --arch HypotensionGRULightning

# Do another retraining!
python -u main.py --name 0825_gru --random_search 50 --epochs 300 --arch HypotensionGRULightning


## Run the AIRL for the sepsis and the new MIMIC3 (!!)
python -u main.py --name 0826_mimic3 --random_search 40 --epochs 100 --arch AIRL_MIMIC3_NODEGAM_Lightning
python -u main.py --name 0826_sim --random_search 40 --epochs 100 --arch AIRL_NODEGAM_Lightning
## (TORUN) if those two above succeed, then run lots of things!

## (RUNNING) search for more mimic3 and simulation7u k7
python -u main.py --name 0827_mimic3 --random_search 100 --epochs 100 --arch AIRL_MIMIC3_NODEGAM_Lightning
python -u main.py --name 0826_sim --random_search 40 --epochs 100 --arch AIRL_NODEGAM_Lightning --mdp gam

## (Search more for ga2m and weight decay etc!!!)
python -u main.py --name 0828_mimic3 --random_search 5 --epochs 150 --arch AIRL_MIMIC3_NODEGAM_Lightning

python -u main.py --name 0828_mimic3 --random_search 5 --epochs 150 --arch AIRL_MIMIC3_NODEGAM_Lightning
python -u main.py --name 0828_mimic3 --random_search 95 --epochs 150 --arch AIRL_MIMIC3_NODEGAM_Lightning

## Update the new data!
python -u main.py --name 0829_mimic3 --random_search 5 --epochs 150 --arch AIRL_MIMIC3_NODEGAM_Lightning
python -u main.py --name 0829_mimic3 --random_search 100 --epochs 150 --arch AIRL_MIMIC3_NODEGAM_Lightning

python -u main.py --name 0830_mimic3 --random_search 100 --epochs 150 --arch AIRL_MIMIC3_NODEGAM_Lightning

## Update the generator: the q value won't get obs noise, but only the reward would get
python -u main.py --name 0831_mimic3 --random_search 100 --epochs 100 --arch AIRL_MIMIC3_NODEGAM_Lightning
python -u main.py --name 0831_mimic3 --random_search 30 --epochs 100 --arch AIRL_MIMIC3_Lightning


## Go to 0901! Several changes to both disc and generator
## (1) Record more metrics in generators and discriminators (balanced acc, and disc acc and loss)
## (2) Add label smoothing
## (3) Adapt DQN as soft-DQN. Also tuning the beta itself
python -u main.py --name 0901_mimic3 --random_search 100 --epochs 180 --arch AIRL_MIMIC3_NODEGAM_Lightning

python -u main.py --name 0902_mimic3 --random_search 60 --epochs 100 --arch AIRL_MIMIC3_NODEGAM_Lightning

python -u main.py --name 0903_gru --random_search 5 --epochs 200 --arch HypotensionGRULightning
# Doing slightly more hparams search
python -u main.py --name 0903_mimic3 --random_search 5 --epochs 200 --arch AIRL_MIMIC3_NODEGAM_Lightning
## Running GRU with gaussian likelihood
python -u main.py --name 0903_gru --random_search 100 --epochs 200 --arch HypotensionGRULightning
# Doing slightly more hparams search
python -u main.py --name 0903_mimic3 --random_search 50 --epochs 200 --arch AIRL_MIMIC3_NODEGAM_Lightning
# Do more search (0903 but with more search)
python -u main.py --name 0903_gru --random_search 60 --epochs 200 --patience 50 --arch HypotensionGRULightning


## 0904 do more search for mimic3!
# (1) Change to using GRU-VAE
# (2) search for sample stdev...
# (3) Change to only using features to model (ignore static and ind features)
# (4) Search if using generated states for q is also good
python -u main.py --name 0904_mimic3 --random_search 10 --epochs 150 --arch AIRL_MIMIC3_NODEGAM_Lightning
python -u main.py --name 0904_mimic3 --random_search 100 --epochs 150 --arch AIRL_MIMIC3_NODEGAM_Lightning

(OK) Should change the disc to remove static variables
  (Not sure) Later should allow NODEGAM to only learn interactinos on cov.
(OK) Also, AIRL objective deteriorates val_a. Maybe should do annealing? At first, it might creates more variance...
(OK) Do 2 preprocessing: 48 hours instead or 4 hours binning instead
  (these 2 reduce the length of patients)

# Train new GRU on this new preprocessing!
python -u main.py --name 0905_gru --random_search 80 --epochs 200 --patience 50 --arch HypotensionGRULightning

python -u main.py --name 0905_mimic3 --random_search 100 --epochs 150 --arch AIRL_MIMIC3_NODEGAM_Lightning

## Use one-step pred for GRU training and simulation.
python -u main.py --name 0906_gru --random_search 100 --epochs 200 --patience 50 --arch HypotensionGRULightning
python -u main.py --name 0906_mimic3 --random_search 100 --epochs 150 --arch AIRL_MIMIC3_NODEGAM_Lightning

## The 0906 mimic3 48 hours preprocessing has too low accuracy (~48%). So try to train on data3 for 72 hours
python -u main.py --name 0907_gru --random_search 100 --epochs 200 --patience 50 --arch HypotensionGRULightning

python -u main.py --name 0907_bc --random_search 90 --epochs 200 --patience 50 --arch BC_MIMIC3_Lightning
python -u main.py --name 0907_bc_mlp --random_search 50 --epochs 150 --patience 50 --arch BC_MIMIC3_MLP_Lightning
python -u main.py --name 0908_bc_mlp --random_search 50 --epochs 150 --patience 50 --arch BC_MIMIC3_MLP_Lightning

python -u main.py --name 0909_bc --random_search 50 --epochs 200 --patience 50 --arch BC_MIMIC3_Lightning
python -u main.py --name 0909_bc_mlp --random_search 50 --epochs 300 --patience 50 --arch BC_MIMIC3_MLP_Lightning

python -u main.py --name 0909_bc_mlp --random_search 50 --epochs 300 --patience 50 --arch BC_MIMIC3_MLP_Lightning


# Keep training:
#
./my_sbatch python -u main.py --name 0907_gru_s129_lr0.0005_wd1e-06_bs256_nh64_nl1_dr0.0_fnh512_fnl2_fdr0.15_anh64_anl2_ano96_adr0.3 --epochs 400 --patience 50 --arch HypotensionGRULightning

0907_gru_s129_lr0.0005_wd1e-06_bs256_nh64_nl1_dr0.0_fnh512_fnl2_fdr0.15_anh64_anl2_ano96_adr0.3

## (TODO) try to add a KL divergence regularization to behavior cloning? Or do annealing, or add a small const
## (TODO) Later do the mimic3 training to see acc... No early stopping
python -u main.py --name 0910_mimic3 --random_search 120 --epochs 120 --patience -1 --arch AIRL_MIMIC3_NODEGAM_Lightning

python -u main.py --name 0911_mimic3 --random_search 120 --epochs 100 --patience 60 --arch AIRL_MIMIC3_NODEGAM_Lightning
python -u main.py --name 0911_mimic3 --random_search 20 --epochs 100 --patience 60 --arch AIRL_MIMIC3_NODEGAM_Lightning

python -u main.py --name 0912_mimic3 --random_search 120 --epochs 100 --patience 60 --arch AIRL_MIMIC3_NODEGAM_Lightning
python -u main.py --name 0914_mimic3 --random_search 120 --epochs 100 --patience 60 --arch AIRL_MIMIC3_NODEGAM_Lightning

## Too many 0 I think, instead I do the quantile_transform. Hopefully it's better?
## Then instead of normal likelihood for gru, we use l2 loss instead.
## Maybe that conforms better with Gaussian quantile.
python -u main.py --name 0915_gru_gauss --random_search 35 --epochs 200 --patience 50 --arch HypotensionGRULightning --obj gaussian
python -u main.py --name 0915_gru_l1 --random_search 35 --epochs 200 --patience 50 --arch HypotensionGRULightning --obj l1

python -u main.py --name 0915_gru_l1_w2 --random_search 5 --epochs 200 --patience 50 --arch HypotensionGRULightning --obj l1 --workers 2


# Run mimic3 with L1
python -u main.py --name 0916_mimic3_l1 --random_search 50 --epochs 100 --patience 60 --arch AIRL_MIMIC3_NODEGAM_Lightning --ts_model_name 0915_gru_l1_s97_lr0.0005_wd1e-05_bs128_nh128_nl1_dr0.0_fnh384_fnl2_fdr0.15_anh128_anl0_ano32_adr0.0
python -u main.py --name 0916_mimic3_g --random_search 50 --epochs 100 --patience 60 --arch AIRL_MIMIC3_NODEGAM_Lightning --ts_model_name 0915_gru_gauss_s53_lr0.001_wd0.0_bs256_nh64_nl1_dr0.0_fnh512_fnl2_fdr0.15_anh64_anl1_ano64_adr0.0

# (NOT SURE) Run another mimic3 without bc regularization. Would it look better!?
## Run mimic3 by using exp_loss with only bc (3)

python -u main.py --name 0917_mimic3_l1_onlybc --random_search 50 --epochs 100 --patience 60 --arch AIRL_MIMIC3_NODEGAM_Lightning3 --ts_model_name 0915_gru_l1_s97_lr0.0005_wd1e-05_bs128_nh128_nl1_dr0.0_fnh384_fnl2_fdr0.15_anh128_anl0_ano32_adr0.0
python -u main.py --name 0917_mimic3_l1 --random_search 50 --epochs 100 --patience 60 --arch AIRL_MIMIC3_NODEGAM_Lightning2 --ts_model_name 0915_gru_l1_s97_lr0.0005_wd1e-05_bs128_nh128_nl1_dr0.0_fnh384_fnl2_fdr0.15_anh128_anl0_ano32_adr0.0

# The graph still looks bad; a lot of fluctuation
# It could be the gru is bad since it has higher training loss?
# Choose another model with lower training_loss?
# Choose gru:
#   0915_gru_l1_s146_lr0.001_wd1e-06_bs256_nh64_nl1_dr0.0_fnh512_fnl2_fdr0.3_anh64_anl2_ano32_adr0.3
ts_model_name='0915_gru_l1_s146_lr0.001_wd1e-06_bs256_nh64_nl1_dr0.0_fnh512_fnl2_fdr0.3_anh64_anl2_ano32_adr0.3'
python -u main.py --name 0918_mimic3_l1_onlybc --random_search 40 --epochs 100 --patience 60 --arch AIRL_MIMIC3_NODEGAM_Lightning3 --ts_model_name ${ts_model_name}
python -u main.py --name 0918_mimic3_l1_exploss --random_search 40 --epochs 100 --patience 60 --arch AIRL_MIMIC3_NODEGAM_Lightning2 --ts_model_name ${ts_model_name}
python -u main.py --name 0918_mimic3_l1_vala --random_search 40 --epochs 100 --patience 60 --arch AIRL_MIMIC3_NODEGAM_Lightning --ts_model_name ${ts_model_name}

# Run a random search without bc!
ts_model_name='0915_gru_l1_s146_lr0.001_wd1e-06_bs256_nh64_nl1_dr0.0_fnh512_fnl2_fdr0.3_anh64_anl2_ano32_adr0.3'
python -u main.py --name 0919_mimic3 --notsave_epochs 10 --random_search 30 --epochs 100 --patience 60 --arch AIRL_MIMIC3_NODEGAM_Lightning --ts_model_name ${ts_model_name}
python -u main.py --name 0919_mimic3_nobc --bc_kl 0 --random_search 40 --epochs 100 --patience 60 --arch AIRL_MIMIC3_NODEGAM_Lightning --ts_model_name ${ts_model_name}

# TODO: Then run a 5-fold cross validation
dir='0918_mimic3_l1_vala_dlr0.001_dwd0.0_shp1_aob1_ane0.1_ls0.0_sa1.0_features_aGAM_nl2_nt500_ad0_td4_od0.1_ld0.3_cs0.7_an3000_la0_ll1_ga2m1_sr1000_bta0.25_fnh512_fnl4_fdr0.5_ubn1_glr0.0004_gwd8e-06_uqbg0.0_sstd0.0_ereg1_bc10_bca0.8_s321_dn0.1_dns0.8_bs64'
for fold in '0' '1' '2' '3' '4'; do
./my_sbatch python -u main.py --name 0920_mimic3_best_f${fold} --fold ${fold} --load_from_hparams ${dir}
done

# Train iptw
python -u main.py --name 0921_gru_l1_iptw --random_search 50 --epochs 200 --patience 50 --arch HypotensionGRULightning --obj l1 --iptw 1

# Use iptw gru to train mimic3
# Choose this since it has 2nd-best test_loss and lower train loss
ts_model_name='0921_gru_l1_iptw_s48_lr0.0005_wd0.0_bs256_nh64_nl1_dr0.0_fnh384_fnl2_fdr0.15_anh64_anl2_ano64_adr0.3'
python -u main.py --name 0922_mimic3 --notsave_epochs 10 --random_search 30 --epochs 100 --arch AIRL_MIMIC3_NODEGAM_Lightning --ts_model_name ${ts_model_name}

ts_model_name='0915_gru_l1_s146_lr0.001_wd1e-06_bs256_nh64_nl1_dr0.0_fnh512_fnl2_fdr0.3_anh64_anl2_ano32_adr0.3'
python -u main.py --name 0923_mimic3 --notsave_epochs 10 --random_search 40 --epochs 100 --arch AIRL_MIMIC3_NODEGAM_Lightning --ts_model_name ${ts_model_name}

## Make things smoother: use smaller depth and layers. Also use IPTW (theoretically more principled Orz)
ts_model_name='0921_gru_l1_iptw_s48_lr0.0005_wd0.0_bs256_nh64_nl1_dr0.0_fnh384_fnl2_fdr0.15_anh64_anl2_ano64_adr0.3'
python -u main.py --name 0924_mimic3 --notsave_epochs 10 --random_search 40 --epochs 100 --patience 50 --arch AIRL_MIMIC3_NODEGAM_Lightning --ts_model_name ${ts_model_name}

## Still jumpy; run 5 fold
dir='0924_mimic3_dlr0.0005_ane0.1_ls0.005_aGAM_nl2_nt200_ad0_td3_od0.1_ld0.5_cs0.5_an3000_la0_ll1_fnl3_fdr0.5_glr0.0004_gwd0.0_uqbg0.5_bc5_bca0.8_s23_dn0.0_dns0.8_bs64'
for fold in '0' '1' '2' '3' '4'; do
./my_sbatch python -u main.py --name 0925_mimic3_best_f${fold} --fold ${fold} --load_from_hparams ${dir}
done

## Improve sync rate; search more slightly
ts_model_name='0921_gru_l1_iptw_s48_lr0.0005_wd0.0_bs256_nh64_nl1_dr0.0_fnh384_fnl2_fdr0.15_anh64_anl2_ano64_adr0.3'
python -u main.py --name 0927_mimic3 --notsave_epochs 10 --random_search 30 --epochs 100 --patience 50 --arch AIRL_MIMIC3_NODEGAM_Lightning --ts_model_name ${ts_model_name}

## Run a sepsis with a new discriminator
python -u main.py --qos deadline --name 0926_sim --random_search 10 --epochs 100 --arch AIRL_NODEGAM_Lightning --mdp cgam
# After getting above, see if GAM recovers correlrations to the wrong feature...

# Run another
python -u main.py --qos deadline --name 0927_sim_cgam --random_search 6 --epochs 100 --arch AIRL_NODEGAM_Lightning --mdp cgam
python -u main.py --qos deadline --name 0927_sim_gam --random_search 6 --epochs 100 --arch AIRL_NODEGAM_Lightning --mdp gam

# Run this for 5 folds again Orz...
dir='0924_mimic3_dlr0.0005_ane0.1_ls0.005_aGAM_nl2_nt200_ad0_td3_od0.1_ld0.5_cs0.5_an3000_la0_ll1_fnl3_fdr0.5_glr0.0004_gwd0.0_uqbg0.5_bc5_bca0.8_s23_dn0.0_dns0.8_bs64'
for fold in '0' '1' '2' '3' '4'; do
./my_sbatch --qos deadline python -u main.py --name 0928_mimic3_best_f${fold} --fold ${fold} --load_from_hparams ${dir}
done


# Run the linear baselines of course
# Modify it to use standard preprocessing that has the
ts_model_name='0921_gru_l1_iptw_s48_lr0.0005_wd0.0_bs256_nh64_nl1_dr0.0_fnh384_fnl2_fdr0.15_anh64_anl2_ano64_adr0.3'
python -u main.py --name 0930_mimic3_linear --notsave_epochs 10 --random_search 30 --epochs 60 --patience 40 --arch AIRL_MIMIC3_Lightning --ts_model_name ${ts_model_name}

# Unclear how to fairly compare across linear models

## Rerun all baselines
python -u main.py --name 1001_lmdp_nodegam --random_search 25 --epochs 100 --patience 50 --mdp linear --arch AIRL_NODEGAM_Lightning
python -u main.py --name 1001_lmdp_linear --random_search 15 --epochs 100 --patience 50  --mdp linear --arch AIRLLightning
python -u main.py --name 1001_gammdp_gam --random_search 25 --epochs 100 --patience 50  --mdp gam --arch AIRL_NODEGAM_Lightning
python -u main.py --name 1001_gammdp_linear --random_search 15 --epochs 100 --patience 50  --mdp gam --arch AIRLLightning

## Get MMA baselines (CIRL)
gamma='0.9'
pi_init='uniform'
for mdp in 'gam' 'linear'; do
    name="1002_${mdp}mdp_${pi_init}_g${gamma}"
    ./my_sbatch -p cpu --gpu 0 --cpu 8 python -u run_mma.py --name ${name} --pi_init ${pi_init} --mdp ${mdp} --gamma ${gamma}
done
## Get MMA w/ current states
gamma='0.9'
pi_init='uniform'
disc_state_type='current'
for mdp in 'gam' 'linear'; do
    name="1003_S${disc_state_type}_${mdp}mdp_${pi_init}_g${gamma}"
    ./my_sbatch -p cpu --gpu 0 --cpu 8 python -u run_mma.py --name ${name} --pi_init ${pi_init} --mdp ${mdp} --gamma ${gamma} --disc_state_type ${disc_state_type}
done

## Run AIRL based on the original states
disc_state_time='current'
python -u main.py --name 1003_lmdp_s${disc_state_time}_gam --random_search 25 --epochs 100 --patience 50 --mdp linear --arch AIRL_NODEGAM_Lightning --disc_state_time ${disc_state_time}
python -u main.py --name 1003_lmdp_s${disc_state_time}_linear --random_search 15 --epochs 100 --patience 50  --mdp linear --arch AIRLLightning --disc_state_time ${disc_state_time}
python -u main.py --name 1003_gammdp_s${disc_state_time}_gam --random_search 25 --epochs 100 --patience 50  --mdp gam --arch AIRL_NODEGAM_Lightning --disc_state_time ${disc_state_time}
python -u main.py --name 1003_gammdp_s${disc_state_time}_linear --random_search 15 --epochs 100 --patience 50  --mdp gam --arch AIRLLightning --disc_state_time ${disc_state_time}

## Do current as well for mimic
ts_model_name='0921_gru_l1_iptw_s48_lr0.0005_wd0.0_bs256_nh64_nl1_dr0.0_fnh384_fnl2_fdr0.15_anh64_anl2_ano64_adr0.3'
disc_state_time='current'
python -u main.py --name 1003_mimic3_s${disc_state_time}_linear --notsave_epochs 10 --random_search 20 --epochs 60 --patience 40 --arch AIRL_MIMIC3_Lightning --ts_model_name ${ts_model_name} --disc_state_time ${disc_state_time}

ts_model_name='0921_gru_l1_iptw_s48_lr0.0005_wd0.0_bs256_nh64_nl1_dr0.0_fnh384_fnl2_fdr0.15_anh64_anl2_ano64_adr0.3'
disc_state_time='current'
python -u main.py --name 1003_mimic3_s${disc_state_time} --notsave_epochs 10 --random_search 30 --epochs 60 --patience 40 --arch AIRL_MIMIC3_NODEGAM_Lightning --ts_model_name ${ts_model_name} --disc_state_time ${disc_state_time}

## TORUN:///
## For Simulation: 1001 seems fine (we can run more on 1001 for linear model?
# Ok update the hyperparameter search; search more in lmdp with linear...
python -u main.py --name 1001_lmdp_linear_new --random_search 20 --epochs 100 --patience 50 --mdp linear --arch AIRLLightning
python -u main.py --name 1001_lmdp_gam_new --random_search 20 --epochs 100 --patience 50 --mdp linear --arch AIRL_NODEGAM_Lightning
python -u main.py --name 1001_gammdp_gam_new --random_search 25 --epochs 100 --patience 50 --mdp gam --arch AIRL_NODEGAM_Lightning

## For current baselines: I update the current to use current state instead of next state!
disc_state_time='current'
python -u main.py --name 1004_lmdp_s${disc_state_time}_linear --random_search 20 --epochs 100 --patience 50  --mdp linear --arch AIRLLightning --disc_state_time ${disc_state_time}
python -u main.py --name 1004_lmdp_s${disc_state_time}_gam --random_search 25 --epochs 100 --patience 50 --mdp linear --arch AIRL_NODEGAM_Lightning --disc_state_time ${disc_state_time}
python -u main.py --name 1004_gammdp_s${disc_state_time}_gam --random_search 25 --epochs 100 --patience 50 --mdp gam --arch AIRL_NODEGAM_Lightning --disc_state_time ${disc_state_time}


## For real data: rerun everything I feel...
# Linear in MIMIC3
ts_model_name='0921_gru_l1_iptw_s48_lr0.0005_wd0.0_bs256_nh64_nl1_dr0.0_fnh384_fnl2_fdr0.15_anh64_anl2_ano64_adr0.3'
python -u main.py --name 1005_mimic3_linear --notsave_epochs 20 --random_search 30 --epochs 100 --patience 40 --arch AIRL_MIMIC3_Lightning --ts_model_name ${ts_model_name} --bc_kl_anneal 0.2 # So only save epochs after no kl reg is used
ts_model_name='0921_gru_l1_iptw_s48_lr0.0005_wd0.0_bs256_nh64_nl1_dr0.0_fnh384_fnl2_fdr0.15_anh64_anl2_ano64_adr0.3'
python -u main.py --name 1005_mimic3_gam --notsave_epochs 20 --random_search 40 --epochs 100 --patience 50 --arch AIRL_MIMIC3_NODEGAM_Lightning --ts_model_name ${ts_model_name} --bc_kl_anneal 0.2 # So only save epochs after no kl reg is used

# Do the current thing
disc_state_time='current'
ts_model_name='0921_gru_l1_iptw_s48_lr0.0005_wd0.0_bs256_nh64_nl1_dr0.0_fnh384_fnl2_fdr0.15_anh64_anl2_ano64_adr0.3'
python -u main.py --name 1006_mimic3_linear_s${disc_state_time} --notsave_epochs 20 --random_search 30 --epochs 100 --patience 40 --arch AIRL_MIMIC3_Lightning --ts_model_name ${ts_model_name} --bc_kl_anneal 0.2 --disc_state_time ${disc_state_time} # So only save epochs after no kl reg is used
ts_model_name='0921_gru_l1_iptw_s48_lr0.0005_wd0.0_bs256_nh64_nl1_dr0.0_fnh384_fnl2_fdr0.15_anh64_anl2_ano64_adr0.3'
python -u main.py --name 1006_mimic3_gam_s${disc_state_time} --notsave_epochs 20 --random_search 40 --epochs 100 --patience 50 --arch AIRL_MIMIC3_NODEGAM_Lightning --ts_model_name ${ts_model_name} --bc_kl_anneal 0.2 --disc_state_time ${disc_state_time} # So only save epochs after no kl reg is used



#### 1006: Ok my idea fails. Let me run mimic3 with bc all along (no decay). Run a version of 1 100 (just to see if in extreme what perf would I get)
# (1) Run more in 1001_lmdp_linear_new
python -u main.py --name 1001_lmdp_linear_new --random_search 20 --epochs 100 --patience 50 --mdp linear --arch AIRLLightning


## First, run a version of 100 bc_kl. Just want to see the performance of behavior cloning
ts_model_name='0921_gru_l1_iptw_s48_lr0.0005_wd0.0_bs256_nh64_nl1_dr0.0_fnh384_fnl2_fdr0.15_anh64_anl2_ano64_adr0.3'
python -u main.py --name 1007_mimic3_gam --notsave_epochs 10 --random_search 5 --epochs 100 --patience 50 --arch AIRL_MIMIC3_NODEGAM_Lightning --ts_model_name ${ts_model_name} --bc_kl 100
## Then seaarch normally
ts_model_name='0921_gru_l1_iptw_s48_lr0.0005_wd0.0_bs256_nh64_nl1_dr0.0_fnh384_fnl2_fdr0.15_anh64_anl2_ano64_adr0.3'
python -u main.py --name 1007_mimic3_gam --notsave_epochs 10 --random_search 40 --epochs 100 --patience 50 --arch AIRL_MIMIC3_NODEGAM_Lightning --ts_model_name ${ts_model_name}


### 1006 today!!!!!
## Search normally for linear model after updating the hparams range....
ts_model_name='0921_gru_l1_iptw_s48_lr0.0005_wd0.0_bs256_nh64_nl1_dr0.0_fnh384_fnl2_fdr0.15_anh64_anl2_ano64_adr0.3'
python -u main.py --name 1008_mimic3_linear --notsave_epochs 10 --random_search 30 --epochs 100 --patience 50 --arch AIRL_MIMIC3_Lightning --ts_model_name ${ts_model_name}
python -u main.py --name 1008_mimic3_gam --notsave_epochs 10 --random_search 30 --epochs 100 --patience 50 --arch AIRL_MIMIC3_NODEGAM_Lightning --ts_model_name ${ts_model_name}

disc_state_time='current'
ts_model_name='0921_gru_l1_iptw_s48_lr0.0005_wd0.0_bs256_nh64_nl1_dr0.0_fnh384_fnl2_fdr0.15_anh64_anl2_ano64_adr0.3'
python -u main.py --name 1009_mimic3_linear_s${disc_state_time} --notsave_epochs 20 --random_search 30 --epochs 100 --patience 40 --arch AIRL_MIMIC3_Lightning --ts_model_name ${ts_model_name} --disc_state_time ${disc_state_time} # So only save epochs after no kl reg is used
ts_model_name='0921_gru_l1_iptw_s48_lr0.0005_wd0.0_bs256_nh64_nl1_dr0.0_fnh384_fnl2_fdr0.15_anh64_anl2_ano64_adr0.3'
python -u main.py --name 1009_mimic3_gam_s${disc_state_time} --notsave_epochs 20 --random_search 30 --epochs 100 --patience 50 --arch AIRL_MIMIC3_NODEGAM_Lightning --ts_model_name ${ts_model_name} --disc_state_time ${disc_state_time} # So only save epochs after no kl reg is used



#### (2) Run FCNN
python -u main.py --name 1010_lmdp_fcnn --random_search 25 --epochs 100 --patience 50 --mdp linear --arch AIRL_FCNN_Lightning
python -u main.py --name 1010_gammdp_fcnn --random_search 25 --epochs 100 --patience 50 --mdp gam --arch AIRL_FCNN_Lightning

ts_model_name='0921_gru_l1_iptw_s48_lr0.0005_wd0.0_bs256_nh64_nl1_dr0.0_fnh384_fnl2_fdr0.15_anh64_anl2_ano64_adr0.3'
python -u main.py --name 1010_mimic3_fcnn --notsave_epochs 10 --random_search 25 --epochs 100 --patience 50 --arch AIRL_MIMIC3_FCNN_Lightning --ts_model_name ${ts_model_name}
###### 'Current'
disc_state_time='current'
python -u main.py --name 1011_lmdp_fcnn_s${disc_state_time} --random_search 25 --epochs 100 --patience 50 --mdp linear --arch AIRL_FCNN_Lightning --disc_state_time ${disc_state_time}
python -u main.py --name 1011_gammdp_fcnn_s${disc_state_time} --random_search 25 --epochs 100 --patience 50 --mdp gam --arch AIRL_FCNN_Lightning --disc_state_time ${disc_state_time}
ts_model_name='0921_gru_l1_iptw_s48_lr0.0005_wd0.0_bs256_nh64_nl1_dr0.0_fnh384_fnl2_fdr0.15_anh64_anl2_ano64_adr0.3'
python -u main.py --name 1011_mimic3_fcnn_s${disc_state_time} --notsave_epochs 20 --random_search 25 --epochs 100 --patience 50 --arch AIRL_MIMIC3_FCNN_Lightning --ts_model_name ${ts_model_name} --disc_state_time ${disc_state_time} # So only save epochs after no kl reg is used


#### (1) Add a 0.5 baseline?
python -u sepsis_expert_gen.py --mdp gam --gamma 0.5
python -u sepsis_expert_gen.py --mdp linear --gamma 0.5
#### Run simulations
gamma='0.5'
python -u main.py --name 1012_lmdp_linear_g${gamma} --random_search 25 --epochs 100 --patience 50 --mdp linear --arch AIRLLightning --gamma ${gamma}
python -u main.py --name 1012_gammdp_linear_g${gamma} --random_search 25 --epochs 100 --patience 50 --mdp gam --arch AIRLLightning --gamma ${gamma}
python -u main.py --name 1012_lmdp_gam_g${gamma} --random_search 20 --epochs 100 --patience 50 --mdp linear --arch AIRL_NODEGAM_Lightning --gamma ${gamma}
python -u main.py --name 1012_gammdp_gam_g${gamma} --random_search 20 --epochs 100 --patience 50 --mdp gam --arch AIRL_NODEGAM_Lightning --gamma ${gamma}
python -u main.py --name 1012_lmdp_fcnn_g${gamma} --random_search 25 --epochs 100 --patience 50 --mdp linear --arch AIRL_FCNN_Lightning --gamma ${gamma}
python -u main.py --name 1012_gammdp_fcnn_g${gamma} --random_search 25 --epochs 100 --patience 50 --mdp gam --arch AIRL_FCNN_Lightning --gamma ${gamma}

pi_init='uniform'
for mdp in 'gam' 'linear'; do
    name="1012_${mdp}mdp_${pi_init}_g${gamma}"
    ./my_sbatch -p cpu --gpu 0 --cpu 8 python -u run_mma.py --name ${name} --pi_init ${pi_init} --mdp ${mdp} --gamma ${gamma}
done
## Get MMA w/ current states
pi_init='uniform'
disc_state_type='current'
for mdp in 'gam' 'linear'; do0915_
    name="1012_S${disc_state_type}_${mdp}mdp_${pi_init}_g${gamma}"
    ./my_sbatch -p cpu --gpu 0 --cpu 8 python -u run_mma.py --name ${name} --pi_init ${pi_init} --mdp ${mdp} --gamma ${gamma} --disc_state_type ${disc_state_type}
done

## Run current with gamma=0.5
gamma='0.5'
disc_state_time='current'
python -u main.py --name 1013_lmdp_fcnn_s${disc_state_time}_g${gamma} --random_search 25 --epochs 100 --patience 50 --mdp linear --arch AIRL_FCNN_Lightning --disc_state_time ${disc_state_time} --gamma ${gamma}
python -u main.py --name 1013_gammdp_fcnn_s${disc_state_time}_g${gamma} --random_search 25 --epochs 100 --patience 50 --mdp gam --arch AIRL_FCNN_Lightning --disc_state_time ${disc_state_time} --gamma ${gamma}
python -u main.py --name 1013_lmdp_s${disc_state_time}_linear_g${gamma} --random_search 20 --epochs 100 --patience 50  --mdp linear --arch AIRLLightning --disc_state_time ${disc_state_time} --gamma ${gamma}
python -u main.py --name 1013_lmdp_s${disc_state_time}_gam_g${gamma} --random_search 25 --epochs 100 --patience 50 --mdp linear --arch AIRL_NODEGAM_Lightning --disc_state_time ${disc_state_time} --gamma ${gamma}
python -u main.py --name 1013_gammdp_s${disc_state_time}_gam_g${gamma} --random_search 25 --epochs 100 --patience 50 --mdp gam --arch AIRL_NODEGAM_Lightning --disc_state_time ${disc_state_time} --gamma ${gamma}


### Run a MIMIC without any bc regularization loss.... Just to see which one performs better
# Ok without bc reg it will fail miserably.
ts_model_name='0921_gru_l1_iptw_s48_lr0.0005_wd0.0_bs256_nh64_nl1_dr0.0_fnh384_fnl2_fdr0.15_anh64_anl2_ano64_adr0.3'
python -u main.py --name 1014_mimic3_linear_nobc --random_search 30 --epochs 100 --patience 50 --arch AIRL_MIMIC3_Lightning --ts_model_name ${ts_model_name} --bc_kl 0
python -u main.py --name 1014_mimic3_gam_nobc --random_search 40 --epochs 100 --patience 50 --arch AIRL_MIMIC3_NODEGAM_Lightning --ts_model_name ${ts_model_name} --bc_kl 0
python -u main.py --name 1014_mimic3_fcnn_nobc --random_search 30 --epochs 100 --patience 50 --arch AIRL_MIMIC3_FCNN_Lightning --ts_model_name ${ts_model_name} --bc_kl 0


### 1007: Run more 1008 mimic3....
ts_model_name='0921_gru_l1_iptw_s48_lr0.0005_wd0.0_bs256_nh64_nl1_dr0.0_fnh384_fnl2_fdr0.15_anh64_anl2_ano64_adr0.3'
python -u main.py --name 1008_mimic3_gam --notsave_epochs 10 --random_search 30 --epochs 100 --patience 50 --arch AIRL_MIMIC3_NODEGAM_Lightning --ts_model_name ${ts_model_name}
python -u main.py --name 1010_mimic3_fcnn --notsave_epochs 10 --random_search 30 --epochs 100 --patience 50 --arch AIRL_MIMIC3_FCNN_Lightning --ts_model_name ${ts_model_name}
disc_state_time='current'
python -u main.py --name 1009_mimic3_gam_s${disc_state_time}_new --notsave_epochs 10 --random_search 30 --epochs 100 --patience 50 --arch AIRL_MIMIC3_NODEGAM_Lightning --ts_model_name ${ts_model_name} --disc_state_time ${disc_state_time}
python -u main.py --name 1011_mimic3_fcnn_s${disc_state_time}_new --notsave_epochs 10 --random_search 30 --epochs 100 --patience 50 --arch AIRL_MIMIC3_FCNN_Lightning --ts_model_name ${ts_model_name} --disc_state_time ${disc_state_time}
python -u main.py --name 1009_mimic3_linear_s${disc_state_time}_new --notsave_epochs 10 --random_search 30 --epochs 100 --patience 50 --arch AIRL_MIMIC3_Lightning --ts_model_name ${ts_model_name} --disc_state_time ${disc_state_time}



### How about gamma=0.5 in MIMIC3?
ts_model_name='0921_gru_l1_iptw_s48_lr0.0005_wd0.0_bs256_nh64_nl1_dr0.0_fnh384_fnl2_fdr0.15_anh64_anl2_ano64_adr0.3'
gamma='0.5'
python -u main.py --name 1014_mimic3_gam_g${gamma} --notsave_epochs 10 --random_search 30 --epochs 100 --patience 50 --arch AIRL_MIMIC3_NODEGAM_Lightning --ts_model_name ${ts_model_name} --gamma ${gamma}
python -u main.py --name 1014_mimic3_linear_g${gamma} --notsave_epochs 10 --random_search 30 --epochs 100 --patience 50 --arch AIRL_MIMIC3_Lightning --ts_model_name ${ts_model_name} --gamma ${gamma}

### How about gamma=0.5 with no bc!?
ts_model_name='0921_gru_l1_iptw_s48_lr0.0005_wd0.0_bs256_nh64_nl1_dr0.0_fnh384_fnl2_fdr0.15_anh64_anl2_ano64_adr0.3'
gamma='0.5'
python -u main.py --name 1015_mimic3_gam_g${gamma}_nobc --notsave_epochs 10 --random_search 30 --epochs 100 --patience 50 --arch AIRL_MIMIC3_NODEGAM_Lightning --ts_model_name ${ts_model_name} --gamma ${gamma} --bc_kl 0


### Find a huge bug in MMA evaluation of reward :( rerun
for gamma in '0.9' '0.5'; do
  pi_init='uniform'
  for mdp in 'gam' 'linear'; do
      name="1015_${mdp}mdp_${pi_init}_g${gamma}"
      ./my_sbatch -p cpu --gpu 0 --cpu 16 python -u run_mma.py --name ${name} --pi_init ${pi_init} --mdp ${mdp} --gamma ${gamma}
  done
  ## Get MMA w/ current states
  pi_init='uniform'
  disc_state_type='current'
  for mdp in 'gam' 'linear'; do
      name="1015_S${disc_state_type}_${mdp}mdp_${pi_init}_g${gamma}"
      ./my_sbatch -p cpu --gpu 0 --cpu 16 python -u run_mma.py --name ${name} --pi_init ${pi_init} --mdp ${mdp} --gamma ${gamma} --disc_state_type ${disc_state_type}
  done
done

gamma='0.9'
pi_init='uniform'
mdp='linear'
name="1015_${mdp}mdp_${pi_init}_g${gamma}"
./my_sbatch -p cpu --gpu 0 --cpu 16 python -u run_mma.py --name ${name} --pi_init ${pi_init} --mdp ${mdp} --gamma ${gamma}


## For current baselines: I update the current to use current state instead of next state!
#disc_state_time='current'
#python -u main.py --name 1004_lmdp_s${disc_state_time}_linear --random_search 20 --epochs 100 --patience 50  --mdp linear --arch AIRLLightning --disc_state_time ${disc_state_time}
#python -u main.py --name 1004_lmdp_s${disc_state_time}_gam --random_search 25 --epochs 100 --patience 50 --mdp linear --arch AIRL_NODEGAM_Lightning --disc_state_time ${disc_state_time}
#python -u main.py --name 1004_gammdp_s${disc_state_time}_gam --random_search 25 --epochs 100 --patience 50 --mdp gam --arch AIRL_NODEGAM_Lightning --disc_state_time ${disc_state_time}

MIMIC3 best(!!)
for d in \
'1008_mimic3_gam_dlr0.0008_aob1_ls0.01_aGAM_nl2_nt200_ad0_td2_ld0.3_cs0.5_an3000_la0_ll1_od0.1_fnl4_fdr0.5_glr0.0008_bca0.5_s97_dn0.0_dns0.8' \
'1008_mimic3_linear_dlr0.0005_aob0_ls0.01_fnl3_fdr0.5_glr0.0008_bca0.5_s165_dn0.1_dns0.8' \
'1009_mimic3_gam_scurrent_dlr0.0005_aob1_ls0.01_aGAM_nl3_nt200_ad0_td2_ld0.5_cs0.5_an3000_la0_ll1_od0.2_fnl3_fdr0.5_glr0.0008_bca0.5_s29_dn0.0_dns0.8' \
'1009_mimic3_linear_scurrent_dlr0.001_aob0_ls0.01_fnl4_fdr0.5_glr0.0004_bca0.5_s130_dn0.1_dns0.8' \
'1010_mimic3_fcnn_dlr0.001_aob0_ls0.005_dnl3_dnh256_ddr0.1_fnl3_fdr0.5_glr0.0008_bca0.5_s22_dn0.0_dns0.8' \
'1011_mimic3_fcnn_scurrent_dlr0.001_aob1_ls0.01_dnl4_dnh32_ddr0.1_fnl4_fdr0.5_glr0.0008_bca0.5_s97_dn0.0_dns0.8' \
; do
   postfix=${d:11}
   echo ${postfix}
  for fold in '0' '1' '2' '3' '4'; do
  ./my_sbatch python -u main.py --name 1015_mimic3_best_f${fold}_${postfix} --fold ${fold} --load_from_hparams ${d}
  done
done

for d in \
'1008_mimic3_gam_dlr0.0005_aob0_ls0.005_aGAM_nl2_nt300_ad0_td2_ld0.3_cs0.5_an3000_la0_ll1_od0.1_fnl3_fdr0.5_glr0.0008_bca0.5_s120_dn0.0_dns0.8' \
; do
   postfix=${d:11}
   echo ${postfix}
  for fold in '0' '1' '2' '3' '4'; do
  ./my_sbatch python -u main.py --name 1016_mimic3_best_f${fold}_${postfix} --fold ${fold} --load_from_hparams ${d}
  done
done




for d in \
'1008_mimic3_linear_dlr0.001_aob0_ls0.005_fnl3_fdr0.5_glr0.0008_bca0.5_s192_dn0.0_dns0.8' \
; do
   postfix=${d:11}
   echo ${postfix}
  for fold in '0' '1' '2' '3' '4'; do
  ./my_sbatch python -u main.py --name 1016_mimic3_best_f${fold}_${postfix} --fold ${fold} --load_from_hparams ${d}
  done
done


## Run linear current
for gamma in '0.5' '0.9'; do
disc_state_time='current'
python -u main.py --name 1013_gammdp_s${disc_state_time}_linear_g${gamma} --random_search 10 --epochs 100 --patience 50  --mdp gam --arch AIRLLightning --disc_state_time ${disc_state_time} --gamma ${gamma}
done


### This should be the best model 1008; not sure why I pick
for d in \
'1008_mimic3_gam_dlr0.0008_aob1_ls0.005_aGAM_nl3_nt200_ad0_td4_ld0.5_cs0.5_an3000_la0_ll1_od0.1_fnl3_fdr0.5_glr0.0008_bca0.5_s190_dn0.1_dns0.8' \
; do
   postfix=${d:11}
   echo ${postfix}
  for fold in '0' '1' '2' '3' '4'; do
  ./my_sbatch python -u main.py --name 1017_mimic3_best_f${fold}_${postfix} --fold ${fold} --load_from_hparams ${d}
  done
done

# Run behavior cloning!

for d in \
'0909_bc_s55_lr0.001_wd1e-05_bs128_nh64_nl1_dr0.0_fnh256_fnl4_fdr0.5' \
; do
   postfix=${d:4}
   echo ${postfix}
  for fold in '0' '1' '2' '3' '4'; do
  ./my_sbatch python -u main.py --name 1018_bc_best_f${fold}_${postfix} --fold ${fold} --load_from_hparams ${d}
  done
done


# Run more random seeds for MMA; don't want to be too presumptious
for seed in '51' '15' '64' '74' '25'; do
  for gamma in '0.9' '0.5'; do
    pi_init='uniform'
    for mdp in 'gam' 'linear'; do
        name="1015_${mdp}mdp_${pi_init}_g${gamma}_s${seed}"
        ./my_sbatch -p cpu --gpu 0 --cpu 16 python -u run_mma.py --name ${name} --pi_init ${pi_init} --mdp ${mdp} --gamma ${gamma} --seed ${seed}
    done
    ## Get MMA w/ current states
    pi_init='uniform'
    disc_state_type='current'
    for mdp in 'gam' 'linear'; do
        name="1015_S${disc_state_type}_${mdp}mdp_${pi_init}_g${gamma}_s${seed}"
        ./my_sbatch -p cpu --gpu 0 --cpu 16 python -u run_mma.py --name ${name} --pi_init ${pi_init} --mdp ${mdp} --gamma ${gamma} --disc_state_type ${disc_state_type} --seed ${seed}
    done
  done
done



#### TODO:
# (2) Maybe compute the distance of MMA & CIRL & Linear-AIRL. (Ok)
# (1.5) Add a pseudo-code of CAIRL? We can remove unnecesary details to appendix...
  - It will be similar to AIRL with phi and GAM. Not sure it worths it. We can put it in the appendix
# (3) Hyperparameter tables: the search range, hyperparameters...
# (4) describe how each model is trained
# (1) Consider describing DSFN, MMA in the related work?


#### Analyzing results
(1) I guess the linear current performs surprisingly better. Their test_a is not good (as expected) but test_r is not bad...
  - Do more search on linear + linear mdp
(2) GAM is not bad

### Come back to the MIMIC





## For sim
python -u main.py --name 1003_lmdp_s${disc_state_time}_gam --random_search 25 --epochs 100 --patience 50 --mdp linear --arch AIRL_NODEGAM_Lightning --disc_state_time ${disc_state_time}
python -u main.py --name 1003_gammdp_s${disc_state_time}_gam --random_search 25 --epochs 100 --patience 50  --mdp gam --arch AIRL_NODEGAM_Lightning --disc_state_time ${disc_state_time}



## After updating that for MIMIC we use only reward (not AIRL obj) to optimize generator
## So update both next and current




## Get baselines of using current state for MMA
- Why? Since it is the original apprenticeship learning and AIRL as well
  - At least in simulation







## TODO: baselines like CIRL and DFSN
  - (Focused!) CIRL just uses transition model to do roll-out and take decayed-average (quite easy)
    - No need bc. Just do the roll-out...
    - Setup a max-margin classifier that seperates both
  - Then train another DQN that will generate feature expectance (discriminator updates...) and do this...
    - BC initialization: in first iteration, disc derives an initial weights that seperate between expert mu and dsfn
      derived from behavior cloning policy


- First, see the 0926_sim and see if it picks up spurious features
  - A notebook
    - Run a mdp solve given this reward and get a policy
    - Then run a policy on a second environment and see reward
    - And maybe do graph editing to remove the reward from GAM and do the same thing


## TODO: ablation study
  - Different objective of the transition model (easy)
  - Different gamma (easy)


## Might need a complete DQN for apprenticeship learning?
## So each time a new weight coming, training a fresh DQN on the recovered reward?

## Improve IL with TRIL? Transition-regularized (multi-task learning?)

# TODO: visualize 0926_sim if it captures spurious correlations





## TODO
(1) Design a toy dataset that has a spurious feature
- First, create a dimension that correlates with reward
- Then learn an agent on top of that
- Then make a gradient reversal layer that removes spurious correlation




- Problem: choosing policy based on val acc will choose policy with early stopping epochs and large penalization...
  (0) Add a counterfactual loss for transition models?

  (1) Run without it for this and our baseline?
  (2) Keep generator as bc (maybe fixed) and choose the best disc? So hopefully if the noise is large enough, the disc learns something
    And that becomes supervised learning objective
      - It could be more stable and easily understandable?
      - I can fix the bc next-state dataset?
        - Or I can keep a ts model and load bc prob to keep generating!



or
# Baslines: DNDT, and behavi cloning, and compare with Bioca CHIL? (Like using Apprenticeship with RNN)
- Behavior cloning
- DNDT: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7233064/


# Maybe implement counterfactual loss?
  # Like it's not too difficult. Just add another reversal layer that representation can not predict action(?)
  # Not sure


## Thoughts
- Maybe for Q-learning only model the 9 actions instead of 16 (the 7 actions only account for 4%...)


- See why MIMIC3 performs so bad? Is it disc loss?
- See if GRU can get a better version and do more random search, and probably train 5 best to get an ensemble and get uncertainty!

## Ablation study
- GRU
  - L1 loss
  - Gaussian likelihood

- Adding Gaussian noise or not (checked)
- Q-learning:
  - Do normal DQN (checked)
  - Do soft-DQN (checked)
  - Do softmax-DQN
- Should we iterate on generate states or just expert states?
- AIRL objective
  - Disc Do / Not / Annealing
  - Gen Do / Not (checked)
- Shaping
  - Do / Not (checked)
- Different beta? (Maybe just put 1) (checked)

Some points observed:
- Generator loss keeps increasing!

## TODO for preprocessing
(1) Add indicator of dead or leavining ICU. This might help the ending state modeling.

## TOREAD
- Finish updating the discriminator part
  (1) Maybe read offline IRL: https://arxiv.org/pdf/2106.05068.pdf
  (2) In GAN, how do we not make discriminator becomes useless?
    Complicated: https://arxiv.org/pdf/1702.01691.pdf
    Then why IRL makes sense? Early stopping?

## TODO for transition model
(1) Train an ensemble of model
- Long term rollouts
(2) When training transition model, also predicting the measurement indicator as well
  - It will help in long-term rollouts, and is more signal...
(2) Train both teacher and student forcing for long-term trajectory roll-outs
  - Since it serves as an environment in IRL training...
(3) May do a propensity weighting, since I would need the behavior cloning anyway

## TODO: Final experiments
(1) What are the usage of these?
  - Explain the expert preference in terms of future data
    - Expert might forget everything they need to specify
    - It is not easy to specify the degree of reward like in DeepMind datacenter cooling project
  - See Finale paper on this
    - Feature importance
    - Shape graph
    - Case study: like treatment at time t, the potential trajectory, and the reward...
  - Limitations
    - It would not work to take the oscilating line. Like that reward is a multi-way interaction and
      can not be captured by a GAM. But main effects might help
  - Intelligible models for utility modeling
  - Probably claim that we can model the preference of doctors about side effects or tradeoffs
    - Claim that understanding normal region, and maybe a bit of descriptive case? (not easy)


Q: how do I model the terminal state?
A: ok ignore it for now. Just run the AIRL and see if this helps? Like having an indicator
of leaving hospital early or dead etc...
- I can put a state as terminal state, and use a next-state predictor for that?
  - A: Do not think that would work too well.
  - Probably a good idea to model that?
- Terminal reward is different from gamma

Theoretical analysis:
- L1 distance to gt is bounded for the culmulative reward difference.


## Thinking:
(1) Do proposenity weighting?
  - Problem: unclear if they increase variance too much
    - And I should not include 3rd covariates that predicts actions but not affecting outcomes
    - But seems easy to be implemented...
      - Can only predict discretized vaso and fluids
  - I can make it multi-task learning? Predict whether next action and next state would happen?



## Done
- (1) The copy-baseline. What is the MSE?
- (2) Use LR. What is the MSE?






## TORUN new expert policy!
for expert_pol in 'optimal'; do
  python -u run_airl.py --name 0719_lmdp_nodegam --random_search 30 --epochs 80 --mdp linear --arch AIRL_NODEGAM_Lightning --expert_pol ${expert_pol}
  python -u run_airl.py --name 0719_gammdp_nodegam --random_search 30 --epochs 80 --mdp gam --arch AIRL_NODEGAM_Lightning --expert_pol ${expert_pol}
  python -u run_airl.py --name 0719_lmdp_linear --random_search 20 --epochs 80 --mdp linear --arch AIRLLightning --expert_pol ${expert_pol}
  python -u run_airl.py --name 0719_gammdp_linear --random_search 20 --epochs 80 --mdp gam --arch AIRLLightning --expert_pol ${expert_pol}
done

## TODO
# (1) Run the apprenticeship learning
# (2) Maybe implement the DFSN?
# (3) Normalize the GAM plot to be just ranking between 0 and 1, then calculate the error weighted by samples

## Others:
# (1) Think about the GAM with one-class prediction
  - That means in the latent space each output is a GAM, and each dimension can be treated as the center of that interaction / main effect term. And since L2 distance can be added up, that would correspond to the average distance to all centers. Maybe I should treat it as max distance?
  - The graph looks like the 2d plot with number as distance, and I know which are normal points in terms of features



# RERUN LINEAR: it's wrong lol
#python -u run_airl.py --name 0716_linear --random_search 15 --epochs 50 --arch AIRLLightning
#for expert_pol in 'eps0.07' 'eps0.14'; do
#  python -u run_airl.py --name 0716_linear_${expert_pol} --random_search 15 --epochs 50 --expert_pol ${expert_pol} --arch AIRLLightning
#done



## TORUN: the GAM mdp with linear...
#python -u run_airl.py --name 0715_linear --random_search 15 --epochs 50 --expert_pol eps0.07




# Linear with GAM MDP. Of course would get very worse....
#python -u run_airl.py --name 0715_gam --random_search 15 --epochs 50 --mdp gam



# TODO: do different policy. What if I learn from eps greedy policy? (Like imperfect demonstration?)


#python -u run_airl.py --name 0714_node_gammdp --random_search 15 --epochs 50 --mdp gam --arch AIRL_NODEGAM_Lightning


#
## Run a different initialization
#for gamma in '0.9' '0.99'; do
#  python main.py --name 0702 --pi_init bc --mdp linear --opt_method projection --gamma ${gamma} &
#done
#
## Rerun the died one
#gamma='0.9'
#python main.py --name 0702 --pi_init uniform --mdp linear --opt_method projection --gamma ${gamma}
