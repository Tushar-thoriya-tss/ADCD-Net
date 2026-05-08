import os.path as op

# ============================================================================
# TRAINING CONFIG
# When training, you only need to look at this section.
# ============================================================================

# ---------------- Mode ----------------
# 'train' = train the model;  'val' = run validation/inference only.
run_mode = 'val'   # 'train' | 'val'
check_val = False

# ---------------- Runtime ----------------
gpus = '0'
device_n = len(gpus.split(','))
dl_workers = 0

# ---------------- Paths ----------------
root = '/sharedrive/Tushar Thoriya/GitHub techniques/DATAAAAASET'
pkl_dir = op.join(root, 'path_pkl')              # built by build_path_pkl.py
qt_path = op.join(root, 'exp_data/qt_table.pk')  # built by gen_qt_table.py
train_ds_name = 'aadhar_tamper_train'

# ---------------- Weight initialization ----------------
# How to initialize model weights when training. Ignored in val mode.
#   'scratch'     -> train from scratch (only DocRes backbone is loaded)
#   'pretrained'  -> fine-tune from the checkpoint at `pretrained_ckpt`
# In val mode, `pretrained_ckpt` MUST point to a trained checkpoint.
init_weights = 'scratch'  # 'scratch' | 'pretrained'

pretrained_ckpt  = '/sharedrive/Tushar Thoriya/GitHub techniques/DATAAAAASET/exp_out/ADCDNet/Log_v05071857/ckpt/Step80000_Score0.7659.pth'
docres_ckpt_path = '/sharedrive/Tushar Thoriya/GitHub techniques/ADCD-Net/ADCD-Net_exp_data/docres.pkl'

# ---------------- Schedule ----------------
train_bs       = 2          # batch size per device
step_per_epoch = 1000         # 1000 for full runs
epochs         = 200       # 200 for full runs
accum_step     = 2          # gradient accumulation steps
init_S         = 0          # JPEG curriculum start step
print_log_step = 50
val_step       = step_per_epoch * 10   # mid-training val cadence
ds_len = sample_per_epoch = step_per_epoch * train_bs * device_n
total_step = step_per_epoch * epochs

# ---------------- Optimizer / loss ----------------
lr           = 3e-4
min_lr       = 1e-5
weight_decay = 1e-4
ce_w    = 3
rec_w   = 1
focal_w = 0.2
norm_w  = 0.1

# ---------------- Model ----------------
exp_root_name = 'ADCDNet'
img_size      = 256

# Resolved checkpoint used by the training/val code.
# Do not edit directly — change `init_weights` / `run_mode` above.
if run_mode == 'val':
    ckpt = pretrained_ckpt
elif init_weights == 'pretrained':
    ckpt = pretrained_ckpt
elif init_weights == 'scratch':
    ckpt = None
else:
    raise ValueError(f"Invalid init_weights: {init_weights!r}. Use 'scratch' or 'pretrained'.")


# ============================================================================
# VAL / INFERENCE CONFIG
# Only edit this section when running validation or inference.
# ============================================================================

val_bs        = 1
val_max_size  = 1024         # (512) set very large to disable resize (watch GPU mem)
val_sample_n  = 10           # cap for mid-training val sampling (100 for full)

# Validation dataset pkl base names (without the .pkl suffix).
all_ds_name = ['aadhar_tamper_val']

# In train mode, mid-training validation uses '<name>_sample' to cap at val_sample_n.
all_ds_name_s = [name + '_sample' for name in all_ds_name]
val_name_list = all_ds_name_s if run_mode == 'train' else all_ds_name
