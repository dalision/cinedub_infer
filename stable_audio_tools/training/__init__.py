# Intentionally left empty at inference release time.
# The training tree (factory.py, diffusion.py, autoencoders.py, lm.py, losses/) was removed
# from this open-source inference release. Only training/utils.py is kept because
# generate_v2a_cond.py imports a handful of helpers (copy_state_dict, replace_mp4_wav,
# generate_multimodal_tasks) from it. Do NOT re-export the training wrappers here.
