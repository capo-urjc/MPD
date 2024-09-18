python train_NO2.py --horizon 8 \
--sample_freq 250 \
--diffusion models.GaussianDiffusionImitationCondition \
--n_train_steps 5000 \
--n_steps_per_epoch 1000 \
--save_freq 500 \
--action_weight 0 \
--loader  datasets.GoalDataset \
--loss_type l1 \
--renderer utils.EMARenderer \
--n_diffusion_steps 20 \
--learning_rate 1e-4 \
--fn_condition apply_harder_direction_coord_conditioning
