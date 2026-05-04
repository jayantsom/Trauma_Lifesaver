# Trauma Lifesaver

## RSNA Classification-With-Mask Training

Use this command in Google Colab to train on RSNA config 3:
`classification-with-mask`.

This is mask-aware multitask training. It trains the segmentation mask head and
the injury-label classification head together.

```bash
python training/train_unet_segmenter.py \
  --config classification-with-mask \
  --num_epochs 10 \
  --batch_size 1 \
  --gradient_accumulation_steps 4 \
  --slices_per_volume 6 \
  --image_size 256 \
  --cache_dir /content/rsna_cache \
  --keep_nifti_cache \
  --output_dir /content/drive/MyDrive/Trauma_Lifesaver/models/unet_hemorrhage
```

The checkpoint is saved as:

```text
/content/drive/MyDrive/Trauma_Lifesaver/models/unet_hemorrhage/unet_hemorrhage_rsna_mask_aware.pth
```

Supported trainer configs:

- `segmentation`: trains segmentation masks only.
- `classification-with-mask`: trains segmentation plus injury-label classification.
- `both`: combines the two mask-aware configs.
