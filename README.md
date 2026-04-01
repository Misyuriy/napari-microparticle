# Powder Particle Analysis (napari-microparticle) 🔬

---
[Napari](https://github.com/napari/napari) plugin for powder particle size measurement and morphology analysis in 2D microscopy images.

This plugin is meant to be used together with the [micro-sam](https://github.com/computational-cell-analytics/micro-sam) tools. 
[Segment Anything for Microscopy](https://www.nature.com/articles/s41592-024-02580-4) provides a convenient napari interface
for interactive and automatic image segmentation using the Segment Anything Models (SAM).

**This plugin implements napari applications for:**
1. Cleaning SAM segmentation artifacts
2. Measuring size (feret diameter, equivalent diameter and area) of segmented masks
3. Segmenting porous microstructure in SEM images with watershed
4. Measuring pore area fraction for porous particles

All measured data can be exported to .xlsx or .csv spreadsheets

---
## Installation

1. Install `micro-sam` from conda or from an installer. Check the [micro-sam installation documentation](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#installation) for details. **Installation from installer is not recommended**
2. Open napari. If you installed `micro-sam` from conda, this is done with the `napari` command.
3. Go to "Plugins" > "Install/Uninstall Plugins..."
4. Search for "napari-microparticle" and click the blue "Install" button
5. Wait for the installation to finish. Reopen napari.

## Usage

To use any tool from this plugin, go to "Plugins" > "Powder Particle Analysis" and select an option from the menu. This will open a panel inside the napari interface.
Any number of panels can be open at the same time.

To use this plugin, you need a Labels layer with particle segmentation masks. You can quickly segment particles using the
micro-sam Annotator2D tool.

Check the [micro-sam video tutorials](https://youtube.com/playlist?list=PLwYZXQJ3f36GQPpKCrSbHjGiH39X4XjSO&si=qNbB8IFXqAX33r_Z) for details on how to use the Annotator2D

---
## SAM finetuning
This repository also contains code (`./sam_finetuning`) that was used to finetune SAM checkpoints for particle segmentation in SEM (Scanning Electron Microscope) images of powder.
It is based on the training pipeline implemented in micro-sam.