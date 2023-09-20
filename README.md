# Spike sorting : how good are we to correct for motion ?


Here all the material to generate and reproduce all figures for the paper 
**A modular approach to handle in-vivo drift correction for high-density extracellular recordings**

preprint here: https://www.biorxiv.org/content/10.1101/2023.06.29.546882v1



## step 1 : configure

  * open and modify the script : `configuration.py`


## step 2 : generate fake dataset

  * run the script : `generate.py`.
    This can be very very long


## step 3 : run all motion estimation

  * run the script : `run_benchmark_estimate_motion.py`.


## step 4 : run all motion interpolation

  * run the script : `run_benchmark_interpolate_motion.py`.


## make the figure with jupyter lab

  * open jupyter lab
  * open: 
    * `make_figure_explain_drifts.ipynb`
    * `make_figure_estimation.ipynb`
    * `make_figure_interpolation.ipynb`
    * `make_figure_waveforms_distortion.ipynb`
  * and run the notebooks


## Additionall files

Theses extra scipt have a collection of function to construct figures in the jupyter notebooks

  * `figure_estimation.py`
  * `figure_interpolation.py`
  * `figure_waveforms_distortion.py`
  * `plotting_tools.py`




