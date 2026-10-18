# Retinal-Wave-Inspired Stimuli in Flyvis

**A mini-project exploring how propagating activity patterns influence a connectome-constrained visual system model.**

## Overview

Retinal waves are spontaneous, spatially and temporally correlated patterns of neural activity observed in the developing retina. They are thought to contribute to the organization and maturation of visual circuits before normal visual experience.

This project explores a simplified question:

**How does a pretrained visual neural network respond to propagating wave-like activity compared with conventional motion stimuli and random noise?**

To investigate this, synthetic retinal-wave-inspired stimuli were generated and presented to **Flyvis**, a connectome-constrained model of the *Drosophila* (fruit fly) visual system.

The project is an exploratory proof of concept, rather than a biologically detailed simulation of retinal development.

## Experimental Pipeline

1. Generate synthetic spatiotemporal stimuli on a 64 × 64 grid.
2. Normalize stimulus movies to comparable global luminance and contrast statistics.
3. Project the movies onto Flyvis' hexagonal visual lattice using `BoxEye`.
4. Simulate responses using a pretrained Flyvis network.
5. Extract and compare temporal responses from selected cell types.

## Stimuli

Three stimulus conditions were implemented:

- **Retinal-wave-inspired stimulus:** a Gaussian ring expanding outward from a fixed origin
- **Moving bar:** a bright vertical bar translating horizontally across the visual field
- **Random noise:** spatially and temporally uncorrelated Gaussian noise

All stimuli contain **60 frames** and are simulated with a time step of **10 ms**.

### Retinal-wave-inspired stimulus animation

The main stimulus used in this project is an expanding wavefront.  
The animation below shows the synthetic retinal-wave-inspired stimulus before projection into Flyvis.

![Retinal wave animation](results/retinal_wave.gif)

### Stimulus overview

The figure below compares representative frames from all three stimulus conditions after global normalization.

![Stimulus overview](results/stimulus_overview.png)

The retinal wave contains a propagating expanding structure, the moving bar provides a conventional motion stimulus, and the random-noise condition serves as an unstructured control.

## Flyvis Model

The experiments use the pretrained `flow/0000/000` network from [Flyvis](https://github.com/TuragaLab/flyvis).

The network is based on the connectivity of the fruit fly visual system and contains approximately 45,669 modeled neurons across 65 cell types.

The visual input is projected onto 721 hexagonal sampling positions.

**Input dimensions:** `(1, 60, 1, 721)`

**Neural response dimensions:** `(1, 60, 45669)`

No additional network training or fine-tuning was performed.

## Preliminary Results

### 1. Response to the retinal-wave-inspired stimulus

The first analysis examines how selected Flyvis cell types respond to the synthetic retinal wave alone.

![Cell type responses](results/cell_type_responses.png)

Responses were examined for five representative cell types:

- `R1`: photoreceptor
- `L1`: lamina neuron
- `Mi1`: medulla neuron
- `T4a`: motion-processing pathway
- `T5a`: motion-processing pathway

These cell types exhibit distinct temporal response profiles, indicating that the propagating wavefront drives structured activity throughout multiple stages of the Flyvis visual pathway.

### 2. Comparison with control stimuli

The next analysis compares responses across the three stimulus classes: retinal wave, moving bar, and random noise.  
Before comparison, the stimulus movies were normalized to comparable global mean luminance and standard deviation.

![Normalized stimulus comparison](results/stimulus_comparison_normalized.png)

Several qualitative observations can be made from this figure:

- The **retinal wave** produces clear time-varying responses across all selected cell types.
- The **moving bar** often produces partially similar dynamics, especially in later visual pathways such as `T5a`.
- The **random-noise** condition generally remains closer to baseline in the spatially averaged responses shown here.

These results suggest that the pretrained Flyvis model responds differently to structured propagating input than to uncorrelated noise.  
They also suggest that the retinal-wave-inspired stimulus shares some dynamic properties with conventional motion stimuli, while still producing distinct temporal response patterns.

### 3. Whole-network average response

For completeness, the average activity across all modeled neurons was also examined for the retinal-wave-inspired stimulus.

![Whole-network response](results/flyvis_wave_response.png)

This plot serves mainly as a sanity check showing that the stimulus drives non-trivial temporal dynamics in the network.  
Because it averages across all neurons, it is less informative than the cell-type-specific plots above.

## Installation

The project was developed under Ubuntu on WSL2 using Python 3.11.

Create a Conda environment:

```bash
conda create -n retinal-waves python=3.11 -y
conda activate retinal-waves
```

Install dependencies:

```bash
pip install flyvis numpy torch matplotlib pillow
```

Download the pretrained networks:

```bash
flyvis download-pretrained
```

## Reproducing the Experiments

Generate the wave animation:

```bash
python wave_generator.py
```

Run the pretrained network simulation:

```bash
python run_wave_through_flyvis.py
```

Plot individual cell-type responses:

```bash
python plot_cell_type_responses.py
```

Compare the three stimulus conditions:

```bash
python compare_stimuli_responses.py
```

Generate the stimulus overview figure:

```bash
python plot_stimuli.py
```

Outputs are saved in the `results/` directory.
