# GlobalLocalOddballs stimulus generation code

This repository contains the original code used to generate the stimuli for the **GlobalLocalOddballs project**, an [Allen Institute for Brain Science](https://alleninstitute.org/what-we-do/brain-science/) [OpenScope](https://alleninstitute.org/what-we-do/brain-science/news-press/press-releases/openscope-first-shared-observatory-neuroscience) project.
&nbsp;

The GlobalLocalOddballs experiment was conceptualized by [Jake Westerberg](http://www.westerberg.science/) (Vanderbilt University), [André Bastos](https://www.bastoslabvu.com/) (Vanderbilt University), and [Alex Maier](http://www.maierlab.com/) (Vanderbilt University). The stimuli were coded by [Jake Westerberg].

The experiment details, analyses, and results are forthcoming.
&nbsp;

## Installation
### Dependencies:
- Windows OS (see **Camstim package**)
- python 2.7
- psychopy 1.82.01
- camstim 0.2.4
&nbsp;

### Camstim 0.2.4:
- Built and licensed by the [Allen Institute](https://alleninstitute.org/).
- Written in **Python 2** and designed for **Windows OS** (requires `pywin32`).
- Pickled stimulus presentation logs are typically saved under `user/camstim/output`.
&nbsp;

### Installation with [Anaconda](https://docs.anaconda.com/anaconda/install/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html):
1. Navigate to repository and install conda environment.  
    `conda env create -f openscope-glo-stim.yml`
2. Activate the environment.  
    `conda activate openscope-glo-stim`
3. Install the AIBS `camstim` package in the environment.  
    `pip install camstim/.`
4. Download and install [`AVbin`](https://avbin.github.io/AVbin/Download.html) for your OS.  
&nbsp;

## Scripts  
You can try out a test script by navigating into the test-scripts folder and running any of the example file: `python cohort-1-habituation-5min-drifting.py`
&nbsp;

## Log files
- Pickled stimulus presentation logs are typically saved under `user/camstim/output`.
- Sweep parameters are under a few keys of `['stimuli'][n]`, where `n` is the stimulus number.
- Stimulus parameters are in the following dictionary: `['stimuli'][0]['stimParams']` or `['stimuli'][0]['stim_params']`.  
&nbsp;

## Additional notes
- On some computers, the black screen that should intervene each sequence appears as a white screen instead. Change the color_inversion value to `True` in the SESSION_PARAMS at the top of the .py file to fix.
&nbsp;
