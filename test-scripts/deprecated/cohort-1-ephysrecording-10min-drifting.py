"""

Title:          GlobalLocalOddballs (GLO) stimulus generation code
Author:         Jacob A. Westerberg (Vanderbilt University)
Contact:        jacob.a.westerberg@vanderbilt.edu
Git Repo:       openscope-glo-stim (westerberg-science)
Written:        2022-03-18
Updated:        2022-03-22

"""

import Tkinter as tk
import tkSimpleDialog
from psychopy import monitors, visual

from camstim import SweepStim, Stimulus, Foraging
from camstim import Window, Warp

import pickle as pkl
import os
import random
import itertools
import time

from psychopy import event, logging, core
from psychopy.visual import ElementArrayStim
from psychopy.tools.arraytools import val2array
from psychopy.tools.attributetools import attributeSetter, setAttribute

import numpy as np

SESSION_PARAMS  = { 'subject_id':                   'test',                 # subject identifier information
                    'session_id':                   'test',                 # session identifier information
                    'cohort':                       1,                      # which orientation cohort (1, 2)
                    'habituation_duration':         60 * 1,                 # desired habituation block duration (sec)
                    'glo_duration':                 60 * 7,                 # desired GLO block duration (sec)
                    'control_duration':             60 * 2,                 # desired control block duration (sec)
                    'pre_blank':                    5,                      # blank before stim starts (sec)
                    'post_blank':                   5,                      # blank after all stims end (sec)
                    'stimulus_orientations':        [135, 45],              # two orientations
                    'stimulus_drift_rate':          4.0,                    # stimulus drift rate (0 for static)
                    'stimulus_spatial_freq':        0.04,                   # spatial frequency of grating
                    'stimulus_duration':            0.5,                    # stimulus presentation duration (sec)
                    'stimulus_contrast':            0.8,                    # stimulus contrast (0-1)
                    'stimulus_phase':               [0.0, 0.25, 0.5, 0.75], # possible phases for gratings (0-1)
                    'interstimulus_duration':       0.5,                    # blank between all stims (sec)
                    'global_oddball_proportion':    0.2,                    # proportion of global oddball trials in GLO block (0-1)
                    'color_inversion':              False,
                    }

RIG_PARAMS      = {}

"""
RIG_PARAMS       = { 'syncpulse':                   True,
                     'syncpulseport':               1,
                     'syncpulselines':              [4, 7],  # frame, start/stop
                     'trigger_delay_sec':           0.0,
                     'bgcolor':                     (-1,-1,-1),
                     'eyetracker':                  False,
                     'eyetrackerip':                "W7DT12722",
                     'eyetrackerport':              1000,
                     'syncsqr':                     True,
                     'syncsqrloc':                  (0,0),
                     'syncsqrfreq':                 60,
                     'syncsqrsize':                 (100,100),
                     'showmouse':                   True
                     }
"""

def winVar(win, units):
    """Returns width and height of the window in units as tuple.
    Takes window and units.
    """
    dist = win.monitor.getDistance()
    width = win.monitor.getWidth()

    # get values to convert deg to pixels
    deg_wid = np.rad2deg(np.arctan((0.5*width)/dist)) * 2 # about 120
    deg_per_pix = deg_wid/win.size[0] # about 0.07

    if units == 'deg':
        deg_hei = deg_per_pix * win.size[1] # about 67
        # Something is wrong with deg as this does not fill screen
        init_wid = deg_wid
        init_hei = deg_hei
        fieldSize = [init_wid, init_hei]

    elif units == 'pix':
        init_wid = win.size[0]
        init_hei = win.size[1]
        fieldSize = [init_wid, init_hei]

    else:
        raise ValueError('Only implemented for deg or pixel units so far.')

    return fieldSize, deg_per_pix

def init_grating(window, session_params, contrast, phase, tf, sf, ori):

        grating            = Stimulus(visual.GratingStim(window,
                                        pos                 = (0, 0),
                                        units               = 'deg',
                                        size                = (1000, 1000),
                                        mask                = "None",
                                        texRes              = 256,
                                        sf                  = 0.1,
                                        ),
                                        sweep_params        = { 'Contrast': ([contrast], 0),
                                                                'Phase': ([phase], 1),
                                                                'TF': ([tf], 2),
                                                                'SF': ([sf], 3),
                                                                'Ori': ([ori], 4),
                                                                 },
                                        sweep_length        = session_params['stimulus_duration'],
                                        start_time          = 0.0,
                                        blank_length        = session_params['interstimulus_duration'],
                                        blank_sweeps        = 0,
                                        runs                = 1,
                                        shuffle             = False,
                                        save_sweep_table    = True,
                                        )

        return grating

def generate_sequence(window, session_params, in_session_time, stimulus_counter, duration, global_oddball_proportion, is_control):

    sequence = {}

    sequence_count      = int(np.floor(duration /
                                (5 * (session_params['stimulus_duration'] +
                                session_params['interstimulus_duration']))))

    sequence['stimulus_count']  = int(sequence_count * 5)
    sequence['trial_count']     = int(sequence_count)

    try:
        gratings = session_params['gratings']
        stimulus_counter = stimulus_counter
    except:
        gratings = []
        stimulus_counter = 0

    if session_params['cohort'] == 1:
        o1              = session_params['stimulus_orientations'][0]
        o2              = session_params['stimulus_orientations'][1]
    elif session_params['cohort'] == 2:
        o1              = session_params['stimulus_orientations'][1]
        o2              = session_params['stimulus_orientations'][0]

    if is_control:
        trial_vector    = [o1, o1, o1, o1, o1]
        sequence['orientations']        = np.squeeze(np.tile(trial_vector, (1, sequence_count)))
        rints                           = session_params['rng'].choice(np.arange(0, sequence['stimulus_count'] - 1, 1),
                                                                                size = sequence['stimulus_count'] / 2,
                                                                                replace = False)
        for i in rints:
            sequence['orientations'][i] = o2

    else:
        trial_vector                    = [o1, o1, o1, o1, o2]
        sequence['orientations']        = np.squeeze(np.tile(trial_vector, (1, sequence_count)))

        if global_oddball_proportion > 0:
            global_oddball_count    = int(np.round(sequence_count * global_oddball_proportion))
            rints                   = np.multiply(session_params['rng'].choice(np.arange(1, sequence_count, 1),
                                                    size = global_oddball_count,
                                                    replace = False), 5)
            for i in rints:
                sequence['orientations'][i-1] = o1

    # Insert blank 'presentations'
    sequence['blanks']              = np.squeeze(np.tile([True, False, False, False, False], (1, sequence_count)))

    # randomize stim phase from params
    if np.size(session_params['stimulus_phase']) == 1:
        sequence['phases']              = np.squeeze(np.tile(session_params['stimulus_phase'], (1, sequence['stimulus_count'])))
    else:
        sequence['phases']              = np.squeeze(session_params['rng'].choice(session_params['stimulus_phase'],
                                                                                size = sequence['stimulus_count'],
                                                                                replace = True))

    # randomize stim phase from params
    if np.size(session_params['stimulus_contrast']) == 1:
        sequence['contrasts']              = np.squeeze(np.tile(session_params['stimulus_contrast'], (1, sequence['stimulus_count'])))
    else:
        sequence['contrasts']           = np.squeeze(session_params['rng'].choice(session_params['stimulus_contrast'],
                                                                                size = sequence['stimulus_count'],
                                                                                replace = True))

    # randomize stim phase from params
    if np.size(session_params['stimulus_drift_rate']) == 1:
        sequence['TFs']              = np.squeeze(np.tile(session_params['stimulus_drift_rate'], (1, sequence['stimulus_count'])))
    else:
        sequence['TFs']                 = np.squeeze(session_params['rng'].choice(session_params['stimulus_drift_rate'],
                                                                                size = sequence['stimulus_count'],
                                                                                replace = True))

    # randomize stim phase from params
    if np.size(session_params['stimulus_spatial_freq']) == 1:
        sequence['SFs']              = np.squeeze(np.tile(session_params['stimulus_spatial_freq'], (1, sequence['stimulus_count'])))
    else:
        sequence['SFs']                 = np.squeeze(session_params['rng'].choice(session_params['stimulus_spatial_freq'],
                                                                                size = sequence['stimulus_count'],
                                                                                replace = True))

    # generate the stimuli
    for i in np.arange(0, sequence['stimulus_count'], 1):

        if sequence['blanks'][i] == True:

            if session_params['color_inversion'] == True:
                blank_color     = 1
            else:
                blank_color     = -1

            gratings.append(init_grating(window, session_params, blank_color, 0, 0, 0, 0,))

        else:

            gratings.append(init_grating(window,
                                          session_params,
                                          sequence['contrasts'][i],
                                          sequence['phases'][i],
                                          sequence['TFs'][i],
                                          sequence['SFs'][i],
                                          sequence['orientations'][i],
                                          ))

        gratings[stimulus_counter].set_display_sequence([(in_session_time, in_session_time
                                                            + session_params['stimulus_duration']
                                                            + session_params['interstimulus_duration'])])

        in_session_time = in_session_time + session_params['stimulus_duration'] + session_params['interstimulus_duration'];
        stimulus_counter = stimulus_counter + 1

    return gratings, in_session_time, stimulus_counter

if __name__ == "__main__":

    dist = 15.0
    wid = 52.0

    # create a monitor
    monitor = monitors.Monitor("testMonitor", distance=dist, width=wid) #"Gamma1.Luminance50"

    # Create display window
    window = Window(fullscr=True, # Will return an error due to default size. Ignore.
                    monitor=monitor,  # Will be set to a gamma calibrated profile by MPE
                    screen=0,
                    warp=Warp.Spherical
                    )

    # Init stimulus time tracking
    in_session_time = 0
    stimulus_counter = 0

    # randomly set a seed for the session and create a dictionary
    SESSION_PARAMS['seed'] = random.choice(range(0, 48000))
    # SESSION_PARAMS['seed'] = # override by setting seed manually
    SESSION_PARAMS['rng'] = np.random.RandomState(SESSION_PARAMS['seed'])

    total_time_calc             = SESSION_PARAMS['habituation_duration'] + SESSION_PARAMS['glo_duration'] + SESSION_PARAMS['control_duration']
    habituation_trial_count     = int(np.floor(SESSION_PARAMS['habituation_duration'] / ( 5 * (SESSION_PARAMS['stimulus_duration'] + SESSION_PARAMS['interstimulus_duration']))))
    glo_trial_count             = int(np.floor(SESSION_PARAMS['glo_duration'] / ( 5 * (SESSION_PARAMS['stimulus_duration'] + SESSION_PARAMS['interstimulus_duration']))))
    control_trial_count         = int(np.floor(SESSION_PARAMS['control_duration'] / ( 5 * (SESSION_PARAMS['stimulus_duration'] + SESSION_PARAMS['interstimulus_duration']))))
    local_oddball_count         = glo_trial_count - int(np.round(glo_trial_count * SESSION_PARAMS['global_oddball_proportion']))
    global_oddball_count        = int(np.round(glo_trial_count * SESSION_PARAMS['global_oddball_proportion']))

    print('')
    print('%%%%% TASK SEQUENCE INFORMATION %%%%%')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('')
    print('Total time (min)            : ' + str(total_time_calc / 60))
    print('Habituation trial count     : ' + str(habituation_trial_count))
    print('Total GLO trial count       : ' + str(glo_trial_count))
    print('Local oddball trial count   : ' + str(local_oddball_count))
    print('Global oddball trial count  : ' + str(global_oddball_count))
    print('Control trial count         : ' + str(control_trial_count))
    print('')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('')

    # Create the habituation block
    SESSION_PARAMS['gratings'], in_session_time, stimulus_counter        = generate_sequence(window,
                                                                                        SESSION_PARAMS,
                                                                                        in_session_time,
                                                                                        stimulus_counter,
                                                                                        SESSION_PARAMS['habituation_duration'],
                                                                                        0,
                                                                                        False)

    # add global oddball and control blocks if an ephys session
    if SESSION_PARAMS['glo_duration'] > 0:

        SESSION_PARAMS['gratings'], in_session_time, stimulus_counter    = generate_sequence(window,
                                                                                        SESSION_PARAMS,
                                                                                        in_session_time,
                                                                                        stimulus_counter,
                                                                                        SESSION_PARAMS['glo_duration'],
                                                                                        SESSION_PARAMS['global_oddball_proportion'],
                                                                                        False)

        SESSION_PARAMS['gratings'], in_session_time, stimulus_counter    = generate_sequence(window,
                                                                                        SESSION_PARAMS,
                                                                                        in_session_time,
                                                                                        stimulus_counter,
                                                                                        SESSION_PARAMS['control_duration'],
                                                                                        0,
                                                                                        True)

    ss          = SweepStim(window,
                            stimuli         = SESSION_PARAMS['gratings'],
                            pre_blank_sec   = SESSION_PARAMS['pre_blank'],
                            post_blank_sec  = SESSION_PARAMS['post_blank'],
                            params          = RIG_PARAMS,  # will be set by MPE to work on the rig
                            )

    # add in foraging so we can track wheel, potentially give rewards, etc
    f               = Foraging(window       = window,
                                auto_update = False,
                                params      = RIG_PARAMS,
                                nidaq_tasks = {'digital_input': ss.di, 'digital_output': ss.do,},
                                )
    ss.add_item(f, "foraging")

    # run it
    ss.run()
