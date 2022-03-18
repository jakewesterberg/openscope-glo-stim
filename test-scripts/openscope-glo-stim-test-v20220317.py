"""

Title:          GlobalLocalOddballs (GLO) stimulus generation code
Author:         Jacob A. Westerberg (Vanderbilt University)
Contact:        jacob.a.westerberg@vanderbilt.edu
Git Repo:       openscope-glo-stim (westerberg-science)
Written:        2022-03-16
Updated:        2022-03-17

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

SESSION_PARAMS  = { 'type':                         'habituation',          # type of session (habituation or electrophysiology)
                    'seed':                         1,                      # seed to use for rng
                    'habituation_duration':         60 * 0.25,              # desired habituation block duration (sec)
                    'glo_duration':                 60 * 0,                 # desired GLO block duration (sec)
                    'control_duration':             60 * 0,                 # desired control block duration (sec)
                    'pre_blank':                    1,                      # blank before stim starts (sec)
                    'post_blank':                   1,                      # blank after all stims end (sec)
                    'stimulus_orientations':        [45, 135],              # two orientations
                    'stimulus_drift_rate':          4.0,                    # stimulus drift rate (0 for static)
                    'stimulus_spatial_freq':        0.004,                   # spatial frequency of grating
                    'stimulus_duration':            0.5,                    # stimulus presentation duration (sec)
                    'stimulus_contrast':            0.75,                   # stimulus contrast (0-1)
                    'stimulus_phase':               [0.0, 0.25, 0.5, 1],    # possible phases for gratings (0-1)
                    'interstimulus_duration':       0.5,                    # blank between all stims (sec)
                    'global_oddball_proportion':    0.5,                    # proportion of global oddball trials in GLO block (0-1)
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

def generate_sequence(session_params, duration, global_oddball_proportion, is_control):

    sequence_count = int(np.floor(duration /
                        ( 5 * session_params['stimulus_duration'] +
                        session_params['interstimulus_duration'])))

    if session_params['seed'] == 1:
        o1              = session_params['stimulus_orientations'][0]
        o2              = session_params['stimulus_orientations'][1]
    elif session_params['seed'] % 2 == 0:
        o1              = session_params['stimulus_orientations'][1]
        o2              = session_params['stimulus_orientations'][0]

    if is_control:
        trial_vector    = [o1, o1, o1, o1, o1]
        sequence        = np.squeeze(np.tile(trial_vector, (1, sequence_count)))
        rints           = session_params['rng'].choice(np.arange(0, np.size(sequence)-1, 1),
                                                                    size=np.round(np.size(sequence)/2),
                                                                    replace=False)
        for i in rints:
            sequence[i] = o2

    else:
        trial_vector    = [o1, o1, o1, o2, o1]
        sequence        = np.squeeze(np.tile(trial_vector, (1, sequence_count)))

        if global_oddball_proportion > 0:
            global_oddball_count    = int(np.round(sequence_count * global_oddball_proportion))
            rints                   = np.multiply(session_params['rng'].choice(np.arange(1, sequence_count, 1),
                                                    size = global_oddball_count,
                                                    replace = False), 5)
            for i in rints:
                sequence[i-2] = o1

    return sequence

def init_sequence(window, session_params, sequence):

        contrasts           = np.zeros((1,np.size(sequence))) + session_params['stimulus_contrast']
        blank_presentation  = np.arange(4, np.size(sequence), 5)
        for i in blank_presentation
            contrasts[i]    = 0

        phases              = session_params['rng'].choice(session_params['stimulus_phase'],
                                                            size = np.size(sequence),
                                                            replace = True)

        print(phases)
        print(contrasts)
        print(sequence)

        gratings            = Stimulus(visual.GratingStim(window,
                                        pos                 = (0, 0),
                                        units               = 'deg',
                                        size                = (500, 500),
                                        mask                = "None",
                                        texRes              = 256,
                                        sf                  = 0.1,
                                        ),
                                        sweep_params        = { 'Contrast':     ([contrasts], 0),
                                                                'Phase':        ([phases], 1),
                                                                'TF':           ([session_params['stimulus_drift_rate']], 2),
                                                                'SF':           ([session_params['stimulus_spatial_freq']], 3),
                                                                'Ori':          ([sequence], 4),
                                                                 },
                                        sweep_length        = session_params['stimulus_duration'],
                                        start_time          = 0.0,
                                        blank_length        = session_params['interstimulus_duration'],
                                        blank_sweeps        = 0,
                                        runs                = 1,
                                        shuffle             = False,
                                        save_sweep_table    = True,
                                        )

        return gratings

if __name__ == "__main__":

    dist = 15.0
    wid = 52.0

    # load and record parameters. Leave False.
    promptID = False

    # create a monitor
    monitor = monitors.Monitor("testMonitor", distance=dist, width=wid) #"Gamma1.Luminance50"

    # get animal ID and session ID
    if promptID == True: # using a prompt
        myDlg = tk.Tk()
        myDlg.withdraw()
        subj_id = tkSimpleDialog.askstring("Input",
                                           "Subject ID (only nbrs, letters, _ ): ",
                                           parent=myDlg)
        sess_id = tkSimpleDialog.askstring("Input",
                                           "Session ID (only nbrs, letters, _ ): ",
                                           parent=myDlg)

        if subj_id is None or sess_id is None:
            raise ValueError('No Subject and/or Session ID entered.')

    else: # Could also just enter it here.
        # if subj_id is left as None, will skip loading subj config.
        subj_id = None
        sess_id = None
        seed    = SESSION_PARAMS['seed']

    # Create display window
    window = Window(fullscr=True, # Will return an error due to default size. Ignore.
                    monitor=monitor,  # Will be set to a gamma calibrated profile by MPE
                    screen=0,
                    warp=Warp.Spherical
                    )

    # Create rng
    SESSION_PARAMS['rng'] = np.random.RandomState(SESSION_PARAMS['seed'])

    total_time_calc             = SESSION_PARAMS['habituation_duration'] + SESSION_PARAMS['glo_duration'] + SESSION_PARAMS['control_duration']
    habituation_trial_count     = int(np.floor(SESSION_PARAMS['habituation_duration'] / ( 5 * SESSION_PARAMS['stimulus_duration'] + SESSION_PARAMS['interstimulus_duration'])))
    glo_trial_count             = int(np.floor(SESSION_PARAMS['glo_duration'] / ( 5 * SESSION_PARAMS['stimulus_duration'] + SESSION_PARAMS['interstimulus_duration'])))
    control_trial_count         = int(np.floor(SESSION_PARAMS['control_duration'] / ( 5 * SESSION_PARAMS['stimulus_duration'] + SESSION_PARAMS['interstimulus_duration'])))
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
    SESSION_PARAMS['habituation_sequence']    = generate_sequence(SESSION_PARAMS, SESSION_PARAMS['habituation_duration'], 0, False)
    habituation_block                         = init_sequence(window, SESSION_PARAMS, SESSION_PARAMS['habituation_sequence'])
    habituation_ds                            = [(0, SESSION_PARAMS['habituation_duration'])]
    habituation_block.set_display_sequence(habituation_ds);

    # add global oddball and control blocks if an ephys session
    if SESSION_PARAMS['glo_duration'] > 0:

        SESSION_PARAMS['glo_sequence']        = generate_sequence(SESSION_PARAMS, SESSION_PARAMS['glo_duration'], SESSION_PARAMS['global_oddball_proportion'], False)
        SESSION_PARAMS['control_sequence']    = generate_sequence(SESSION_PARAMS, SESSION_PARAMS['control_duration'], 0, True)

        glo_block                             = init_sequence(window, SESSION_PARAMS, SESSION_PARAMS['glo_sequence'])
        control_block                         = init_sequence(window, SESSION_PARAMS, SESSION_PARAMS['control_sequence'])

        glo_ds      = [(SESSION_PARAMS['habituation_duration'],
                                SESSION_PARAMS['habituation_duration'] + SESSION_PARAMS['glo_duration'])]
        control_ds  = [(SESSION_PARAMS['habituation_duration'] + SESSION_PARAMS['glo_duration'],
                        SESSION_PARAMS['habituation_duration'] + SESSION_PARAMS['glo_duration'] + SESSION_PARAMS['control_duration'])]

        glo_block.set_display_sequence(glo_ds);
        control_block.set_display_sequence(control_ds);

        ss          = SweepStim(window,
                                 stimuli         = [habituation_block, glo_block, control_block], #, glo, control],
                                 pre_blank_sec   = SESSION_PARAMS['pre_blank'],
                                 post_blank_sec  = SESSION_PARAMS['post_blank'],
                                 params          = RIG_PARAMS,  # will be set by MPE to work on the rig
                                 )

    else:
        ss          = SweepStim(window,
                                 stimuli         = [habituation_block],
                                 pre_blank_sec   = SESSION_PARAMS['pre_blank'],
                                 post_blank_sec  = SESSION_PARAMS['post_blank'],
                                 params          = RIG_PARAMS,  # will be set by MPE to work on the rig
                                 )

    # add in foraging so we can track wheel, potentially give rewards, etc
    f               = Foraging(window     = window,
                                auto_update = False,
                                params      = RIG_PARAMS,
                                nidaq_tasks = {'digital_input': ss.di, 'digital_output': ss.do,},
                                )
    ss.add_item(f, "foraging")

    # run it
    ss.run()
