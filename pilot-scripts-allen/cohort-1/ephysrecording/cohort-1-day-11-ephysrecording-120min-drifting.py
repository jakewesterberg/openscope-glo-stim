"""

Title:          GlobalLocalOddballs (GLO) stimulus generation code
Author:         Jacob A. Westerberg (Vanderbilt University)
Contact:        jacob.a.westerberg@vanderbilt.edu
Git Repo:       openscope-glo-stim (westerberg-science)
Written:        2022-03-24
Updated:        2022-04-12 (Jerome Lecoq)

"""
import camstim
from psychopy import visual
from camstim import SweepStim, Stimulus, Foraging
from camstim import Window, Warp
import random
import numpy as np

"""
runs optotagging code for ecephys pipeline experiments
by joshs@alleninstitute.org, corbettb@alleninstitute.org, chrism@alleninstitute.org, jeromel@alleninstitute.org

(c) 2018 Allen Institute for Brain Science
"""    
from toolbox.IO.nidaq import AnalogOutput
from toolbox.IO.nidaq import DigitalOutput
import datetime
import time
import pickle as pkl
import argparse
import yaml
from copy import deepcopy
from camstim.misc import get_config
from camstim.zro import agent

def run_optotagging(levels, conditions, waveforms, isis, sampleRate = 10000.):
    
    from toolbox.IO.nidaq import AnalogOutput
    from toolbox.IO.nidaq import DigitalOutput

    sweep_on = np.array([0,0,1,0,0,0,0,0], dtype=np.uint8)
    stim_on = np.array([0,0,1,1,0,0,0,0], dtype=np.uint8)
    stim_off = np.array([0,0,1,0,0,0,0,0], dtype=np.uint8)
    sweep_off = np.array([0,0,0,0,0,0,0,0], dtype=np.uint8)

    ao = AnalogOutput('Dev1', channels=[1])
    ao.cfg_sample_clock(sampleRate)
    
    do = DigitalOutput('Dev1', 2)
    
    do.start()
    ao.start()
    
    do.write(sweep_on)
    time.sleep(5)
    
    for i, level in enumerate(levels):
        
        print(level)
    
        data = waveforms[conditions[i]]
    
        do.write(stim_on)
        ao.write(data * level)
        do.write(stim_off)
        time.sleep(isis[i])
        
    do.write(sweep_off)
    do.clear()
    ao.clear()
 
def generatePulseTrain(pulseWidth, pulseInterval, numRepeats, riseTime, sampleRate = 10000.):
    
    data = np.zeros((int(sampleRate),), dtype=np.float64)    
   # rise_samples =     
    
    rise_and_fall = (((1 - np.cos(np.arange(sampleRate*riseTime/1000., dtype=np.float64)*2*np.pi/10))+1)-1)/2
    half_length = rise_and_fall.size / 2
    rise = rise_and_fall[:half_length]
    fall = rise_and_fall[half_length:]
    
    peak_samples = int(sampleRate*(pulseWidth-riseTime*2)/1000)
    peak = np.ones((peak_samples,))
    
    pulse = np.concatenate((rise, \
                           peak, \
                           fall))
    
    interval = int(pulseInterval*sampleRate/1000.)
    
    for i in range(0, numRepeats):
        data[i*interval:i*interval+pulse.size] = pulse
        
    return data

def optotagging(mouse_id, operation_mode='experiment', level_list = [1.15, 1.28, 1.345], output_dir = 'C:/ProgramData/camstim/output/'):

    sampleRate = 10000

    # 1 s cosine ramp:
    data_cosine = (((1 - np.cos(np.arange(sampleRate, dtype=np.float64)
                                * 2*np.pi/sampleRate)) + 1) - 1)/2  # create raised cosine waveform

    # 1 ms cosine ramp:
    rise_and_fall = (
        ((1 - np.cos(np.arange(sampleRate*0.001, dtype=np.float64)*2*np.pi/10))+1)-1)/2
    half_length = rise_and_fall.size / 2

    # pulses with cosine ramp:
    pulse_2ms = np.concatenate((rise_and_fall[:half_length], np.ones(
        (int(sampleRate*0.001),)), rise_and_fall[half_length:]))
    pulse_5ms = np.concatenate((rise_and_fall[:half_length], np.ones(
        (int(sampleRate*0.004),)), rise_and_fall[half_length:]))
    pulse_10ms = np.concatenate((rise_and_fall[:half_length], np.ones(
        (int(sampleRate*0.009),)), rise_and_fall[half_length:]))

    data_2ms_10Hz = np.zeros((sampleRate,), dtype=np.float64)

    for i in range(0, 10):
        interval = sampleRate / 10
        data_2ms_10Hz[i*interval:i*interval+pulse_2ms.size] = pulse_2ms

    data_5ms = np.zeros((sampleRate,), dtype=np.float64)
    data_5ms[:pulse_5ms.size] = pulse_5ms

    data_10ms = np.zeros((sampleRate,), dtype=np.float64)
    data_10ms[:pulse_10ms.size] = pulse_10ms

    data_10s = np.zeros((sampleRate*10,), dtype=np.float64)
    data_10s[:-2] = 1

    # for experiment

    isi = 1.5
    isi_rand = 0.5
    numRepeats = 50

    condition_list = [2, 3]
    waveforms = [data_2ms_10Hz, data_5ms, data_10ms, data_cosine]
    
    opto_levels = np.array(level_list*numRepeats*len(condition_list)) #     BLUE
    opto_conditions = condition_list*numRepeats*len(level_list)
    opto_conditions = np.sort(opto_conditions)
    opto_isis = np.random.random(opto_levels.shape) * isi_rand + isi
    
    p = np.random.permutation(len(opto_levels))
    
    # implement shuffle?
    opto_levels = opto_levels[p]
    opto_conditions = opto_conditions[p]
    
    # for testing
    
    if operation_mode=='test_levels':
        isi = 2.0
        isi_rand = 0.0

        numRepeats = 2

        condition_list = [0]
        waveforms = [data_10s, data_10s]
        
        opto_levels = np.array(level_list*numRepeats*len(condition_list)) #     BLUE
        opto_conditions = condition_list*numRepeats*len(level_list)
        opto_conditions = np.sort(opto_conditions)
        opto_isis = np.random.random(opto_levels.shape) * isi_rand + isi

    elif operation_mode=='pretest':
        numRepeats = 1
        
        condition_list = [0]
        data_2s = data_10s[-sampleRate*2:]
        waveforms = [data_2s]
        
        opto_levels = np.array(level_list*numRepeats*len(condition_list)) #     BLUE
        opto_conditions = condition_list*numRepeats*len(level_list)
        opto_conditions = np.sort(opto_conditions)
        opto_isis = [1]*len(opto_conditions)
    # 

    outputDirectory = output_dir
    fileDate = str(datetime.datetime.now()).replace(':', '').replace(
        '.', '').replace('-', '').replace(' ', '')[2:14]
    fileName = outputDirectory + fileDate + '_'+mouse_id + '.opto.pkl'

    print('saving info to: ' + fileName)
    fl = open(fileName, 'wb')
    output = {}

    output['opto_levels'] = opto_levels
    output['opto_conditions'] = opto_conditions
    output['opto_ISIs'] = opto_isis
    output['opto_waveforms'] = waveforms

    pkl.dump(output, fl)
    fl.close()
    print('saved.')

    # 
    run_optotagging(opto_levels, opto_conditions,
                    waveforms, opto_isis, float(sampleRate))
""" 
end of optotagging section
"""

SESSION_PARAMS  = { 'subject_id':                   'test',                 # subject identifier information
                    'session_id':                   'test',                 # session identifier information
                    'cohort':                       1,                      # which orientation cohort (1, 2)
                    'habituation_duration':         60 * 4.20,              # desired habituation block duration (sec)
                    'glo_duration':                 60 * 79.2,              # desired GLO block duration (sec)
                    'randomized_control_duration':  60 * 25.2,              # desired radomized control block duration (sec)
                    'sequenced_control_duration':   60 * 8.40,              # desired sequenced control block duration (sec)
                    'pre_blank':                    10,                     # blank before stim starts (sec)
                    'post_blank':                   10,                     # blank after all stims end (sec)
                    'stimulus_orientations':        [135, 45],              # two orientations
                    'stimulus_drift_rate':          4.0,                    # stimulus drift rate (0 for static)
                    'stimulus_spatial_freq':        0.04,                   # spatial frequency of grating
                    'stimulus_duration':            0.5,                    # stimulus presentation duration (sec)
                    'stimulus_contrast':            0.8,                    # stimulus contrast (0-1)
                    'stimulus_phase':               [0.0, 0.25, 0.5, 0.75], # possible phases for gratings (0-1)
                    'interstimulus_duration':       0.5,                    # blank between all stims (sec)
                    'global_oddball_proportion':    0.2,                    # proportion of global oddball trials in GLO block (0-1)
                    'intermission_frequency':       100,                    # number of sequences between black-blank 'intermission'
                    'intermission_duration':        10,                     # duration of blank intermissions (sec)
                    'intermission_color':           -1,                     # in the event white screen appear instead of black set '1'
                    }


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

def init_grating(window, session_params, contrast, phase, tf, sf, ori, opacity):

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
                                                                'Opacity': ([opacity], 5),
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

def init_intermission(window, session_params):

        grating            = Stimulus(visual.GratingStim(window,
                                        pos                 = (0, 0),
                                        units               = 'deg',
                                        size                = (1000, 1000),
                                        mask                = "None",
                                        texRes              = 256,
                                        sf                  = 0.1,
                                        ),
                                        sweep_params        = { 'Contrast': ([session_params['intermission_color']], 0),
                                                                'Phase': ([0], 1),
                                                                'TF': ([0], 2),
                                                                'SF': ([0], 3),
                                                                'Ori': ([0], 4),
                                                                'Opacity': ([1], 5),
                                                                 },
                                        sweep_length        = session_params['intermission_duration'],
                                        start_time          = 0.0,
                                        blank_length        = 0.0,
                                        blank_sweeps        = 0,
                                        runs                = 1,
                                        shuffle             = False,
                                        save_sweep_table    = True,
                                        )

        return grating

def generate_sequence(window, session_params, in_session_time, stimulus_counter, duration, global_oddball_proportion, is_control, type_control):

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

        if type_control == 'randomized':

            patterns_sequence           = []
            sequence['orientations']    = []

            patterns_prob           = [ 3.0,  3.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  3.0,  1.0,  1.0,  1.0,]
            prob_correction         = np.sum(patterns_prob)
            patterns_prob           = patterns_prob / prob_correction

            patterns        = [ [o1, o1, o1, o1, o1],       [o1, o1, o1, o1, o2],       [o1, o1, o1, o2, o1],       [o1, o1, o1, o2, o2],
                                [o1, o1, o2, o1, o1],       [o1, o1, o2, o2, o1],       [o1, o1, o2, o1, o2],       [o1, o1, o2, o2, o2],
                                [o1, o2, o1, o1, o1],       [o1, o2, o2, o1, o1],       [o1, o2, o1, o2, o1],       [o1, o2, o1, o1, o2],
                                [o1, o2, o2, o2, o1],       [o1, o2, o2, o1, o2],       [o1, o2, o1, o2, o2],       [o1, o2, o2, o2, o2], ]

            patterns_prez           = np.floor(patterns_prob * sequence_count)

            for i in np.arange(16):
                for j in np.arange(patterns_prez[i]):
                    patterns_sequence.append(i)

            session_params['rng'].shuffle(patterns_sequence)

            for i in np.arange(np.size(patterns_sequence)):
                temp_pattern = patterns[patterns_sequence[i]]
                for j in np.arange(np.size(temp_pattern)):
                    sequence['orientations'].append(temp_pattern[j])

            if np.size(sequence['orientations']) < sequence['stimulus_count']:
                missing_seqs = (sequence['stimulus_count'] - np.size(sequence['orientations'])) / 5
                for i in np.arange(missing_seqs):
                    random_seq = int(session_params['rng'].choice(patterns_sequence, 1))
                    temp_pattern = patterns[random_seq]
                    for j in np.arange(np.size(temp_pattern)):
                        sequence['orientations'].append(temp_pattern[j])

        elif type_control == 'sequenced':

            trial_vector                = [o1, o1, o1, o1, o1]
            sequence['orientations']    = np.squeeze(np.tile(trial_vector, (1, sequence_count)))

            t_ctr = 0
            for i in np.arange(0, sequence['stimulus_count'], 1):
                t_ctr = t_ctr + 1
                if (t_ctr % 10) > 5 or t_ctr == 10:
                    sequence['orientations'][i] = o2
                    if t_ctr == 10:
                        t_ctr = 0

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

    # init intermission counts
    intermission_ctr = session_params['intermission_frequency']

    # generate the stimuli
    for i in np.arange(sequence['stimulus_count']):

        if sequence['blanks'][i] == True:

            if intermission_ctr % (session_params['intermission_frequency']) == 0:
                gratings.append(init_intermission(window, session_params,))
                gratings[stimulus_counter].set_display_sequence([(in_session_time, in_session_time
                                                                + session_params['intermission_duration'])])

                in_session_time = in_session_time + session_params['intermission_duration'];
                stimulus_counter = stimulus_counter + 1
                intermission_ctr = 1

            else:
                intermission_ctr = intermission_ctr + 1

            gratings.append(init_grating(window, session_params, sequence['contrasts'][i], 0, 0, 0, 0, 0,))

        else:

            gratings.append(init_grating(window, session_params,
                                          sequence['contrasts'][i],
                                          sequence['phases'][i],
                                          sequence['TFs'][i],
                                          sequence['SFs'][i],
                                          sequence['orientations'][i],
                                          1.0
                                          ))

        gratings[stimulus_counter].set_display_sequence([(in_session_time, in_session_time
                                                            + session_params['stimulus_duration']
                                                            + session_params['interstimulus_duration'])])

        in_session_time = in_session_time + session_params['stimulus_duration'] + session_params['interstimulus_duration'];
        stimulus_counter = stimulus_counter + 1

    return gratings, in_session_time, stimulus_counter

def create_receptive_field_mapping(number_runs = 15):
    x = np.arange(-40,45,10)
    y = np.arange(-40,45,10)
    position = []
    for i in x:
        for j in y:
            position.append([i,j])

    stimulus = Stimulus(visual.GratingStim(window,
                        units='deg',
                        size=20,
                        mask="circle",
                        texRes=256,
                        sf=0.1,
                        ),
        sweep_params={
                'Pos':(position, 0),
                'Contrast': ([0.8], 4),
                'TF': ([4.0], 1),
                'SF': ([0.08], 2),
                'Ori': ([0,45,90], 3),
                },
        sweep_length=0.25,
        start_time=0.0,
        blank_length=0.0,
        blank_sweeps=0,
        runs=number_runs,
        shuffle=True,
        save_sweep_table=True,
        )
    
    return stimulus

if __name__ == "__main__":
    
    # This part load parameters from mtrain
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", nargs="?", type=str, default="")

    args, _ = parser.parse_known_args() # <- this ensures that we ignore other arguments that might be needed by camstim
    # print args
    with open(args.json_path, 'r') as f:
        # we use the yaml package here because the json package loads as unicode, which prevents using the keys as parameters later
        json_params = yaml.load(f)
    # end of mtrain part
        
    dist = 15.0
    wid = 52.0

    # Create display window
    window = Window(fullscr=True, # Will return an error due to default size. Ignore.
                    screen=1,
                    monitor='Gamma1.Luminance50',
                    warp=Warp.Spherical
                    )

    # Init stimulus time tracking
    in_session_time = 0
    stimulus_counter = 0

    # randomly set a seed for the session and create a dictionary
    SESSION_PARAMS['seed'] = random.choice(range(0, 48000))
    # SESSION_PARAMS['seed'] = # override by setting seed manually
    SESSION_PARAMS['rng'] = np.random.RandomState(SESSION_PARAMS['seed'])

    total_time_calc              = SESSION_PARAMS['habituation_duration'] + SESSION_PARAMS['glo_duration'] + SESSION_PARAMS['randomized_control_duration'] + SESSION_PARAMS['sequenced_control_duration']
    habituation_trial_count      = int(np.floor(SESSION_PARAMS['habituation_duration'] / ( 5 * (SESSION_PARAMS['stimulus_duration'] + SESSION_PARAMS['interstimulus_duration']))))
    glo_trial_count              = int(np.floor(SESSION_PARAMS['glo_duration'] / ( 5 * (SESSION_PARAMS['stimulus_duration'] + SESSION_PARAMS['interstimulus_duration']))))
    random_control_trial_count   = int(np.floor(SESSION_PARAMS['randomized_control_duration'] / ( 5 * (SESSION_PARAMS['stimulus_duration'] + SESSION_PARAMS['interstimulus_duration']))))
    sequence_control_trial_count = int(np.floor(SESSION_PARAMS['sequenced_control_duration'] / ( 5 * (SESSION_PARAMS['stimulus_duration'] + SESSION_PARAMS['interstimulus_duration']))))
    local_oddball_count          = glo_trial_count - int(np.round(glo_trial_count * SESSION_PARAMS['global_oddball_proportion']))
    global_oddball_count         = int(np.round(glo_trial_count * SESSION_PARAMS['global_oddball_proportion']))

    # compute number of intermissions
    total_intermissions         = int((np.floor(habituation_trial_count / SESSION_PARAMS['intermission_frequency']) + (SESSION_PARAMS['habituation_duration'] > 0.0)) +
                                  (np.floor(glo_trial_count / SESSION_PARAMS['intermission_frequency']) + (SESSION_PARAMS['glo_duration'] > 0.0)) +
                                  (np.floor(random_control_trial_count / SESSION_PARAMS['intermission_frequency']) + (SESSION_PARAMS['randomized_control_duration'] > 0.0)) +
                                  (np.floor(sequence_control_trial_count / SESSION_PARAMS['intermission_frequency']) + (SESSION_PARAMS['sequenced_control_duration'] > 0.0)))
    total_intermissions_time    = total_intermissions * SESSION_PARAMS['intermission_duration']

    # recompute total time
    total_time_calc = total_time_calc + total_intermissions_time

    print('')
    print('%%%%% TASK SEQUENCE INFORMATION %%%%%')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('')
    print('Total time (min)             : ' + str(total_time_calc / 60))
    print('Habituation trial count      : ' + str(habituation_trial_count))
    print('Total GLO trial count        : ' + str(glo_trial_count))
    print('Local oddball trial count    : ' + str(local_oddball_count))
    print('Global oddball trial count   : ' + str(global_oddball_count))
    print('Random control trial count   : ' + str(random_control_trial_count))
    print('Sequence control trial count : ' + str(sequence_control_trial_count))
    print('Intermissions count          : ' + str(total_intermissions))
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
                                                                                        False,
                                                                                        False)


    # We generate the receptive field section 
    # 15 is the number of repeats (20min). 
    gabors_rf_20 = create_receptive_field_mapping(15)
    gabors_rf_20_ds = [(total_time_calc, total_time_calc+1200)]
    gabors_rf_20.set_display_sequence(gabors_rf_20_ds)

    
    # add global oddball and control blocks if an ephys session
    if SESSION_PARAMS['glo_duration'] > 0:

        SESSION_PARAMS['gratings'], in_session_time, stimulus_counter    = generate_sequence(window,
                                                                                        SESSION_PARAMS,
                                                                                        in_session_time,
                                                                                        stimulus_counter,
                                                                                        SESSION_PARAMS['glo_duration'],
                                                                                        SESSION_PARAMS['global_oddball_proportion'],
                                                                                        False,
                                                                                        False)

        SESSION_PARAMS['gratings'], in_session_time, stimulus_counter    = generate_sequence(window,
                                                                                        SESSION_PARAMS,
                                                                                        in_session_time,
                                                                                        stimulus_counter,
                                                                                        SESSION_PARAMS['randomized_control_duration'],
                                                                                        0,
                                                                                        True,
                                                                                        'randomized')

        SESSION_PARAMS['gratings'], in_session_time, stimulus_counter    = generate_sequence(window,
                                                                                        SESSION_PARAMS,
                                                                                        in_session_time,
                                                                                        stimulus_counter,
                                                                                        SESSION_PARAMS['sequenced_control_duration'],
                                                                                        0,
                                                                                        True,
                                                                                        'sequenced')

    ss  = SweepStim(window,
                            stimuli         = [ SESSION_PARAMS['gratings'], gabors_rf_20],
                            pre_blank_sec   = SESSION_PARAMS['pre_blank'],
                            post_blank_sec  = SESSION_PARAMS['post_blank'],
                            params          = {},  # will be set by MPE to work on the rig
                            )

    # add in foraging so we can track wheel, potentially give rewards, etc
    f = Foraging(window       = window,
                                auto_update = False,
                                params      = {}
                                )
    
    ss.add_item(f, "foraging")

    # run it
    ss.run()
    
    opto_disabled = not json_params.get('disable_opto', True)
    if not(opto_disabled):
        
        opto_params = deepcopy(json_params.get("opto_params"))
        opto_params["mouse_id"] = json_params["mouse_id"]
        opto_params["output_dir"] = agent.OUTPUT_DIR
        #Read opto levels from stim.cfg file
        config_path = agent.CONFIG_PATH
        stim_cfg_opto_params = get_config(
            'Optogenetics',
            path=config_path,
        )
        opto_params["level_list"] = stim_cfg_opto_params["level_list"]

        optotagging(**opto_params)
        