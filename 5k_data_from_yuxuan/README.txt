Data from Animal ICMS 148, with 3 shanks out of 4 successfully implanted. 
Each shank has 32 linearly arranged channels. 
Currents used for all stimulation pulses are 3uA. 
Duration of pulse is a 167-66-167us biphasic structure (with 66us as inter-phase-interval). 
The sampling rate is namely 30K Hz, but there're fluctuations. 
So it's recommended to just use the timestamps already accurately registered in the data files instead of directly converting to physical times as a start point.

For data file in the folder:

【Pattern_Registration.pkl】
Contains a LIST of 5K elements, each containing information of a pattern, indexed by time order. Each element is a DICT, the useful KEYS and meanings within it are:

"pattern_lambda": the \lambda value of Poisson distribution that generates this random pattern. Ranging from 0.4 to 1.2. 
"pattern_timing_index": the temporal order of this pattern in the entire experiment
"steps": a LIST containing 10 DICT containing information of every step of this pattern. Within each DICT element, the KEYS and meanings are:
-------- "channel_delays": a list either being blanck or containing a series of DICT, each having 'channel' and 'delay_mode' info.
----------------'channel': name of stimulating channels
----------------'delay_mode': can be -1, 0, 1, 2. 
----------------------  【0】: 3 pulse of 50Hz stimulation
----------------------  【1】: 3 pulse of 50Hz stimulation, but with 10ms delay compared to mode 0
---------------------- 【-1】：3 pulse of 50Hz stimulation, but with a reversed phase structure compared to mode 0
----------------------  【2】：6 pulses of 100 Hz stsimulation
-------- "index": time order index of this step within the pattern.
-------- "start_timestamp": the accurate sample timestamp at which the first stimulation pulse of this step onsets.


【SpkVecs.npy】
Contains a NumPy Array of void192. Useful FIELDS within it are "sample_index" and "unit_index".
-------- "sample_index": The timestamp when a spike onsets.
-------- "unit_index": Which neuron each spike is from. Note that the number used for the neurons are not continuous. Every 100 indexes are referring to neurons from the same shank. For this data set, there're only 3 shanks implanted, so there's no indexing from 100 to 199.


Experiment notes from YL
=================================
Idea: pre-stim effect has no effect on post-stim. So stimulation should drive it to target region regardless of initial condition. 
There are 10 single-channels + 10C2 pairs of channels plus 3 modes = 145 combinations. A Poisson spike count is theorized to choose
how many patterns, but the configurations themselves are uniformally sampled. 

The Poisson process decides how many configurations (145 possible) are chosen at each of the 10 steps. 
So for each step, 10 Poisson random numbers are generated from the same distro. 

