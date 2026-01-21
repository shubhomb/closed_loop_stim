Data from Animal ICMS 150, with 4 shanks out of 4 successfully implanted. Each shank has 32 linearly arranged channels. Currents used for all stimulation pulses are 3uA. Duration of pulse is a 167-66-167us biphasic structure (with 66us as inter-phase-interval). The sampling rate is namely 30K Hz, but there're fluctuations. So it's recommended to just use the timestamps already accurately registered in the data files instead of directly converting to physical times as a start point.

For data file in the folder:

【Combined_Pattern_Registration.pkl】
Contains a LIST of 5K elements, each containing information of a pattern, indexed by time order. ID (name) of SAMPLE patterns are from 1 to 4000. ORACLE trials contains 50 different patterns (also different from those in sample trials), each repeated for 20 times (1K elements in total). The id (name) of ORACLE patterns are from 4001 to 4050. Each element is a DICT, the useful KEYS and meanings within it are:

"pattern_lambda": the \lambda value of Poisson distribution that generates this random pattern. All equals to 1
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

【Recommendations】
For this dataset, we suggest remove neuron with id of : [19, 104, 201, 211, 300, 310, 311, 312]
These neurons, while having a reasonable waveform shapes, showed extremely high firing rate (>200Hz) during stimulation, which is unlikely to happen given the refractory period of a neuron. There are two possibilities: first, they're multi-neuron unit, not single neuron. Second, they're actually artifacts. We leave the choice to you to decide.