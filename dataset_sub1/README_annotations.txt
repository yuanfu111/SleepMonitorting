%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             README -- CASE_dataset/interpolated/annotations
%
% This short guide to the interpolated data, covers the following topics:
% (1) General Information.
% (2) Extra Information.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

--------------------------------------------------------------------------------
(1) General Information:
--------------------------------------------------------------------------------
This folder contains the interpolated annotation data for all 30 subjects.
Each file (e.g., sub_1.csv) contains the following 4 comma-separated variables
(1 variable per column):

(1) jstime: is the time provided by LabVIEW while logging. It is the global
    time and is also used for physiological files to allow synchronization of
    data across the two files. It is named jstime to keep the variable name
    different from daqtime (used for physiological data). Measurements in
    milliseconds (ms).
(2) valence: values in interval [0.5 9.5].
(3) arousal: values in interval [0.5 9.5].
(4) video: the video-IDs, that are repeated for the entire duration that a
           video appeared in the video-sequence for that participant.

--------------------------------------------------------------------------------
(2) Extra Information:
--------------------------------------------------------------------------------
The joystick annotation data logged in the raw files is in the integer range
from -26225 to +26225 for both the axes. These are then converted to the
interval [0.5 9.5] in pre-processing. More information on this is available in
the data descriptor.

PLEASE NOTE: The annotation data in this folder is interpolated. Please see
the README file one folder up for more information on this topic.
