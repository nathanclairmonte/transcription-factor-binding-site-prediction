import math
import datetime
import pytz
import os
import re
import numpy as np

# CONSTANTS
curr = os.path.dirname(__file__)
DATA_FOLDER = os.path.join(curr, "../data/")

# function to help print time elapsed
def stringTime(start, end, show_ms=False):
    """
    Formats a given number of seconds into hours, minutes, seconds and milliseconds.

    Args:
        start (float): The start time (in seconds)
        end (float): The end time (in seconds)

    Returns:
        t (str): The time elapsed between start and end, formatted nicely.
    """
    h = "{0:.0f}".format((end-start)//3600)
    m = "{0:.0f}".format(((end-start)%3600)//60)
    s = "{0:.0f}".format(math.floor(((end-start)%3600)%60))
    ms = "{0:.2f}".format((((end-start)%3600)%60 - math.floor(((end-start)%3600)%60))*1000) # remember s = math.floor(((end-start)%3600)%60
    h_str = f"{h} hour{'' if float(h)==1 else 's'}"
    m_str = f"{'' if float(h)==0 else ', '}{m} minute{'' if float(m)==1 else 's'}"
    s_str = f"{'' if (float(h)==0 and float(m)==0) else ', '}{s} second{'' if float(s)==1 else 's'}"
    ms_str = f"{'' if (float(h)==0 and float(m)==0 and float(s)==0) else ', '}{ms} ms"

    t = f"{h_str if float(h) != 0 else ''}{m_str if float(m) != 0 else ''}{s_str if float(s) != 0 else ''}{ms_str if show_ms else ''}"
    return t

# get current time
def getTime(timezone="Canada/Eastern"):
    """
    Creates a 'now' object containing information about the current time.
    Used for logging/saving files.

    Returns:
        now (utc timezone object): Object representing the current date & time
    """
    utc_now = pytz.utc.localize(datetime.datetime.utcnow())
    now = utc_now.astimezone(pytz.timezone(timezone))
    return now

# define class to help with text logging
class TextWriter:
    """
    Class that creates an object which can be used for logging to text files.
    Constructor takes a filepath (including filename) to the desired text file.
    Will overwrite file if it already exists.
    Note: also works for non-txt files that still utilise text (such as .fa, .pfm etc)
    """
    def __init__(self, filepath):
        # Define filepath, overwrite file if it already exists
        self.filepath = filepath
        if(os.path.isfile(self.filepath)):
            with open(self.filepath, "w") as txt_file:
                txt_file.write("")
    

    def writeTxt(self, string, print_console=True):
        with open(self.filepath, "a") as txt_file:
            txt_file.write(string + "\n")
            if(print_console): print(string)

def loadRaw(file):
    """
    Loads a raw text-like file into a list of its lines

    Args:
        file (str): Filepath of the file to be loaded

    Returns:
        raw (lis): List of strings corresponding to the lines in the file
    """
    with open(file) as f:
        raw = f.readlines()
    return raw

def removeLineBreaks(raw):
    """
    Removes line breaks in a raw list of strings from a text-like file loaded with loadRaw()
    """
    for i in range(len(raw)):
        raw[i] = re.sub(r'\n', '', raw[i])
    return raw

def getBindingSiteLocs(TF, chromosome='chr21'):
    """
    Takes a specific transcription factor (TF) and name of a chromosome sequence and returns the binding site 
    locations associated with them. Locations taken from 'factorbookMotifPos.txt'.
    Note: Currently only considers the positive strand
    Args:
        TF (str): Name of the TF you want TFBS locations for. Must match the name in the file exactly.
        chromosome (str): Name of the chromosome you want to get locations for. Must match name in the file exactly. Default is 'chr21'.

    Returns:
        bs_locs (list): List of TFBS locations as [start, end] arrays.
    """
    bs_pos_file = DATA_FOLDER + 'factorbookMotifPos-mini.txt'
    bs_pos_raw = loadRaw(bs_pos_file)
    bs_locs = []
    for line in bs_pos_raw:
        if (TF in line and chromosome in line and '+' in line):
            temp = line.split()
            bs_locs.append((int(temp[2]), int(temp[3])))

    return bs_locs

def getPWM(TF):
    """
    Takes a desired transcription factor (TF) and returns its corresponding PWM from the factorbookMotifPwm file

    Args:
        TF (str): The desired TF to extract a PWM for

    Returns:
        (np.array): The PWM for the given TF
    """
    pwm_file = DATA_FOLDER + 'factorbookMotifPwm.txt'
    pwm_raw = loadRaw(pwm_file)
    for line in pwm_raw:
        if (line.split()[0]==TF):
            numbers = []
            for t in line.split()[2:]:
                numbers += t.split(',')

    rows = [[]]
    j = 0
    for i in range(len(numbers)):
        if (numbers[i] != ''):
            rows[j].append(float(numbers[i]))
        elif(i==len(numbers)-1):
            continue
        else:
            rows.append([])
            j += 1
            
    return np.asarray(rows)

def writePWMJaspar(TF):
    """
    Takes a desired transcription factor (TF), gets its PWM using getPWM() and then writes it to a .pfm file
    in JASPAR format. This is so that BioPython can read it from the JASPAR format .pfm file.

    Args:
        TF (str): Desired TF for which to extract its PWM and create a Jaspar file 
    """
    # define filepath
    filepath = DATA_FOLDER + f"/jaspars/{TF}_jaspar.pfm"

    if(not os.path.isfile(filepath)):
        # nucleotides
        nts = ['A', 'C', 'G', 'T']

        # get pwm for the given TF
        pwm = getPWM(TF)

        # create file
        tf = TextWriter(filepath)

        # header line
        tf.writeTxt(f">{TF}", print_console=False)

        # write matrix to file (also prints)
        for i in range(pwm.shape[0]):
            tf.writeTxt(f"{nts[i]} [ {' '.join([str(x) for x in pwm[i, :]])} ]", print_console=False)

def getActiveRegions(chromosome='chr21'):
    """
    Takes a chromosome as a string and extracts its active regions from the list of active regions.

    Args:
        chromosome (str): The name of the chromosome to get active regions for

    Returns:
        active_regions (list): List of start and end locations of the active regions for the given chromosome
    """
    regions_file = DATA_FOLDER + "wgEncodeRegTfbsClusteredV3.GM12878.merged.bed"
    regions_raw = loadRaw(regions_file)

    active_regions = []
    for line in regions_raw:
        if (chromosome in line):
            temp = line.split()
            active_regions.append([int(temp[1]), int(temp[2])])
    
    return active_regions