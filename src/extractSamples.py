import time
import os
from Bio import motifs, SeqIO
import numpy as np
import helperFunctions as hf

# CONSTANTS

# threshold for PWM search, change as needed
PWM_THRESH = 3.0

# list of chromosomes to check on for each transcription factor
# add any chromosomes you'd like to check to this list
# NB: they must exist in the hg19 folder
CHRS = ['chr21']

# folder locations
curr = os.path.dirname(__file__)
DATA_FOLDER = os.path.join(curr, "../data/")
OUT_FOLDER = os.path.join(curr, "../outputs/")
CHR_FOLDER = DATA_FOLDER + "hg19/"

# DATA_FOLDER = "../data/"
# CHR_FOLDER = DATA_FOLDER + "hg19/"
# OUT_FOLDER = "../outputs/"

# list of transcription factors to extract samples for
# add any TF's you'd like to extract samples for to this list
# my project focused on the UAK42 transcription factor, hence its inclusion here
tfs = ['UAK42']

if (__name__=="__main__"):
    # write Jaspar files for the PWM of the chosen transcription factors
    for TF in tfs:
        hf.writePWMJaspar(TF)

    # logging
    now = hf.getTime()
    print(f"Started at {now.strftime(f'%H:%M')} on {now.strftime(f'%Y-%m-%d')}")

    # create subfolder for outputs
    # (outputs will be FASTA files containing positive & negative samples)
    OUT_SUBFOLDER = OUT_FOLDER + now.strftime(f"%Y-%m-%d_%H%M_FASTA_files/")
    if not os.path.exists(OUT_SUBFOLDER):
        os.mkdir(OUT_SUBFOLDER)

    # tracking time
    total_start = time.time()

    # loop through chosen transcription factors and extract samples
    for TF in tfs:
        # time tracking
        subtotal_start = time.time()

        # setting up text files to hold extracted samples
        now = hf.getTime()
        pos_path = OUT_SUBFOLDER + now.strftime(f"%Y-%m-%d_%H%M_{TF}_positive.txt")
        neg_path = OUT_SUBFOLDER + now.strftime(f"%Y-%m-%d_%H%M_{TF}_negative.txt")
        pos_tf = hf.TextWriter(pos_path)
        neg_tf = hf.TextWriter(neg_path)

        # logging
        print("-------------------------------------------------")
        print(f"TF: {TF}\n")

        # load the PWN for the current TF from jaspar file
        jaspar_file = DATA_FOLDER + f"/jaspars/{TF}_jaspar.pfm"
        with open(jaspar_file) as handle:
            motif = motifs.read(handle, 'jaspar')

        # variables to track number of samples extracted
        total_pos, total_neg = 0, 0

        # variables to track labels in the output FASTA file
        pos_label, neg_label = 1, 1

        # search on every chromosome in list
        for chromosome in CHRS:
            # time tracking
            chr_start = time.time()

            # get DNA sequence from chromosome FASTA file
            seq_file = CHR_FOLDER + f"{chromosome}.fa"
            sequence = list(SeqIO.parse(seq_file, "fasta"))[0].seq

            # get active regions for chromosome
            active_regions = hf.getActiveRegions(chromosome)

            # logging
            print(f"{chromosome}: {len(sequence):,} nucleotides ------------------------")
            print(f"Searching for potential {TF} binding sites on {chromosome} active regions...")

            # use PWM to search for potential binding sites on active regions
            positions = []
            scores = []
            t_start = time.time()
            for region in active_regions:
                for pos, score in motif.pssm.search(sequence[region[0]:region[1]-len(motif)], threshold=PWM_THRESH):
                    # only consider motifs found on the positive strand
                    if (pos > 0):
                        positions.append(region[0]+pos)
                        scores.append(score)
            t_end = time.time()

            # logging
            print(f"Done in {hf.stringTime(t_start, t_end)}\n")
            print(f"{len(positions):,} potential TFBS's found on {chromosome} active regions")
            if (len(positions) != 0):
                print(f"Highest score: {max(scores):.3f} at position {positions[np.argmax(scores)]:,} giving sequence {sequence[positions[np.argmax(scores)]:positions[np.argmax(scores)]+len(motif)].upper()}\n")

            # get locations of bound binding sites from factorbookMotifPos file
            bs_locs = hf.getBindingSiteLocs(TF, chromosome)
            bs_start_pos = [x[0] for x in bs_locs]
            print(f"{len(bs_locs):,} bound {TF} TFBS's on {chromosome} found in factorbookMotifPos\n")

            # extracting positive samples
            # (cases where a potential binding site from above is found in the factorbookMotifPos file)
            print("Getting positive samples...")
            t_start = time.time()
            num_matches = 0 # counting positive samples to make sure we match with # of negative samples
            match_starts = []
            for pos in positions:
                if (pos in bs_start_pos):
                    num_matches += 1
                    match_starts.append(pos)
                    pos_tf.writeTxt(f">{pos_label}", print_console=False)
                    pos_tf.writeTxt(str(sequence[pos:pos+len(motif)].upper()), print_console=False)
                    pos_label += 1
            t_end = time.time()
            print(f"Extracted {len(match_starts)} positive samples in {hf.stringTime(t_start, t_end)}\n")
            total_pos += len(match_starts)

            # extracting negative samples
            # (cases where a potential binding site from above is NOT found in the factorbookMotifPos file)
            print("Getting negative samples...")
            t_start = time.time()
            j = 0
            mismatch_starts = []
            for pos in positions:
                if (j == num_matches):
                    break # for equal number of negative + positive samples
                if (pos not in bs_start_pos):
                    mismatch_starts.append(pos)
                    neg_tf.writeTxt(f">{neg_label}", print_console=False)
                    neg_tf.writeTxt(str(sequence[pos:pos+len(motif)].upper()), print_console=False)
                    neg_label += 1
                    j += 1
            t_end = time.time()
            print(f"Extracted {len(mismatch_starts)} negative samples in {hf.stringTime(t_start, t_end)}\n")
            total_neg += len(mismatch_starts)

            # logging time for current chromosome
            chr_end = time.time()
            print(f"Total time for {chromosome}: {hf.stringTime(chr_start, chr_end)}\n")

        # logging time for current Tf
        subtotal_end = time.time()
        print("##########################")
        print(f"Total time for {TF}: {hf.stringTime(subtotal_start, subtotal_end)}\n")
        print(f"Number of {TF} positive samples: {total_pos:,}")
        print(f"Number of {TF} negative samples: {total_neg:,}\n")

    # logging total time
    total_end = time.time()
    print("\n----------------------------------------------------")
    print(f"Total time elapsed: {hf.stringTime(total_start, total_end)}\n")





