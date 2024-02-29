import os
from glob import glob
from utils.mdp.process_io import create_dir
from multiprocessing.pool import Pool
from tqdm import tqdm
import miditoolkit
import math

def process_seg_job(midi_path, dst_dir):
    # -----------------------------------
    # 1） calculate segments
    # -----------------------------------
    segment_note_dict = {}
    min_bar = 8
    dis_empty_bar = 4
    try:
        midi = miditoolkit.MidiFile(midi_path)
    except:
        return None
    midi_tracks = midi.instruments
    bar_resolution = midi.ticks_per_beat * 4
    melody_notes = None
    for ins in midi_tracks:
        if ins.name == "Lead":
            melody_notes = ins.notes
            break

    num_segment = 1
    start_segment = 0
    Full_Song_Flag = True
    try:
        if len(melody_notes) > 0:
            for note_idx, note in enumerate(melody_notes):
                if note_idx == 0:
                    # start_segment = note.start
                    bar_id = int(note.start / bar_resolution)
                    start_segment = bar_id * bar_resolution
                else:
                    pre_bar = (int((melody_notes[note_idx - 1].end-1) / bar_resolution))  # end-1: ending is open interval.
                    cur_bar = int(note.start / bar_resolution)
                    bar_dis = cur_bar - pre_bar - 1

                    if bar_dis > dis_empty_bar:
                        end_segment = (math.ceil(melody_notes[note_idx - 1].end / bar_resolution)) * bar_resolution
                        segment_length = int((end_segment - start_segment) / bar_resolution)
                        Full_Song_Flag = False
                        if segment_length >= min_bar:
                            seg_key_name = f"Seg{num_segment}"
                            segment_note_dict[seg_key_name] = [start_segment, end_segment, segment_length]
                            num_segment += 1
                        # new segment
                        start_segment = int(note.start / bar_resolution) * bar_resolution

                    # last note
                    if not Full_Song_Flag and note_idx == len(melody_notes) - 1: 
                        end_segment = (math.ceil(note.end / bar_resolution)) * bar_resolution
                        segment_length = int((end_segment - start_segment) / bar_resolution)
                        if segment_length >= min_bar:
                            seg_key_name = f"Seg{num_segment}"
                            segment_note_dict[seg_key_name] = [start_segment, end_segment, segment_length]
                            num_segment += 1

            if Full_Song_Flag:
                seg_key_name = f"Seg{num_segment}"
                start_segment = int(melody_notes[0].start / bar_resolution) * bar_resolution
                end_segment = (int(melody_notes[-1].end / bar_resolution) + 1) * bar_resolution
                segment_length = int((end_segment - start_segment) / bar_resolution)
                segment_note_dict[seg_key_name] = [start_segment, end_segment, segment_length]

            # print(segment_note_dict)

            # -----------------------------------
            # 2） segment and save
            # -----------------------------------
            if len(segment_note_dict) >= 1:
                for seg_name, seg in segment_note_dict.items():
                    seg_start, seg_end = seg[0], seg[1]
                    # print(f"the segment start from {seg_start}, end at {seg_end}")
                    temp_midi = miditoolkit.MidiFile(midi_path)
                    # markers
                    markers = temp_midi.markers
                    # print(f"Orginal Markers = {markers}")
                    new_markers = []
                    for idx, marker in enumerate(markers):

                        if idx != len(markers) - 1:
                            marker_start = marker.time
                            marker_end = markers[idx + 1].time
                            # print(f"idx = {idx}, marker = {marker}, marker start = {marker_start}, marker end = {marker_end}")
                            if marker_start < seg_start and marker_end > seg_start:
                                marker.time = 0
                                new_markers.append(marker)
                            elif marker_start >= seg_start and marker_end <= seg_end:
                                marker.time = marker.time - seg_start
                                new_markers.append(marker)
                                # print("in, ",new_markers)
                            elif marker_start < seg_end and marker_end >= seg_end:
                                marker.time = marker.time - seg_start
                                new_markers.append(marker)
                        else:
                            marker_start = marker.time
                            marker_end = temp_midi.max_tick
                            if marker_start < seg_start and marker_end > seg_start:
                                marker.time = 0
                                new_markers.append(marker)
                            elif marker_start >= seg_start and marker_end <= seg_end:
                                marker.time = marker.time - seg_start
                                new_markers.append(marker)
                            elif marker_start < seg_end and marker_end >= seg_end:
                                marker.time = marker.time - seg_start
                                new_markers.append(marker)

                    temp_midi.markers.clear()
                    temp_midi.markers.extend(new_markers)
                    # print(f"new_markers = {new_markers}")

                    # track
                    temp_ins = []
                    for ins in temp_midi.instruments:
                        new_ins = miditoolkit.Instrument(program=ins.program, is_drum=ins.is_drum, name=ins.name)
                        for idx, note in enumerate(ins.notes):
                            # noise, 
                            if (idx == len(ins.notes)-1) and (note.start - ins.notes[idx-1].end > 240):
                                break

                            if seg_start <= note.start <= seg_end:
                                note.start = note.start - seg_start
                                note.end = note.end - seg_start
                                new_ins.notes.append(note)
                        temp_ins.append(new_ins)
                    



                    temp_midi.instruments.clear()
                    temp_midi.instruments.extend(temp_ins)

                    # max_ticks
                    temp_midi.max_tick = seg_end - seg_start
                    new_midi_path = os.path.join(dst_dir, os.path.basename(midi_path)[:-4] + f"_{seg_name}.mid")
                    temp_midi.dump(new_midi_path)
    except Exception as e:
        print(e)

    


# main function
def process_seg(src_dir, dst_dir):
    # collect midis
    midis_list = glob(f"{src_dir}/**/*.mid", recursive=True)
    dataset_name = src_dir.split("/")[-1]
    print(f"{dataset_name} = {len(midis_list)} Songs")

    # create dir 
    create_dir(dst_dir)

    # multiprocessing
    pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    futures = [pool.apply_async(process_seg_job, args=[
        midi_path, dst_dir
    ]) for midi_path in midis_list]
    pool.close()
    progress = [x.get() for x in tqdm(futures)]
    pool.join()

    # Test 
    # for midi_path in midis_list:
    #     process_seg_job(midi_path, dst_dir)