import os
from glob import glob
from utils.mdp.process_io import create_dir
from multiprocessing.pool import Pool
from tqdm import tqdm
import miditoolkit
import numpy as np
import pretty_midi
from itertools import groupby

bar_per_ticks = 1920

def caL_valid_bars(notes, max_bar, max_ticks):    
    # Initialize the bar matrix
    valid_bar_matrix = np.zeros(max_bar)

    # Check valid bars
    for note in notes:
        idx = int(note.start / bar_per_ticks)
        valid_bar_matrix[idx] = 1
    
    # Calculate the number and percentage of valid bars
    valid_bar = int(np.sum(valid_bar_matrix))
    valid_bar_percent = valid_bar / max_bar
    
    return valid_bar, valid_bar_percent




def cal_max_continue_pitch(notes):
    '''
    The groupby function will be grouped according to successive identical pitches (groupby函数将会按照连续的相同音高进行分组)
    Then use the generator expression (generator expression) to generate the length of each packet (然后使用生成器表达式（generator expression）来生成每个分组的长度)
    Finally, use the max function to get the maximum length (最后使用max函数来获取最大长度。)
    '''
    pitch_list = [n.pitch for n in notes]
    return max(len(list(group)) for key, group in groupby(pitch_list))


def cal_precision_percent(notes, precision):
    notes_list = [note for note in notes if (note.end - note.start) == precision]
    return len(notes_list)/len(notes)
    
def cal_empty_ratio(notes, max_ticks):
    valid_length = sum([note.end-note.start for note in notes])
    return (max_ticks-valid_length)/max_ticks

def statistic(midi_path):
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
        time = pm.instruments[0].notes[-1].end
        tempo = pm.get_tempo_changes()

        midi = miditoolkit.MidiFile(midi_path)
        notes = midi.instruments[0].notes
        max_ticks = notes[-1].end

        # chord number 
        num_chords = len(midi.markers)

        # max bars
        max_bar = int(max_ticks/1920)+1

        # note count
        notes_num = len(notes)

        # pitch
        pitch_list = set([note.pitch for note in notes])
        pitch_count = len(pitch_list)
        pitch_range = max(pitch_list) - min(pitch_list)
        pitch_class = len(set([note.pitch%12 for note in notes]))

        # duration
        duration_class = len(set([note.end - note.start for note in notes]))
        

        # valid bars without empty
        valid_bar, valid_bar_percent = caL_valid_bars(notes, max_bar, max_ticks)

        # low 16 percent
        low_than_16_percent_note = len([note for note in notes if (note.end-note.start) < 120]) / notes_num

        # 64 percent
        percent_64_note = cal_precision_percent(notes, precision=30)

        # max continue same pitch
        max_time_note = cal_max_continue_pitch(notes=notes)

        # empty length
        empty_ratio = cal_empty_ratio(notes, max_ticks)


        info_dict = {
            'midi_path': midi_path,
            'max_bars': max_bar,
            'valid_bars': valid_bar,
            'valid_bar_percent': valid_bar_percent,
            'num_chords': num_chords,
            'tempo': tempo,
            'time_length': time,
            'note_count': notes_num,
            'pitch_count': pitch_count,
            'pitch_range':pitch_range,
            'pitch_class': pitch_class,
            "duration_class": duration_class,
            "percent_64": percent_64_note,
            "low_16": low_than_16_percent_note,
            "max_time_note": max_time_note,
            "empty_ratio": empty_ratio,
        }

        return info_dict
    except Exception as e:
        return None


'''
reuqirments：
    1）主旋律的音符数量大于24个 (>=8 bars * 3 notes)
    2）有效小节数量大于8 bars
    3) 有效小节比例大于 75%
    # 4）64分音符的比例低于10%
'''


def process_quality_job(midi_path, dst_dir):
    midi_melody = miditoolkit.MidiFile(midi_path)
    melody_notes = midi_melody.instruments[0].notes
    melody_notes_num = len(melody_notes)

    info = statistic(midi_path)
    if info == None:
        return None
    '''
    MelodyGLM:
    + info['valid_bars'] >= 8 and \
    '''
    if melody_notes_num >= 24 and \
            info['max_bars'] >=32 and \
            info['num_chords'] >=16 and \
            info['valid_bar_percent'] >= 0.70 and \
            info['pitch_class'] >= 5 and \
            info['duration_class'] >= 4 and \
            info['max_time_note'] <= 15 and \
            info['percent_64'] <= 0.2 and \
            info['empty_ratio'] < 0.25:
        midi_melody.dump(f"{dst_dir}/{os.path.basename(midi_path)}")


# main function
def process_quality(src_dir, dst_dir):
    # collect midis
    midis_list = glob(f"{src_dir}/**/*.mid", recursive=True)
    dataset_name = src_dir.split("/")[-1]
    print(f"{dataset_name} = {len(midis_list)} Songs")

    # create dir 
    create_dir(dst_dir)

    # multiprocessing
    pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    futures = [pool.apply_async(process_quality_job, args=[
        midi_path, dst_dir
    ]) for midi_path in midis_list]
    pool.close()
    progress = [x.get() for x in tqdm(futures)]
    pool.join()

    files = glob(f"{dst_dir}/**/*.mid", recursive=True)
    print(f"Satisfied Files Count = {len(files)}")

    # Test 
    # for midi_path in midis_list:
    #     process_normalization_job(midi_path, dst_dir)