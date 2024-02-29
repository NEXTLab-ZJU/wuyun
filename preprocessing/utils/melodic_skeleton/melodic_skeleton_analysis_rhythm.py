''' __authors__: Xinda Wu* and Tieyao Zhang*'''
import miditoolkit
from itertools import chain

# class Note:
#     def __init__(self, start, end, pitch, velocity, index, priority=4): # 4 = prolongation note 装饰音
#         self.start = start
#         self.end = end
#         self.pitch = pitch
#         self.velocity = velocity
#         self.index = index
#         self.priority = priority


class Rhythm_Skeleton:
    def __init__(self) -> None:
        ''' (Default) Time Signature = 4/4 '''
        self.bar_ticks = 1920
        self.default_resolution = 480
        self.ticks_per_beat = 480
        self.cell = 480 * 4 / 16        # 16th note
    
    def group_bars(self, notes) -> dict:
        '''group melodic notes by bar'''

        bar_dict = dict()
        for note in notes:
            start = note.start
            end = note.end
            duration = end - start
            # The duration of melodic skeleton notes shoud be longer than a 16th note.
            # Syncopation notes with a duration less than 16th notes is relatively rare.)
            if duration >= self.cell:
                bar_id = int(start // self.bar_ticks)
                if bar_id not in bar_dict:
                    bar_dict[bar_id] = []
                bar_dict[bar_id].append(note)
        return bar_dict
    
    # syncopation
    def cal_syncopation(self, bar_dict):
        '''
        # 切分音识别: 
        # 1. 音符的开始与结束位置: 音符起于弱拍位置开始处，结束于强拍内（不含强拍开始位置）且需要超过强拍时值的一半。
        # 2. 不同音符的粒度：在4分音符，8分音符和16分音符为网格单元下进行切分音识别。
        # ---------------------------------------------------------------------------------------------------------
        # Syncopation Recognition:
        # 1. Onsets and offsets: notes start at the beginning of the weak beat position, end within the strong beat
        #                       (excluding the strong beat start position) and need to be more than half the duration of the strong beat.
        # 2. Granularity: Syncopation is recognized on a grid of 4th, 8th and 16th notes, respectively.
        # ---------------------------------------------------------------------------------------------------------
        '''

        sync_dict = dict()      # syncopation result
        sync_dict_4 = dict()    # unit test
        sync_dict_8 = dict()    # unit test
        sync_dict_16 = dict()   # unit test

        grid16 = self.cell
        for bar_id, bar_notes in bar_dict.items():
            if bar_id not in sync_dict:
                sync_dict[bar_id] = []
                sync_dict_4[f'{bar_id}'] = []
                sync_dict_8[f'{bar_id}'] = []
                sync_dict_16[f'{bar_id}'] = []

            start = self.bar_ticks * bar_id
            note_start_4 = [4 * grid16 + start, 12 * grid16 + start]
            note_start_8 = [i * grid16 + start for i in range(2, 16, 4)]
            note_start_16 = [i * grid16 + start for i in range(1, 16, 2)]

            for note in bar_notes:
                # 4th note
                if (note.start == note_start_4[0]) and (note.end > (8 * grid16 + start)):
                    sync_dict[bar_id].append(note)
                    sync_dict_4[f'{bar_id}'].append(note)
                elif (note.start == note_start_4[1]) and (note.end > (16 * grid16 + start)):
                    sync_dict[bar_id].append(note)
                    sync_dict_4[f'{bar_id}'].append(note)
                # 8th note
                elif (note.start == note_start_8[0]) and (note.end > 4 * grid16 + start):
                    sync_dict[bar_id].append(note)
                    sync_dict_8[f'{bar_id}'].append(note)
                elif (note.start == note_start_8[1]) and (note.end > 8 * grid16 + start):
                    sync_dict[bar_id].append(note)
                    sync_dict_8[f'{bar_id}'].append(note)
                elif (note.start == note_start_8[2]) and (note.end > 12 * grid16 + start):
                    sync_dict[bar_id].append(note)
                    sync_dict_8[f'{bar_id}'].append(note)
                elif (note.start == note_start_8[3]) and (note.end > 16 * grid16 + start):
                    sync_dict[bar_id].append(note)
                    sync_dict_8[f'{bar_id}'].append(note)
                # 16th note
                elif (note.start == note_start_16[0]) and (note.end > 2 * grid16 + start):
                    sync_dict[bar_id].append(note)
                    sync_dict_16[f'{bar_id}'].append(note)
                elif (note.start == note_start_16[1]) and (note.end > 4 * grid16 + start):
                    sync_dict[bar_id].append(note)
                    sync_dict_16[f'{bar_id}'].append(note)
                elif (note.start == note_start_16[2]) and (note.end > 6 * grid16 + start):
                    sync_dict[bar_id].append(note)
                    sync_dict_16[f'{bar_id}'].append(note)
                elif (note.start == note_start_16[3]) and (note.end > 8 * grid16 + start):
                    sync_dict[bar_id].append(note)
                    sync_dict_16[f'{bar_id}'].append(note)
                elif (note.start == note_start_16[4]) and (note.end > 10 * grid16 + start):
                    sync_dict[bar_id].append(note)
                    sync_dict_16[f'{bar_id}'].append(note)
                elif (note.start == note_start_16[5]) and (note.end > 12 * grid16 + start):
                    sync_dict[bar_id].append(note)
                    sync_dict_16[f'{bar_id}'].append(note)
                elif (note.start == note_start_16[6]) and (note.end > 14 * grid16 + start):
                    sync_dict[bar_id].append(note)
                    sync_dict_16[f'{bar_id}'].append(note)
                elif (note.start == note_start_16[7]) and (note.end > 16 * grid16 + start):
                    sync_dict[bar_id].append(note)
                    sync_dict_16[f'{bar_id}'].append(note)
                else:
                    continue
        return sync_dict

    # downbeat
    def cal_downbeat(self, bar_dict):
        '''
        # 节拍重音识别：
        # 1. 在4/4拍的MIDI中，节拍重音起始于每小节的第一拍或者第三拍开始位置。
        # 2. 去除无效节拍重音。当切分音出现在节拍重音结束后的3个16分音符的时值内，则节拍重音转变为普通音符。
        # --------------------------------------------------------------------------------------------------------------
        # Beat Accent Recognition:
        # 1. In MIDI in 4/4 time, the beat accent starts at the first or third beat of each measure.
        # 2. Remove invalid beat accents. When a syncopated note occurs within 3 16th note time values of the end of a beat accent, the beat accent is transformed to a normal note.
        '''
        beat_dict = dict()
        beat_dict_clean = dict()
        sync_dict = self.cal_syncopation(bar_dict)

        # basic beat accent
        for bar_id, bar_notes in bar_dict.items():
            start = self.bar_ticks * (bar_id)
            first_beat_position = start
            third_beat_postion = start + 8 * self.cell
            if bar_id not in beat_dict:
                beat_dict[bar_id] = []
            for note in bar_notes:
                if (note.start == first_beat_position) or (note.start == third_beat_postion):
                    beat_dict[bar_id].append(note)

        # 2 filter invalid beat accent
        for beat_bar_id, beat_bar_notes in beat_dict.items():
            if beat_bar_id not in beat_dict_clean:
                beat_dict_clean[beat_bar_id] = []

            for beat_note in beat_bar_notes:
                beat_note_flag = True

                for sync_bar_id, sync_bar_notes in sync_dict.items():
                    if sync_bar_id < beat_bar_id:
                        continue

                    for sync_note in sync_bar_notes:
                        pos_delta = sync_note.start - beat_note.start
                        if 0 <= pos_delta < 3 * self.cell:
                            beat_note_flag = False
                            break

                    if beat_note_flag == False:
                        break

                if beat_note_flag:
                    beat_dict_clean[beat_bar_id].append(beat_note)

        return beat_dict_clean

    # long note
    def cal_long(self, bar_dict):
        '''
        长音识别: 
        1.长音为每小节中音符时值最长的音符。
        2.若存在多个长音，只保留第一个出现的长音。
        3.若所有音符时长相同，跳过。
        # --------------------------------------------------------------------------------------------------------------
        Long Note Recognition: 
        1. the long note is the note with the longest value in each bar.
        2. if there are multiple long notes, only the first long note is retained.
        3. if all notes have the same duration, skip them.
        '''

        long_dict = {}
        for bar_id, bar_notes in bar_dict.items():
            long_dict[bar_id] = []

            duration_list = [x.end - x.start for x in bar_notes]
            max_duration = max(duration_list)
            max_duration_count = duration_list.count(max_duration)

            idx_list = []
            if 1 <= max_duration_count < len(duration_list):
                idx_list = [i for i, duration in enumerate(duration_list) if duration == max_duration]
                idx_list = idx_list[:1]         

            for idx in idx_list:
                long_dict[bar_id].append(bar_notes[idx])

        return long_dict

    # rhythm skeleton
    def cal_rhythm(self, bar_dict):
        beat_dict = self.cal_downbeat(bar_dict)
        long_dict = self.cal_long(bar_dict)
        sync_dict = self.cal_syncopation(bar_dict)
    
        downbeat_list = list(chain(*beat_dict.values()))
        long_list = list(chain(*long_dict.values()))
        sync_list = list(chain(*sync_dict.values()))

        skeleton_dict = dict()
        for bar_id, bar_notes in bar_dict.items():
            if bar_id not in skeleton_dict:
                skeleton_dict[bar_id] = []

            for note in bar_notes:
                # 1. Downbeat | Beat Accent
                if ((note in downbeat_list) and (note not in long_list) and (note not in sync_list)):
                    skeleton_dict[bar_id].append(note)
                # 2. Downbeat & Long Note
                elif ((note in downbeat_list) and (note in long_list) and (note not in sync_list)):
                    skeleton_dict[bar_id].append(note)
                # 3. Sync & Long Note
                elif ((note not in downbeat_list) and (note in long_list) and (note in sync_list)):
                    skeleton_dict[bar_id].append(note)

        return skeleton_dict

    
    def extract_rhythm_skeleton(self, midi_path, skeleton_name='rhythm'):
        '''add a melodic skeleton track into the orignal midi.'''

        midi_obj = miditoolkit.MidiFile(midi_path)
        notes = midi_obj.instruments[0].notes       # pick your melody tracks.
        bar_dict = self.group_bars(notes)
        if skeleton_name == 'syncopation':
            skeleton_dict = self.cal_syncopation(bar_dict)
        elif skeleton_name == 'downbeat':
            skeleton_dict = self.cal_downbeat(bar_dict)
        elif skeleton_name == 'long':
            skeleton_dict = self.cal_long(bar_dict)
        elif skeleton_name == 'rhythm':
            skeleton_dict = self.cal_rhythm(bar_dict)
        return skeleton_dict

    def save_skeleton(self, midi_path, save_path, skeleton_dict, skeleton_name):
        midi_obj = miditoolkit.MidiFile(midi_path)

        skeleton_list = []
        for note_list in skeleton_dict.values():
            for note in note_list:
                skeleton_note = miditoolkit.midi.containers.Note(start=note.start,end=note.end,velocity=note.velocity,pitch=note.pitch)
                skeleton_list.append(skeleton_note)
        skeleton_track = miditoolkit.Instrument(program=0, is_drum=False, name=skeleton_name)
        skeleton_track.notes.extend(skeleton_list)
        midi_obj.instruments.append(skeleton_track)
        midi_obj.dump(save_path)


    '''
    # perference rules (WuYun Version1; left as is, untested)
    def rhythm_skeleton_PR(self, sync_dict, beat_dict, long_dict, bar_dict):

        heavy_list = list(chain(*beat_dict.values()))
        long_list = list(chain(*long_dict.values()))
        sync_list = list(chain(*sync_dict.values()))

        skeleton_dict = dict()

        skeleton_note_list = []
        continuous_note_list = []
        continuous_note_index_list = []
        note_index = 0
        for k, v in bar_dict.items():
            if k not in skeleton_dict:
                skeleton_dict[k] = []

            for note in v:
                # add note in note_list
                note_object = Note(start=note.start, end=note.end, \
                                   pitch=note.pitch, velocity=note.velocity, index=note_index)
                note_index += 1
                # 第1次挑选 ｜ 当音符只属于节拍重音集合时
                if ((note in heavy_list) and (note not in long_list) and (note not in sync_list)):
                    skeleton_dict[k].append(note)
                    note_object.priority = 3
                    skeleton_note_list.append(note_object)
                    continuous_note_index_list.append(note_object.index)
                # 第2次挑选 ｜ 当音符属于节奏重音和长音时
                elif ((note in heavy_list) and (note in long_list) and (note not in sync_list)):
                    skeleton_dict[k].append(note)
                    note_object.priority = 1
                    skeleton_note_list.append(note_object)
                    continuous_note_index_list.append(note_object.index)
                # 第3次挑选 ｜ 当音符属于长音和切分音时
                elif ((note not in heavy_list) and (note in long_list) and (note in sync_list)):
                    skeleton_dict[k].append(note)
                    note_object.priority = 2
                    skeleton_note_list.append(note_object)
                    continuous_note_index_list.append(note_object.index)

        # print(">>>>>>>>>")
        # pprint.pprint(skeleton_dict)
        # print()

        # 将连续的骨干音放到同一个列表中
        last_note_index = 0
        for idx, note in enumerate(skeleton_note_list):
            if idx==0:
                continuous_note_list.append([note])
                last_note_index = note.index
            else:
                if note.index == last_note_index +1:
                    continuous_note_list[-1].append(note)
                    last_note_index = note.index
                else:
                    continuous_note_list.append([note])
                    last_note_index = note.index

        # print("Step2, 连续骨干音集合 = \n")
        continuous_note_list_len = [len(i) for i in continuous_note_list]
        # print(continuous_note_list_len)
        # print(f"num of skeleton = {sum(continuous_note_list_len)}")
        # print(continuous_note_index_list)

        # 筛选连续骨干音
        final_skeleton_note_list = []
        for group_idx, note_group in enumerate(continuous_note_list):
            # 不存在骨干音连续情况
            if len(note_group) == 1:
                final_skeleton_note_list.append(note_group)
            # 存在骨干音连续情况
            else:
                priority_list = []

                for note in note_group:
                    priority_list.append(note.priority)
                priority_set = set(priority_list)
                priority_set_length = len(priority_set)
                max_priority = min(priority_set) # 数字越小，优先级越高
                # print(f"Group_idx = {group_idx}, len = {len(note_group)}, priority_list = {priority_list}, priority_set = {priority_set}, priority_set_length = {priority_set_length}, max_priority = {max_priority}")

                # ------------------------------------------------
                # 仅含有一种优先级的骨干音， 不考虑都是次强拍的情况
                # ------------------------------------------------
                if priority_set_length == 1:
                    if max_priority == 1:
                        temp_group = []
                        for note in note_group:
                            if note.start % 1920 == 0: # 只选用强拍
                                temp_group.append(note)
                        final_skeleton_note_list.append(temp_group)
                    elif max_priority == 2:
                        temp_group = []
                        bar_group = dict()
                        for note in note_group:
                            note_bar = int(note.start / 1920)
                            if note_bar not in bar_group:
                                bar_group[note_bar] = []
                            bar_group[note_bar].append(note)
                        for k, v in bar_group.items():
                            if len(v) == 1:
                                temp_group.append(v[0])
                            else:
                                notes_length = [note.end - note.start for note in v] #  the common spit
                                max_length_note_index = notes_length.index(max(notes_length))
                                temp_group.append(v[max_length_note_index])
                        final_skeleton_note_list.append(temp_group)
                    elif max_priority == 3:
                        temp_group = []
                        for note in note_group:
                            if note.start % 1920 == 0: # 只选用强拍
                                temp_group.append(note)
                        final_skeleton_note_list.append(temp_group)
                # ------------------------------------------------
                # 含有两种优先级的骨干音
                # ------------------------------------------------
                elif priority_set_length == 2 or priority_set_length == 3:
                    # Condition >>>>> (1,2), (1,3), (1,2,3)
                    if 1 in priority_set:
                        # 筛选出优先级是1的
                        tempo_note_group_1 = []
                        for note in note_group:
                            if note.priority == 1:
                                tempo_note_group_1.append(note)
                        # 在优先级为1的情况下，筛选连续的次强拍
                        temp_group = []
                        if len(tempo_note_group_1) == 1:
                            temp_group.append(tempo_note_group_1[0])
                            final_skeleton_note_list.append(temp_group)
                        else:
                            for idx, note in enumerate(tempo_note_group_1):
                                if idx == len(tempo_note_group_1)-1:
                                    if note.index -1 == tempo_note_group_1[idx-1].index: # 相邻
                                        if note.start % 1920 ==0:
                                            temp_group.append(note)
                                    else:
                                        temp_group.append(note)
                                elif idx == 0:
                                    if note.index + 1 == tempo_note_group_1[idx+1].index: # 相邻
                                        if note.start % 1920 ==0:
                                            temp_group.append(note)
                                    else:
                                        temp_group.append(note)
                                else:
                                    # 1) 都不相邻
                                    if note.index + 1 != tempo_note_group_1[idx + 1].index and \
                                            note.index -1 != tempo_note_group_1[idx-1].index :
                                        temp_group.append(note)
                                    # 2）都相邻 & 左相邻，右不相邻 & 右相邻，左不相邻 ==>相邻
                                    else:
                                        if note.start % 1920 ==0:
                                            temp_group.append(note)
                            final_skeleton_note_list.append(temp_group)
                    # Condition >>>>>  (2,3)
                    else:
                        # 筛选出优先级是2的
                        tempo_note_group_2 = []
                        for note in note_group:
                            if note.priority == 2:
                                tempo_note_group_2.append(note)

                        temp_group = []
                        if len(tempo_note_group_2) == 1:
                            temp_group.append(tempo_note_group_2[0])
                            final_skeleton_note_list.append(temp_group)
                        else:
                            tempo_split_note_dict = dict()
                            for idx, note in enumerate(tempo_note_group_2):
                                note_bar = int(note.start/1920)
                                if note_bar not in tempo_split_note_dict:
                                    tempo_split_note_dict[note_bar] = []
                                tempo_split_note_dict[note_bar].append(note)
                            for k, v in tempo_split_note_dict.items():
                                if len(v) == 1:
                                    temp_group.append(v[0])
                                else:
                                    notes_length = [note.end - note.start for note in v]  # the common spit
                                    max_length_note_index = notes_length.index(max(notes_length))
                                    temp_group.append(v[max_length_note_index])
                            final_skeleton_note_list.append(temp_group)

        skeleton_melody_notes_list = []
        for note_list in final_skeleton_note_list:
            for note in note_list:
                start = note.start
                end = note.end
                pitch = note.pitch
                velocity = note.velocity
                skeleton_melody_notes_list.append(miditoolkit.Note(start=start,end=end,velocity=velocity,pitch=pitch))
        skeleton_melody_notes_list.sort(key=lambda x: (x.start, -x.end))
        # print(f"after filter num = {len(skeleton_melody_notes_list)}")

        return skeleton_melody_notes_list
    '''

if __name__ == "__main__":
    
    midi_path = './test_midi/test_1.mid'
    skeleton_mode = 'syncopation'
    save_path = f'./test_midi/test_1_{skeleton_mode}.mid'

    rs = Rhythm_Skeleton()
    skeleton_dict = rs.extract_rhythm_skeleton(midi_path, skeleton_mode)
    rs.save_skeleton(midi_path, save_path, skeleton_dict, skeleton_mode)


    