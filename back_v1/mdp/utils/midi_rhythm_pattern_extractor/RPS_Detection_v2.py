from typing import List
import miditoolkit
from miditoolkit.midi import parser as mid_parser
from miditoolkit.midi import containers as ct
from itertools import chain
import os
from tqdm import tqdm

interrupt_interval = 240

default_resolution = 480
beats_per_bar = 4
ticks_per_beat = 480  # default resolution = 480 ticks per quarter note, 四分音符 480ticks，十六分音符，120ticks
grid_per_bar = 16
cell = ticks_per_beat * 4 / grid_per_bar
grids_triple = 32
grids_normal = 64
file_name = ''
dst_path = ''

def print_formated(name, value):
    print("====================" + name + "====================")
    print(value)
    print("\n")

def print_note(notes):
    for note in notes:
        print("start: {}, end: {}, pitch: {}, velocity: {}, priority: {}, index: {}".format(note.start, note.end, note.pitch, note.velocity, note.priority, note.index))
    print("note_length: {}".format(len(notes)))

def print_rs(rhythm_seg):
    for rs in rhythm_seg:
        for note in rs:
            if note.priority < 5:
                print("start: {}, end: {}, pitch: {}, velocity: {}, priority: {}, index: {}".format(note.start, note.end, note.pitch, note.velocity, note.priority, note.index))
        if note.priority == 5:
            print("------------------------Phrase------------------------")
        if rs[0].priority != 5:
            print("------------------------RS----------------------------")

def print_cell(rhythm_cells):
    for rs in rhythm_cells:
        for cell in rs:
            if len(cell) == 1:
                continue
                print("------------------------Phrase------------------------")
            else:
                for note in cell:
                    print("start: {}, end: {}, pitch: {}, velocity: {}, priority: {}, index: {}".format(note.start, note.end, note.pitch, note.velocity, note.priority, note.index))
                print("------------------------Cell--------------------------")
        if rs[0][0].priority != 5:
            print("------------------------RS----------------------------")

class Note:
    def __init__(self, start, end, pitch, velocity, index, priority = 4):
        self.start = start
        self.end = end
        self.pitch = pitch
        self.velocity = velocity
        self.index = index
        
        # 4 = prolongation note 装饰音
        # 5 = interrupt note 中断音
        self.priority = priority

class RPS_Detection:

    def __init__(self, midi_path, resolution=480, grids=16):
        self.midi_path = midi_path
        self.file_name = os.path.basename(midi_path)
        self.resolution = resolution  # 默认为 四分音符 480ticks,一拍
        self.grids = grids  # 十六分音符，120ticks
        self.step = resolution * 4 / grids  # one step = 1/16 分音符
        self.bar_ticks = resolution * 4
        self.subsections = self._divide_subsections()

    def _divide_subsections(self):
        midi = miditoolkit.MidiFile(self.midi_path)
        notes = midi.instruments[0].notes
        res_dict = dict()
        for note in notes:
            start = note.start
            end = note.end
            duration = end - start
            # 筛选最小音符
            if duration >= self.step:
                key = int(start // self.bar_ticks)  # 按小节进行保存
                if key not in res_dict:
                    res_dict[key] = []
                res_dict[key].append(note)
        return res_dict

    # ---------------------
    # 类型一：切分音识别，规则如下：
    # ---------------------
    def _get_split(self):
        split_dict = dict()  # 切分音集合
        split_dict_4 = dict()  # 切分音集合 ｜ 测试使用
        split_dict_8 = dict()
        split_dict_16 = dict()

        step16 = self.step
        for bar_id, bar_notes in self.subsections.items():
            if bar_id not in split_dict:
                split_dict[bar_id] = []
                split_dict_4[f'{bar_id}'] = []
                split_dict_8[f'{bar_id}'] = []
                split_dict_16[f'{bar_id}'] = []

            start = self.bar_ticks * bar_id
            note_start_4 = [4 * step16 + start, 12 * step16 + start]
            note_start_8 = [i * step16 + start for i in range(2, 16, 4)]
            note_start_16 = [i * step16 + start for i in range(1, 16, 2)]

            for note in bar_notes:
                # 1）过滤小于16分音符时值的音符，无切分音意义； # 开始的位置帮助我们判断切分音类型，时长帮助我们判断是不是切分音
                note_duration = note.end - note.start
                if note_duration >= step16:

                    # 2.1）根据4分音符的所有切分音情形进行筛选
                    if (note.start == note_start_4[0]) and (note.end > (8 * step16 + start)):
                        split_dict[bar_id].append(note)
                        split_dict_4[f'{bar_id}'].append(note)
                    elif (note.start == note_start_4[1]) and (note.end > (16 * step16 + start)):
                        split_dict[bar_id].append(note)
                        split_dict_4[f'{bar_id}'].append(note)

                    # 2.2）根据8分音符的所有切分音情形进行筛选
                    elif (note.start == note_start_8[0]) and (note.end > 4 * step16 + start):
                        split_dict[bar_id].append(note)
                        split_dict_8[f'{bar_id}'].append(note)
                    elif (note.start == note_start_8[1]) and (note.end > 8 * step16 + start):
                        split_dict[bar_id].append(note)
                        split_dict_8[f'{bar_id}'].append(note)
                    elif (note.start == note_start_8[2]) and (note.end > 12 * step16 + start):
                        split_dict[bar_id].append(note)
                        split_dict_8[f'{bar_id}'].append(note)
                    elif (note.start == note_start_8[3]) and (note.end > 16 * step16 + start):
                        split_dict[bar_id].append(note)
                        split_dict_8[f'{bar_id}'].append(note)

                    # 2.3）根据16分音符的所有切分音情形进行筛选：音符开头在小节线上，音符结尾至少要超过最近强拍的一半时值
                    elif (note.start == note_start_16[0]) and (note.end > 2 * step16 + start):
                        split_dict[bar_id].append(note)
                        split_dict_16[f'{bar_id}'].append(note)
                    elif (note.start == note_start_16[1]) and (note.end > 4 * step16 + start):
                        split_dict[bar_id].append(note)
                        split_dict_16[f'{bar_id}'].append(note)
                    elif (note.start == note_start_16[2]) and (note.end > 6 * step16 + start):
                        split_dict[bar_id].append(note)
                        split_dict_16[f'{bar_id}'].append(note)
                    elif (note.start == note_start_16[3]) and (note.end > 8 * step16 + start):
                        split_dict[bar_id].append(note)
                        split_dict_16[f'{bar_id}'].append(note)
                    elif (note.start == note_start_16[4]) and (note.end > 10 * step16 + start):
                        split_dict[bar_id].append(note)
                        split_dict_16[f'{bar_id}'].append(note)
                    elif (note.start == note_start_16[5]) and (note.end > 12 * step16 + start):
                        split_dict[bar_id].append(note)
                        split_dict_16[f'{bar_id}'].append(note)
                    elif (note.start == note_start_16[6]) and (note.end > 14 * step16 + start):
                        split_dict[bar_id].append(note)
                        split_dict_16[f'{bar_id}'].append(note)
                    elif (note.start == note_start_16[7]) and (note.end > 16 * step16 + start):
                        split_dict[bar_id].append(note)
                        split_dict_16[f'{bar_id}'].append(note)
                else:
                    continue
        return split_dict, split_dict_4, split_dict_8, split_dict_16

    # --------------------------------------------------------------------------------------------------------------
    # 类型二：节拍重音
    # --------------------------------------------------------------------------------------------------------------
    def _get_stress(self):
        heavy_dict = dict()
        heavy_dict_clean = dict()
        split_dict, _, _, _ = self._get_split()

        # 1 采集节拍重音：第一拍 or 第三拍
        for bar_id, bar_notes in self.subsections.items():  # k = bar
            start = self.bar_ticks * (bar_id)  # [0,2],[2,4],[4,6]...
            first_beat_position = start
            third_beat_postion = start + 8 * self.step
            if bar_id not in heavy_dict:
                heavy_dict[bar_id] = []
            for note in bar_notes:
                if (note.start == first_beat_position) or (note.start == third_beat_postion):
                    heavy_dict[bar_id].append(note)

        # 2 过滤节拍重音
        for heavy_bar_id, heavy_bar_notes in heavy_dict.items():
            if heavy_bar_id not in heavy_dict_clean:
                heavy_dict_clean[heavy_bar_id] = []

            for heavy_note in heavy_bar_notes:
                heavy_note_flag = True
                heavy_note_length = heavy_note.end - heavy_note.start

                for split_bar_id, split_bar_notes in split_dict.items():
                    for split_note in split_bar_notes:
                        split_note_length = split_note.end - split_note.start
                        start_delta = split_note.start - heavy_note.start
                        if 0 <= start_delta < 3 * self.step:
                            heavy_note_flag = False
                            break

                    if heavy_note_flag == False:
                        break

                if heavy_note_flag:
                    heavy_dict_clean[heavy_bar_id].append(heavy_note)
        return heavy_dict_clean

    # ------------------------------------------------------------
    # 类型三：长音
    # ------------------------------------------------------------
    def _get_long(self):
        long_dict = dict()
        for bar_id, bar_notes in self.subsections.items():
            # 1. 创建长音字典
            if bar_id not in long_dict:
                long_dict[bar_id] = []

            # 2. 获取小节中时值最长的1个或多个音符索引（即存在多个时值最长且相同的音符）
            duration_list = [x.end - x.start for x in bar_notes]
            max_duration = max(duration_list)
            tup = [(i, duration_list[i]) for i in range(len(duration_list))]
            idx_list = [i for i, n in tup if n == max_duration]  # 相同时值长音的索引列表

            for idx in idx_list:
                long_dict[bar_id].append(bar_notes[idx])
        return long_dict

    # -------------------------------
    # 骨干音提取，筛选规整如下：
    # 1）过滤小于16分音符时值的音符，无切分音意义；
    # 2）三次挑选，查看石墨文档 3.2.1 提取结构线条音
    # -----------

    def prepare_dict(self):
        split_dict, _, _, _ = self._get_split()  # 切分音字典
        heavy_dict = self._get_stress()  # 节奏重音字典
        long_dict = self._get_long()  # 长音字典

        heavy_list = list(chain(*heavy_dict.values()))
        long_list = list(chain(*long_dict.values()))
        split_list = list(chain(*split_dict.values()))

        return split_list, heavy_list, long_list

    def extract_skeleton(self, heavy_list, long_list, split_list):
        skeleton_dict = dict()
        skeleton_note_list = []
        prolongation_note_list = []

        note_index = 0

        for k, v in self.subsections.items():  # 遍历每个小节的音符
            if k not in skeleton_dict:
                skeleton_dict[k] = []
            for note in v:
                # 添加音符
                note_object = Note(start=note.start, end=note.end, pitch=note.pitch, velocity=note.velocity,
                                   index=note_index)
                note_index += 1
                # 第1次挑选 ｜ 当音符只属于节拍重音集合时
                if ((note in heavy_list) and (note not in long_list) and (note not in split_list)):
                    skeleton_dict[k].append(note)
                    note_object.priority = 3
                    skeleton_note_list.append(note_object)
                # 第2次挑选 ｜ 当音符属于节奏重音和长音时
                elif ((note in heavy_list) and (note in long_list) and (note not in split_list)):
                    skeleton_dict[k].append(note)
                    note_object.priority = 1
                    skeleton_note_list.append(note_object)
                # 第3次挑选 ｜ 当音符属于长音和切分音时
                elif ((note not in heavy_list) and (note in long_list) and (note in split_list)):
                    skeleton_dict[k].append(note)
                    note_object.priority = 2
                    skeleton_note_list.append(note_object)
                else:
                    note_object.priority = 4
                    prolongation_note_list.append(note_object)
        
        return skeleton_note_list, prolongation_note_list
    
    def filter_continuous_skeleton(self, need_filter, skeleton_note_list, prolongation_note_list):
        # 按照Bar分组音符
        continuous_note_list = []
        last_note_index = 0
        for idx, note in enumerate(skeleton_note_list):
            if idx == 0:
                continuous_note_list.append([note])
                last_note_index = note.index
            else:
                if note.index == last_note_index + 1:
                    continuous_note_list[-1].append(note)
                    last_note_index = note.index
                else:
                    continuous_note_list.append([note])
                    last_note_index = note.index
        
        if not need_filter:
            return skeleton_note_list, prolongation_note_list

        final_skeleton_note_list = []
        for group_idx, note_group in enumerate(continuous_note_list):
            # 不存在骨干音连续情况
            if len(note_group) <= 2:
                final_skeleton_note_list.append(note_group)
            # 存在骨干音连续情况
            else:
                priority_list = []

                for note in note_group:
                    priority_list.append(note.priority)
                priority_set = set(priority_list)
                priority_set_length = len(priority_set)
                max_priority = min(priority_set)  # 数字越小，优先级越高

                # ------------------------------------------------
                # 仅含有一种优先级的骨干音， 不考虑都是次强拍的情况
                # ------------------------------------------------
                # 优先级：
                # 1：节奏重拍和长音
                # 2：长音和切分音
                # 3：节奏重拍
                if priority_set_length == 1:
                    if max_priority == 1:
                        temp_group = []
                        for note in note_group:
                            if note.start % 1920 == 0:  # 只选用强拍
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
                            if len(v) == 1: # 只有一个骨干音
                                temp_group.append(v[0])
                            else:           # 多个骨干音，选最长的那一个
                                notes_length = [note.end - note.start for note in v]
                                max_length_note_index = notes_length.index(max(notes_length))
                                temp_group.append(v[max_length_note_index])
                        final_skeleton_note_list.append(temp_group)
                    elif max_priority == 3:
                        temp_group = []
                        for note in note_group:
                            if note.start % 1920 == 0:  # 只选用强拍
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
                                if idx == len(tempo_note_group_1) - 1:
                                    if note.index - 1 == tempo_note_group_1[idx - 1].index:  # 相邻
                                        if note.start % 1920 == 0:
                                            temp_group.append(note)
                                    else:
                                        temp_group.append(note)
                                elif idx == 0:
                                    if note.index + 1 == tempo_note_group_1[idx + 1].index:  # 相邻
                                        if note.start % 1920 == 0:
                                            temp_group.append(note)
                                    else:
                                        temp_group.append(note)
                                else:
                                    # 1) 都不相邻
                                    if note.index + 1 != tempo_note_group_1[idx + 1].index and note.index - 1 != \
                                            tempo_note_group_1[idx - 1].index:
                                        temp_group.append(note)
                                    # 2）都相邻 & 左相邻，右不相邻 & 右相邻，左不相邻 ==> 相邻
                                    else:
                                        if note.start % 1920 == 0:
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
                                note_bar = int(note.start / 1920)
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

        unfold_skeleton_note_list = []
        for notes in final_skeleton_note_list:
            for note in notes:
                unfold_skeleton_note_list.append(note)

        for note in skeleton_note_list:
            if note not in unfold_skeleton_note_list:
                n_note = Note(start=note.start, end=note.end, pitch=note.pitch, velocity=note.velocity, index=note.index, priority=4)
                prolongation_note_list.append(n_note)

        return unfold_skeleton_note_list, prolongation_note_list

    def generate_subsection_notes_list(self, all_notes_list):
        subsection_notes_list = []
        for key in self.subsections.keys():
            curr_section = self.subsections[key]
            notes_section = []
            for old_note in curr_section:
                for note in all_notes_list:
                    if note.start == old_note.start and note.end == old_note.end and note.pitch == old_note.pitch and note.velocity == old_note.velocity:
                        notes_section.append(note)
            notes_section.sort(key=lambda x: x.index, reverse=False)
            subsection_notes_list.append(notes_section)
        
        return subsection_notes_list

    def add_interrupt_notes(self, notes_list):
        new_notes_list = []
        previous_note = None
        
        for note in notes_list:
            if previous_note == None:
                previous_note = note
                new_notes_list.append(note)
                continue
            # 判断相邻音符间距, 当duration>=240时, 插入中断音
            else:
                duration = note.start - previous_note.end
                if duration >= interrupt_interval:
                    new_notes_list.append(Note(start=previous_note.end, end=note.start, pitch=0, velocity=0, index=0, priority=5))
                    new_notes_list.append(note)
                else:
                    new_notes_list.append(note)
                previous_note = note
        
        # 更新音符序号
        note_index = 0
        for note in new_notes_list:
            note.index = note_index
            note_index += 1
        
        return new_notes_list

    def rhythm_segmentation(self, notes_list):

        def has_skeleton_note(rhythem_seg):
            for note in rhythem_seg:
                if note.priority < 4:
                    return True
            return False

        rhythm_seg_notes_list = []
        single_rhythm_seg = []
        notes_list_idx = 0
        for notes_list_idx in range(len(notes_list)):
            note = notes_list[notes_list_idx]
            # print("start: {}, end: {}, pitch: {}, velocity: {}, priority: {}, index: {}".format(note.start, note.end, note.pitch, note.velocity, note.priority, note.index))
            # 骨干音
            if note.priority < 4:
                if len(single_rhythm_seg) == 0:
                    single_rhythm_seg.append(note)
                else:
                    # [装饰音 装饰音 ... 骨干音]
                    if single_rhythm_seg[0].priority == 4:
                        single_rhythm_seg.append(note)
                        rhythm_seg_notes_list.append(single_rhythm_seg)
                        single_rhythm_seg = []
                    elif single_rhythm_seg[0].priority < 4:
                        if len(single_rhythm_seg) == 1:
                            # [ ... 骨干音] [骨干音 如果与上一个细胞间隔>=240ticks，放到当前小节；否则放到上一小节
                            if len(rhythm_seg_notes_list) != 0 and single_rhythm_seg[0].start - rhythm_seg_notes_list[-1][-1].end < interrupt_interval:
                                rhythm_seg_notes_list[-1].append(single_rhythm_seg[0])
                                single_rhythm_seg = []
                                single_rhythm_seg.append(note)
                            # 开始 [骨干音 骨干音
                            else:
                                single_rhythm_seg.append(note)
                        # [骨干音 装饰音 装饰音 ... ] [骨干音 
                        else:
                            rhythm_seg_notes_list.append(single_rhythm_seg)
                            single_rhythm_seg = []
                            single_rhythm_seg.append(note)
            # 装饰音
            elif note.priority == 4:
                single_rhythm_seg.append(note)
            # 中断音
            elif note.priority == 5:
                if len(single_rhythm_seg) == 1:
                    # 与上一个音符间隔小于240ticks
                    if len(rhythm_seg_notes_list) != 0 and abs(single_rhythm_seg[0].start - rhythm_seg_notes_list[-1][-1].end) < interrupt_interval:
                        rhythm_seg_notes_list[-1].append(single_rhythm_seg[0])
                        single_rhythm_seg = []
                        continue
                    # 音符前后间隔都大于等于240ticks
                    # 最后一个音符
                    if len(rhythm_seg_notes_list) != 0 and notes_list_idx == len(notes_list) - 1:
                        rhythm_seg_notes_list[-1].append(single_rhythm_seg[0])
                        single_rhythm_seg = []
                        continue
                    if len(rhythm_seg_notes_list) != 0:
                        if notes_list_idx != len(notes_list) - 1:
                            previous_note_interval = abs(single_rhythm_seg[0].start - rhythm_seg_notes_list[-1][-1].end)
                            next_note_interval = abs(single_rhythm_seg[0].end - notes_list[notes_list_idx + 1].start)
                            if previous_note_interval < next_note_interval:
                                rhythm_seg_notes_list[-1].append(single_rhythm_seg[0])
                                single_rhythm_seg = []
                                continue
                            else:
                                continue
                        else:
                            rhythm_seg_notes_list[-1].append(single_rhythm_seg[0])
                            single_rhythm_seg = []
                            continue
                    else:
                        continue
                else:
                    if len(single_rhythm_seg) != 0:
                        if has_skeleton_note(single_rhythm_seg):
                            rhythm_seg_notes_list.append(single_rhythm_seg)
                        else:
                            if len(rhythm_seg_notes_list) != 0:
                                if abs(single_rhythm_seg[0].start - rhythm_seg_notes_list[-1][-1].end) < interrupt_interval:
                                    for n in single_rhythm_seg:
                                        rhythm_seg_notes_list[-1].append(n)
                                else:
                                    rhythm_seg_notes_list.append(single_rhythm_seg)
                            else:
                                rhythm_seg_notes_list.append(single_rhythm_seg)
                single_rhythm_seg = []
        
        if single_rhythm_seg != []:
            if len(single_rhythm_seg) > 1:
                rhythm_seg_notes_list.append(single_rhythm_seg)
            else:
                rhythm_seg_notes_list[-1].append(single_rhythm_seg[0])

        return rhythm_seg_notes_list

    def rhythm_cell_segmentation(self, rhythm_seg_notes_list, subsection_notes_list):
        
        def cell_normalization(cell):
            normalized_cell = []
            start_coefficient = cell[0].start
            pitch_coefficient = cell[0].pitch
            for note in cell:
                normalized_note = Note(start=note.start - start_coefficient, end=note.end - start_coefficient, pitch=note.pitch - pitch_coefficient, velocity=note.velocity, index=note.index, priority=note.priority)
                normalized_cell.append(normalized_note)
            return normalized_cell
        
        def normalized_cell_compare(l: List[Note], r: List[Note]):
            if len(l) != len(r):
                return False
            
            for idx in range(len(l)):
                l_note = l[idx]
                r_note = r[idx]
                if l_note.start == r_note.start and l_note.end == r_note.end:
                    continue
                else: 
                    return False
            
            return True

        def cal_repetition(cell_group, subsection_notes_list):
            key_cell = cell_group[0]
            normalized_key_cell = cell_normalization(key_cell)

            # 将整首乐曲中每一段按照key_cell长度划分，同时进行归一化
            normalized_all_notes_divisions = []
            for section in subsection_notes_list:
                normalized_section_notes_divisions = []
                for idx in range(len(section) - len(key_cell) + 1):
                    curr_division = []
                    for i in range(len(key_cell)):
                        curr_division.append(section[idx + i])
                    normalized_curr_division = cell_normalization(curr_division)
                    normalized_section_notes_divisions.append(normalized_curr_division)
                normalized_all_notes_divisions.append(normalized_section_notes_divisions)
            # 在全曲范围内进行搜索，并统计重复次数
            res = 0
            for section in normalized_all_notes_divisions:
                for division in section:
                    if normalized_cell_compare(normalized_key_cell, division):
                        res += 1

            return res

        def single_rhythm_seg_cells(rhythm_seg, subsection_notes_list):
            curr_rhythm_cell_seg = []
            if len(rhythm_seg) == 1:
                curr_rhythm_cell_seg.append(rhythm_seg)
            elif len(rhythm_seg) == 2:
                curr_rhythm_cell_seg.append(rhythm_seg)
            elif len(rhythm_seg) == 3:
                curr_rhythm_cell_seg.append(rhythm_seg)
            elif len(rhythm_seg) == 4:
                curr_rhythm_cell_seg = [rhythm_seg[0:2], rhythm_seg[2:4]]
            elif len(rhythm_seg) == 5:
                rhythm_cell_choices = [
                    [rhythm_seg[0:3], rhythm_seg[3:5]],
                    [rhythm_seg[0:2], rhythm_seg[2:5]],
                ]
                highest_res = -1
                best_choice = []
                for choice in rhythm_cell_choices:
                    res = cal_repetition(choice, subsection_notes_list)
                    if res > highest_res:
                        highest_res = res
                        best_choice = choice
                curr_rhythm_cell_seg = best_choice
            elif len(rhythm_seg) == 6:
                rhythm_cell_choices = [
                    [rhythm_seg[0:3], rhythm_seg[3:6]],
                    [rhythm_seg[0:2]] + single_rhythm_seg_cells(rhythm_seg[2:len(rhythm_seg)], subsection_notes_list),
                ]
                highest_res = -1
                best_choice = []
                for choice in rhythm_cell_choices:
                    res = cal_repetition(choice, subsection_notes_list)
                    if res > highest_res:
                        highest_res = res
                        best_choice = choice
                curr_rhythm_cell_seg = best_choice
            elif len(rhythm_seg) >= 7:
                rhythm_cell_choices = [
                    [rhythm_seg[0:3]] + single_rhythm_seg_cells(rhythm_seg[3:len(rhythm_seg)], subsection_notes_list),
                    [rhythm_seg[0:2]] + single_rhythm_seg_cells(rhythm_seg[2:len(rhythm_seg)], subsection_notes_list),
                ]
                highest_res = -1
                best_choice = []
                for choice in rhythm_cell_choices:
                    res = cal_repetition(choice, subsection_notes_list)
                    if res > highest_res:
                        highest_res = res
                        best_choice = choice
                curr_rhythm_cell_seg = best_choice
            
            return curr_rhythm_cell_seg

        rhythm_cell_seg_notes_list = []
        for rhythm_seg in rhythm_seg_notes_list:
            rhythm_cell_seg_notes_list.append(single_rhythm_seg_cells(rhythm_seg, subsection_notes_list))
        
        return rhythm_cell_seg_notes_list

    def formatted_rhythm_cell_output(self, rhythm_cell_seg_notes_list, output_file_dir):
        def cell_normalization(cell):
            normalized_cell = []
            start_coefficient = cell[0].start
            pitch_coefficient = cell[0].pitch
            for note in cell:
                normalized_note = Note(start=note.start - start_coefficient, end=note.end - start_coefficient, pitch=note.pitch - pitch_coefficient, velocity=note.velocity, index=note.index, priority=note.priority)
                normalized_cell.append(normalized_note)
            return normalized_cell
        
        def normalized_cell_compare(l: List[Note], r: List[Note]):
            if len(l) != len(r):
                return False
            
            for idx in range(len(l)):
                l_note = l[idx]
                r_note = r[idx]
                if l_note.start == r_note.start and l_note.end == r_note.end:
                    continue
                else: 
                    return False
            
            return True
        
        with open(output_file_dir, "w") as f:
            f.write(file_name + "\n")

            rps_dict = {}
            rps_idx = 1
            for rs_idx in range(len(rhythm_cell_seg_notes_list)):
                rs = rhythm_cell_seg_notes_list[rs_idx]
                first_cell = rs[0]
                first_note = first_cell[0]
                rs_bar = first_note.start // (beats_per_bar * ticks_per_beat)
                f.write("RS{}_bar{}".format(rs_idx + 1, rs_bar + 1) + "\n")
                
                for cell in rs:
                    normalized_cell = cell_normalization(cell)
                    in_rps_dict = False
                    for rps_name in rps_dict.keys():
                        if normalized_cell_compare(normalized_cell, rps_dict[rps_name]):
                            curr_line = "{}: {}[".format(rps_name, len(cell))
                            for idx in range(len(cell)):
                                note = cell[idx]
                                if idx != 0:
                                    curr_line += ", "
                                curr_line += "note{}(start = {}, end = {})".format(idx + 1, note.start, note.end)
                            curr_line += "]"
                            in_rps_dict = True
                            break
                    if not in_rps_dict:
                        rps_dict["RPS{}".format(rps_idx)] = normalized_cell
                        rps_name = "RPS{}".format(rps_idx)
                        rps_idx += 1
                        curr_line = "{}: {}[".format(rps_name, len(cell))
                        for idx in range(len(cell)):
                            note = cell[idx]
                            if idx != 0:
                                curr_line += ", "
                            curr_line += "note{}(start = {}, end = {})".format(idx + 1, note.start, note.end)
                        curr_line += "]"
                    
                    f.write("\t" + curr_line + "\n")
            f.close()

    def export_midi_file(self, rhythm_cell_seg_notes_list):
        mido_obj = mid_parser.MidiFile()
        beat_resol = mido_obj.ticks_per_beat
        track = ct.Instrument(program=0, is_drum=False, name='track1')
        mido_obj.instruments = [track]
        
        color_idx = 0

        for rs in rhythm_cell_seg_notes_list:
            for cell in rs:
                for n in cell:
                    if color_idx % 2 == 0:
                        note = ct.Note(start=n.start, end=n.end, pitch=n.pitch, velocity=60)
                    else:
                        note = ct.Note(start=n.start, end=n.end, pitch=n.pitch, velocity=127)
                    mido_obj.instruments[0].notes.append(note)
                color_idx += 1
        
        mido_obj.dump('result.mid')

    def get_RPS_List(self, rhythm_cell_seg_notes_list):
        RPS_list = []
        for bar in rhythm_cell_seg_notes_list:
            for RP in bar:
                RPS_group = []
                for item in RP:
                    RPS_group.append(miditoolkit.Note(start=item.start, end=item.end, pitch=item.pitch, velocity=item.velocity))
                RPS_list.append(RPS_group)
        return RPS_list

    
    def all_steps(self):
        # ------------------------------------------------------------------------------------------------ #
        # Stage 0: 准备字典
        # ------------------------------------------------------------------------------------------------ #
        
        # split_dict: 切分音字典  heavy_dict: 节奏重音字典  long_dict: 长音字典
        split_list, heavy_list, long_list = self.prepare_dict()


        # ------------------------------------------------------------------------------------------------ #
        # Stage 1: 节奏骨干音提取
        # ------------------------------------------------------------------------------------------------ #

        # -------------------------------- 提取骨干音 -------------------------------- #
        skeleton_note_list, prolongation_note_list = self.extract_skeleton(heavy_list, long_list, split_list)

        # 骨干音: skeleton_note_list
        # 装饰音: prolongation_note_list

        # -------------------------------- 筛选连续骨干音 ----------------------------- #
        filtered_skeleton_note_list, prolongation_note_list = self.filter_continuous_skeleton(need_filter=True, skeleton_note_list=skeleton_note_list, prolongation_note_list=prolongation_note_list)


        # ------------------------------------------------------------------------------------------------ #
        # Stage 2:  节奏识别分段
        # ------------------------------------------------------------------------------------------------ #
        all_notes_list = filtered_skeleton_note_list + prolongation_note_list
        all_notes_list.sort(key=lambda x: x.index, reverse=False)
        subsection_notes_list = self.generate_subsection_notes_list(all_notes_list)

        # -------------------------------- 添加中断音 -------------------------------- #
        refined_notes_list = self.add_interrupt_notes(all_notes_list)

        # -------------------------------- RS划分 ----------------------------------- #
        rhythm_seg_notes_list = self.rhythm_segmentation(refined_notes_list)
        print_rs(rhythm_seg_notes_list)
        print(rhythm_seg_notes_list)


        # ------------------------------------------------------------------------------------------------ #
        # Stage 3:  节奏细胞结构
        # ------------------------------------------------------------------------------------------------ #
        rhythm_cell_seg_notes_list = self.rhythm_cell_segmentation(rhythm_seg_notes_list, subsection_notes_list)
        print_cell(rhythm_cell_seg_notes_list)
        print(rhythm_cell_seg_notes_list)

        # ------------------------------------------------------------------------------------------------ #
        # Stage 4:  节奏细胞结构列表输出
        # ------------------------------------------------------------------------------------------------ #
        self.formatted_rhythm_cell_output(rhythm_cell_seg_notes_list, "{}/{}.txt".format(dst_path, file_name.split(".")[0]))
        self.export_midi_file(rhythm_cell_seg_notes_list)



if __name__ == '__main__':
    # rootdir = '/Users/albertyu/Documents/科研/Dataset_16thNote/Wikifornia_v3/13_dataset_held50/Wikifornia_melody/test_16th/Wikifornia_tonality_nomal_1792.mid'
    rootdir = 'demo.mid'
    
    midi_path = rootdir
    file_name = os.path.basename(midi_path)
    dst_path = '.'
    
    m = RPS_Detection(midi_path)
    m.all_steps()

    # rootdir = '/Users/albertyu/Documents/科研/Dataset_16thNote/Wikifornia_v3/13_dataset_held50/Wikifornia_melody/test_16th'
    # listfile = os.listdir(rootdir)
    # for i in tqdm(range(0, len(listfile))):
    #     midi_path = os.path.join(rootdir, listfile[i])
    #     file_name = os.path.basename(midi_path)
    #     dst_path = './output'
    #     # print("{} doing!".format(file_name))
    #     m = RPS_Detection(midi_path)      # midi对象
    #     m.all_steps()
    #     # break
    