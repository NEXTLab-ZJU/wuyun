import pprint
import miditoolkit
from itertools import chain
import os
import math

default_resolution = 480
ticks_per_beat = 480  # default resolution = 480 ticks per quarter note, 四分音符 480ticks，十六分音符，120ticks
grid_per_bar = 16
cell = ticks_per_beat * 4 / grid_per_bar
grids_triple = 32
grids_normal = 64
file_name = ''
dst_path = ''


def to_left(ls, n):
    return n.start - ls[-1].end


def to_right(ls, n):
    return ls[0].start - n.end


class Note:
    def __init__(self, start, end, pitch, velocity, index, priority=4): # 4 = prolongation note 装饰音
        self.start = start
        self.end = end
        self.pitch = pitch
        self.velocity = velocity
        self.index = index
        self.priority = priority


class Melody_Skeleton_Extractor:

    def __init__(self, midi_path, resolution=480, grids=16):
        self.midi_path = midi_path
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
                        if (heavy_note.end <= split_note.start <= heavy_note.end + 3 * self.step) and (
                                heavy_note_length < split_note_length):
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

    def get_skeleton(self):
        print(f"File Name = {self.midi_path}")
        split_dict, _, _, _ = self._get_split()  # 切分音字典
        heavy_dict = self._get_stress()  # 节奏重音字典
        long_dict = self._get_long()  # 长音字典

        heavy_list = list(chain(*heavy_dict.values()))
        long_list = list(chain(*long_dict.values()))
        split_list = list(chain(*split_dict.values()))

        skeleton_dict = dict()

        skeleton_note_list = []
        continuous_note_list = []
        continuous_note_index_list = []
        prolongation_note_list = []
        note_index = 0

        # ------------------------------------------------------------------------------------------------ #
        # Stage1: 节奏骨干音 | Xinda
        # ------------------------------------------------------------------------------------------------ #

        # -------------------------------- 提取骨干音 -------------------------------- #
        for k, v in self.subsections.items():  # 遍历每个小节的音符
            if k not in skeleton_dict:
                skeleton_dict[k] = []
            for note in v:
                # add note in note_list
                note_object = Note(start=note.start, end=note.end, pitch=note.pitch, velocity=note.velocity,
                                   index=note_index)
                note_index += 1
                # 第1次挑选 ｜ 当音符只属于节拍重音集合时
                if ((note in heavy_list) and (note not in long_list) and (note not in split_list)):
                    skeleton_dict[k].append(note)
                    note_object.priority = 3
                    skeleton_note_list.append(note_object)
                    continuous_note_index_list.append(note_object.index)
                # 第2次挑选 ｜ 当音符属于节奏重音和长音时
                elif ((note in heavy_list) and (note in long_list) and (note not in split_list)):
                    skeleton_dict[k].append(note)
                    note_object.priority = 1
                    skeleton_note_list.append(note_object)
                    continuous_note_index_list.append(note_object.index)
                # 第3次挑选 ｜ 当音符属于长音和切分音时
                elif ((note not in heavy_list) and (note in long_list) and (note in split_list)):
                    skeleton_dict[k].append(note)
                    note_object.priority = 2
                    skeleton_note_list.append(note_object)
                    continuous_note_index_list.append(note_object.index)
                else:
                    note_object.priority = 4
                    prolongation_note_list.append(note_object)

        # -------------------------------- 筛选连续骨干音 -------------------------------- #
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
                max_priority = min(priority_set)  # 数字越小，优先级越高
                # print(f"Group_idx = {group_idx}, len = {len(note_group)}, priority_list = {priority_list}, priority_set = {priority_set}, priority_set_length = {priority_set_length}, max_priority = {max_priority}")

                # ------------------------------------------------
                # 仅含有一种优先级的骨干音， 不考虑都是次强拍的情况
                # ------------------------------------------------
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
                            if len(v) == 1:
                                temp_group.append(v[0])
                            else:
                                notes_length = [note.end - note.start for note in v]  # the common spit
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
                                    # 2）都相邻 & 左相邻，右不相邻 & 右相邻，左不相邻 ==>相邻
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
        # pprint.pprint(final_skeleton_note_list)
        # print(final_skeleton_note_list)


        # ------------------------------------------------------------------------------------------------ #
        # Stage2: 节奏骨干音 | Yanqin
        # ------------------------------------------------------------------------------------------------ #

        # ----------------------------- 第二轮骨干音筛选：强拍、次强拍筛选-------------------------------------- #
        print('***** 第二轮骨干音筛选：强拍、次强拍筛选 *****')
        skeleton_melody_notes_list = []
        for note_list in final_skeleton_note_list:
            for note in note_list:
                start = note.start
                end = note.end
                pitch = note.pitch
                velocity = note.velocity
                skeleton_melody_notes_list.append(miditoolkit.Note(start=start, end=end, velocity=127, pitch=pitch))
        skeleton_melody_notes_list.sort(key=lambda x: (x.start, -x.end))
        print(f"after filter num = {len(skeleton_melody_notes_list)}")
        finallist = []  # 包含成对组合的连续骨干音
        for k, v in self.subsections.items():  # 遍历每个小节的音符
            for n in v:
                # add note in note_list
                note_object = Note(start=n.start, end=n.end, pitch=n.pitch, velocity=n.velocity, index=note_index)
                note_index += 1
                flag = 0
                exist = 0
                for i in final_skeleton_note_list:  # 判断是否是第一轮筛选后的骨干音
                    for note in i:
                        if n.start == note.start:
                            flag = 1
                for i in prolongation_note_list:  # 判断是否是修饰音
                    if n.start == i.start:
                        exist = 1
                if (flag == 0 and exist == 0):  # 如果既不是筛选后骨干音，也不是修饰音，则为第一轮筛选后新增的修饰音
                    note_object.priority = 4
                    prolongation_note_list.append(note_object)
        first_note = 0  # 列表第一个音
        for note_list in final_skeleton_note_list:
            for note in note_list:
                flag = 0
                exist = 0
                note_index = 0
                go = 0
                if (first_note == 0):  # 第一个音符初始化，为previous_xxx生成初值

                    previous_start = -1
                    previous_end = -1
                    previous_pitch = -1
                    previous_velocity = -1
                    previous_index = -1
                    first_note = 1
                # print(note.start,note.end)

                if (note.start != 0):
                    if note.start % 1920 == previous_start % 1920 and note.start % 1920 == 0 and note.start - previous_start == 1920:  # 连续两个在强拍上的骨干音

                        for k, v in self.subsections.items():  # 遍历每个小节的音符，查找这两个音符之间是否存在修饰音
                            for n in v:
                                # add note in note_list
                                note_object = Note(start=n.start, end=n.end, pitch=n.pitch, velocity=n.velocity,
                                                   index=note_index)
                                note_index += 1
                                if (go == 1):
                                    for i in prolongation_note_list:
                                        if i.start == note_object.start:
                                            exist = 1
                                            go = 0
                                            break
                                if (n.start == previous_start):
                                    go = 1
                                    for i in prolongation_note_list:
                                        if i.start == note_object.start:
                                            exist = 1
                                            go = 0
                                            break
                                if (n.start == note.start):
                                    go = 0

                        if exist == 0:  # 如果不存在修饰音，则将两个音符都加入finallist
                            finallist.append(
                                miditoolkit.Note(start=previous_start, end=previous_end, velocity=previous_velocity,
                                                 pitch=previous_pitch))
                            finallist.append(miditoolkit.Note(start=note.start, end=note.end, velocity=note.velocity,
                                                              pitch=note.pitch))

                    if note.start % 1920 == previous_start % 1920 and note.start % 1920 == 960 and note.start - previous_start == 1920:  # 连续两个次强拍上的骨干音
                        for k, v in self.subsections.items():  # 遍历每个小节的音符
                            for n in v:
                                # add note in note_list
                                note_object = Note(start=n.start, end=n.end, pitch=n.pitch, velocity=n.velocity,
                                                   index=note_index)
                                note_index += 1
                                if (go == 1):
                                    for i in prolongation_note_list:
                                        if i.start == note_object.start:
                                            exist = 1
                                            go = 0
                                            break
                                if (n.start == previous_start):
                                    go = 1
                                    for i in prolongation_note_list:
                                        if i.start == note_object.start:
                                            exist = 1
                                            go = 0
                                            break
                                if (n.start == note.start):
                                    go = 0
                        if exist == 0:
                            finallist.append(
                                miditoolkit.Note(start=previous_start, end=previous_end, velocity=previous_velocity,
                                                 pitch=previous_pitch))
                            finallist.append(miditoolkit.Note(start=note.start, end=note.end, velocity=note.velocity,
                                                              pitch=note.pitch))

                previous_start = note.start
                previous_end = note.end
                previous_pitch = note.pitch
                previous_velocity = note.velocity
                previous_index = note.index
                fisrt_note = 1  # 当前音符作为下一个音符的previous
        finallist.sort(key=lambda x: (x.start, -x.end))
        count = 1
        norepeat_final = []  # 去重
        for i in finallist:
            flag = 0
            for j in norepeat_final:
                if j.start == i.start:
                    flag = 1
                    break
            if flag == 0:
                norepeat_final.append(miditoolkit.Note(start=i.start, end=i.end, velocity=i.velocity, pitch=i.pitch))
        # pprint.pprint(norepeat_final)
        step2_list = []
        for i in skeleton_melody_notes_list:
            flag = 0
            for j in norepeat_final:
                if j.start == i.start:
                    flag = 1
            if (flag == 0):
                step2_list.append(miditoolkit.Note(start=i.start, end=i.end, velocity=i.velocity, pitch=i.pitch))
        # pprint.pprint(norepeat_final)
        rhythm_pattern = []
        left = 0
        right = 3840
        first_own_note = 0
        pprint.pprint(norepeat_final)
        for i in norepeat_final:
            if (i.start < left):
                continue
            if (i.start >= right):
                while (1):
                    left = left + 3840
                    right = right + 3840
                    if i.start >= left and i.start < right:
                        break
            if (first_own_note == 0):
                rhythm_pattern.append(i)
                left = left + 3840
                right = right + 3840

        pprint.pprint(rhythm_pattern)
        step4_list = []
        for i in step2_list:
            step4_list.append(miditoolkit.Note(start=i.start, end=i.end, velocity=127, pitch=60))
        for i in rhythm_pattern:
            step4_list.append(miditoolkit.Note(start=i.start, end=i.end, velocity=127, pitch=60))
        step4_list.sort(key=lambda x: (x.start, -x.end))
        # pprint.pprint(step4_list)
        exist_continue = 0  # 用于判断是否进入连续骨干音
        num_skeleton = []  # 用于保存每一段骨干音的数量
        paragraph = 0  # 段数
        length_continue = 0
        length_continue_list = []
        paragraph_list = []
        start_list = []  # 每一段连续骨干音的起始位置列表
        num_item = 0  # 音符数量
        for i in norepeat_final:
            last_note = miditoolkit.Note(start=i.start, end=i.end, velocity=i.velocity, pitch=i.pitch)  # 找到最后一个音符
        # 在处理中，对于第一个和最后一个音符的判断是非常重要的
        first_test2 = 0
        for i in norepeat_final:

            if (i.start == last_note.start):  # 遇到最后一个音符，此段落结束
                paragraph = paragraph + 1
                length_continue = length_continue + i.end - i.start
                length_continue_list.append(length_continue)

                num_skeleton.append(num_item + 1)
            if first_test2 == 0:  # 遇到第一个音符，初始化参数
                previous_start = i.start
                previous_end = i.end
                previous_pitch = i.pitch
                previous_velocity = i.velocity
                first_test2 = 1
                num_item = num_item + 1
                start_list.append(miditoolkit.Note(start=i.start, end=i.end, velocity=i.velocity, pitch=i.pitch))
                continue

            if i.start != 0:  # 如果两个音符之间距离大于一个小节，则段落结束，reset
                if (i.start - previous_end >= 1920):
                    exist_continue = 0
                    num_skeleton.append(num_item)
                    num_item = 0
                    length_continue_list.append(length_continue)
                    paragraph = paragraph + 1
                    start_list.append(miditoolkit.Note(start=i.start, end=i.end, velocity=i.velocity, pitch=i.pitch))
                if (exist_continue == 0):  # 开始进入连续
                    exist_continue = 1
                    length_continue = i.start - previous_start


                elif (i.start - previous_end >= 1920):  # 连续中断
                    exist_continue = 0
                    num_skeleton.append(num_item)
                    num_item = 0
                    length_continue_list.append(length_continue)
                    paragraph = paragraph + 1
                    start_list.append(miditoolkit.Note(start=i.start, end=i.end, velocity=i.velocity, pitch=i.pitch))
                if exist_continue == 1:  # 处理中间音
                    length_continue = length_continue + i.end - i.start
                    num_item = 1 + num_item

            previous_start = i.start
            previous_end = i.end
            previous_pitch = i.pitch
            previous_velocity = i.velocity

        output_name = os.path.basename(midi_path[:-3]) + 'txt'
        data = open("./output_txt/%s" % output_name, 'w+')  # 输出位置
        print('Number of continuous backbone segments:', paragraph, file=data)
        for i in range(paragraph):
            start = start_list[i]
            subsection = start.start / 1920 + 1
            print(file_name, ', %d paragraph' % (i + 1), ' from subsection %d' % subsection,
                  ' , Continuous backbone sound length:%d' % num_skeleton[i], file=data)
            for j in range(num_skeleton[i]):
                j_start = 0
                count = j + 1
                flag = 0
                for k in norepeat_final:  # 找音符的起始位置

                    if (k.start == start):
                        flag = 1
                    if (flag == 1):
                        count = count - 1
                    if (count == 0):
                        j_start = k.start

                term1 = 0
                term2 = 0
                for m in heavy_list:  # 判断是不是节拍重音
                    if m.start == j_start:
                        term1 = 1
                for n in long_list:  # 判断是不是长音
                    if n.start == j_start:
                        term2 = 1
                if term1 == 1 and term2 == 1:  # 判断优先级
                    priority = 1
                else:
                    priority = 3
                if j == num_skeleton[i] - 1:
                    print('x%d : priority %d' % (j + 1, priority), end="\n", file=data)
                else:
                    print('x%d : priority %d' % (j + 1, priority), end=" , ", file=data)

        print('*****')

        #         for i in skeleton_melody_notes_list:
        #             pprint.pprint(i)
        pattern_final = []
        for k, v in self.subsections.items():  # 遍历每个小节的音符，查找这两个音符之间是否存在修饰音
            for n in v:
                # add note in note_list
                note_object = Note(start=n.start, end=n.end, pitch=n.pitch, velocity=n.velocity, index=note_index)
                note_index += 1
                check = 0
                for j in step4_list:
                    if n.start == j.start:
                        check = 1
                if check == 0:
                    pattern_final.append(miditoolkit.Note(start=n.start, end=n.end, velocity=60, pitch=n.pitch))
                else:
                    pattern_final.append(miditoolkit.Note(start=n.start, end=n.end, velocity=127, pitch=n.pitch))
        pattern_final.sort(key=lambda x: (x.start, -x.end))

        # pprint.pprint(pattern_final)
        print('*****Stage3 Start*****')
        stage3_unit_list = []
        item_list = []
        stage3_prolongation = []
        check_ske = 0
        check_prol = 0
        distance = 0
        state_add = 0
        from_ske = 0
        from_prol = 0
        stage3_init = 0
        for i in pattern_final:
            if i.velocity == 60:
                stage3_prolongation.append(
                    miditoolkit.Note(start=i.start, end=i.end, velocity=i.velocity, pitch=i.pitch))
        for i in pattern_final:
            if (stage3_init == 0):
                previous_end = -1
                stage3_init = 1
            check_ske = 0
            check_prol = 0
            length = i.end - i.start
            for j in stage3_prolongation:
                if i.start == j.start:
                    check_prol = 1
            for j in step4_list:
                if i.start == j.start:
                    check_ske = 1
            # print(check_prol,check_ske,end="\n")
            if check_prol == 1 and state_add == 0:
                item_list.append(miditoolkit.Note(start=i.start, end=i.end, velocity=i.velocity, pitch=i.pitch))
                state_add = 1
                previous_end = i.end
                from_prol = 1
                continue
            if check_ske == 1 and state_add == 0:
                item_list.append(miditoolkit.Note(start=i.start, end=i.end, velocity=i.velocity, pitch=i.pitch))
                state_add = 1
                previous_end = i.end
                from_ske = 1
                continue
            if (state_add == 1):
                distance = i.start - previous_end
                if (distance < 240):
                    if (check_ske == 1 and from_prol == 1):
                        item_list.append(miditoolkit.Note(start=i.start, end=i.end, velocity=i.velocity, pitch=i.pitch))
                        state_add = 0
                        from_prol = 0
                        stage3_unit_list.append(item_list)
                        item_list = list()
                    if (check_ske == 1 and from_ske == 1 and (length < 1890 or length > 1950)):
                        stage3_unit_list.append(item_list)
                        item_list = list()
                        item_list.append(miditoolkit.Note(start=i.start, end=i.end, velocity=i.velocity, pitch=i.pitch))
                        from_ske = 1
                    if (check_ske == 1 and from_ske == 1 and (length >= 1890 and length <= 1950)):
                        item_list.append(miditoolkit.Note(start=i.start, end=i.end, velocity=i.velocity, pitch=i.pitch))
                        state_add = 0
                        from_ske = 0
                        stage3_unit_list.append(item_list)
                        item_list = list()
                    if (check_prol == 1 and (length < 1890 or length > 1950)):
                        item_list.append(miditoolkit.Note(start=i.start, end=i.end, velocity=i.velocity, pitch=i.pitch))
                    if (check_prol == 1 and (length >= 1890 and length <= 1950)):
                        item_list.append(miditoolkit.Note(start=i.start, end=i.end, velocity=i.velocity, pitch=i.pitch))
                        state_add = 0
                        from_prol = 0
                        from_ske = 0
                        stage3_unit_list.append(item_list)
                        item_list = list()
                else:
                    #                     pprint.pprint(item_list)
                    #                     print("\n+++\n")
                    stage3_unit_list.append(item_list)
                    item_list = list()

                    item_list.append(miditoolkit.Note(start=i.start, end=i.end, velocity=i.velocity, pitch=i.pitch))

                    if (check_ske == 1):
                        from_ske = 1
                        from_prol = 0
                    if (check_prol == 1):
                        from_prol = 1
                        from_ske = 0
            previous_end = i.end
            # pprint.pprint(stage3_unit_list)

        #         for i in stage3_unit_list:
        #             pprint.pprint(i)
        #             print('\n')
        # pprint.pprint(stage3_unit_list)
        loss_prol = []
        stage3_unit_final = []
        for i in stage3_unit_list:
            flag = 0
            for j in i:
                if j.velocity == 127:
                    flag = 1
            #             pprint.pprint(i)
            #             print(flag)
            if flag == 0:
                for m in i:
                    loss_prol.append(miditoolkit.Note(start=m.start, end=m.end, velocity=m.velocity, pitch=m.pitch))
            else:
                stage3_unit_final.append(i)

        #         for i in stage3_unit_final:
        #             pprint.pprint(i)
        #             print('\n')
        for i in stage3_unit_final:
            if len(i) == 1:
                loss_prol.append(
                    miditoolkit.Note(start=i[0].start, end=i[0].end, velocity=i[0].velocity, pitch=i[0].pitch))
                stage3_unit_final.remove(i)
        loss_prol.sort(key=lambda x: (x.start, -x.end))
        # pprint.pprint(loss_prol)
        #         for i in stage3_unit_final:
        #             pprint.pprint(i)
        #             print('\n')
        start_end = []
        for i in stage3_unit_final:
            start = 999999
            end = 0
            index = 0
            for j in i:
                if j.start < start:
                    start = j.start
                if j.end > end:
                    end = j.end
            start_end.append((start, end, index))
            index = index + 1
        # pprint.pprint(start_end[0][0])
        for j in range(len(loss_prol)):
            left_distance = []
            right_distance = []
            start_end = []
            for p in stage3_unit_final:
                pstart = 999999
                pend = 0
                pindex = 0
                for q in p:
                    if q.start < pstart:
                        pstart = q.start
                    if q.end > pend:
                        pend = q.end
                start_end.append((pstart, pend, pindex))
                pindex = pindex + 1
            for i in loss_prol:
                start = i.start
                end = i.end
                count1 = 0
                count2 = 0
                left = 1000000
                find_left = 0
                find_right = 0
                for k in start_end:

                    if k[1] <= start:
                        left = start - k[1]
                        find_left = count1

                    if k[0] >= end:
                        right = k[0] - end
                        find_right = count2
                        break
                    count1 = count1 + 1
                    count2 = count2 + 1

                left_distance.append((left, find_left))
                right_distance.append((right, find_right))
            m_left = 999999

            n_right = 999999
            count3 = 0

            #             pprint.pprint(left_distance)
            #             pprint.pprint(right_distance)
            for m in left_distance:
                if m[0] < m_left:
                    m_left = m[0]
                    m_index = m[1]
                    index3 = count3

                count3 = count3 + 1
            count4 = 0

            for n in right_distance:
                if n[0] < n_right:
                    n_right = n[0]
                    n_index = n[1]
                    index4 = count4
                count4 = count4 + 1
            if m_left <= n_right:
                stage3_unit_final[m_index].append(miditoolkit.Note(start=loss_prol[index3].start,
                                                                   end=loss_prol[index3].end,
                                                                   velocity=loss_prol[index3].velocity,
                                                                   pitch=loss_prol[index3].pitch))
                loss_prol.remove(loss_prol[index3])
            else:
                stage3_unit_final[n_index].append(miditoolkit.Note(start=loss_prol[index4].start,
                                                                   end=loss_prol[index4].end,
                                                                   velocity=loss_prol[index4].velocity,
                                                                   pitch=loss_prol[index4].pitch))
                loss_prol.remove(loss_prol[index4])
        for i in stage3_unit_final:
            i.sort(key=lambda x: (x.start, -x.end))
            pprint.pprint(i)
            print('\n')
        print('*****stage3 Over*****')



        print('*****stage4*****')
        rps_list = []
        rps_list_first_type = []
        alter_list = []
        first_step_list = []
        for i in stage3_unit_final:
            if len(i) == 2 or len(i) == 3:
                rps_list.append(i)
                if len(i) == 2:
                    item_list = [i[0].start % 1920, i[0].end % 1920, i[1].start % 1920, i[1].end % 1920]
                    rps_list_first_type.append(item_list)
                else:
                    item_list = [i[0].start % 1920, i[0].end % 1920, i[1].start % 1920, i[1].end % 1920,
                                 i[2].start % 1920, i[2].end % 1920]
                    rps_list_first_type.append(item_list)
            elif len(i) == 4:
                first_step_list.append(i)
            elif len(i) > 4:
                # pprint.pprint(i)
                alter_list.append(i)

        # pprint.pprint(alter_list)
        for i in alter_list:
            #             pprint.pprint(i)
            #             print('\n')
            start1 = i[0].start
            end1 = i[0].end
            start2 = i[1].start
            end2 = i[1].end
            start3 = i[2].start
            end3 = i[2].end
            flag = 0
            for j in rps_list_first_type:
                if len(j) == 6 and len(i) > 4:
                    if (j[0] == start1 % 1920 and j[1] == end1 % 1920
                            and j[2] == start2 % 1920 and j[3] == end2 % 1920
                            and j[4] == start3 % 1920 and j[5] == end3 % 1920):
                        item_list = []
                        item_list.append(i[0])
                        item_list.append(i[1])
                        item_list.append(i[2])
                        first_step_list.append(item_list)

                        first_step_list.append(i[3:])
                        flag = 1
                        break
                elif len(j) == 4 and len(i) > 4:
                    if (j[0] == start1 % 1920 and j[1] == end1 % 1920
                            and j[2] == start2 % 1920 and j[3] == end2 % 1920):
                        item_list = []
                        item_list.append(i[0])
                        item_list.append(i[1])
                        first_step_list.append(item_list)
                        first_step_list.append(i[2:])
                        flag = 1
                        break
            if flag == 0:
                first_step_list.append(i)

        first_step_list.sort(key=lambda x: (x[0].start, -x[0].end))
        print(">>>>>>>>")
        pprint.pprint(first_step_list)
        # pprint.pprint(rps_list)
        for i in first_step_list:
            if len(i) == 2 or len(i) == 3:
                rps_list.append(i)
            if len(i) == 4:
                count = 0
                item_list = []
                for j in i:
                    item_list.append(j)
                    count = count + 1
                    if (count == 2):
                        rps_list.append(item_list)
                        item_list = list()
                rps_list.append(item_list)
            if len(i) == 5:
                # pprint.pprint(i)
                start1 = i[0].start
                end1 = i[0].end
                start2 = i[1].start
                end2 = i[1].end
                start3 = i[2].start
                end3 = i[2].end
                item_list = []
                count_two = 0
                count_three = 0
                # print(count_two,count_three)
                for j in rps_list:
                    if (len(j) == 2):
                        if (j[0].start % 1920 == start1 % 1920 and j[0].end % 1920 == end1 % 1920
                                and j[1].start % 1920 == start2 % 1920 and j[1].end % 1920 == end2 % 1920):
                            count_two = count_two + 1
                    if (len(j) == 3):
                        if (j[0].start % 1920 == start1 % 1920 and j[0].end % 1920 == end1 % 1920
                                and j[1].start % 1920 == start2 % 1920 and j[1].end % 1920 == end2 % 1920
                                and j[2].start % 1920 == start3 % 1920 and j[2].end % 1920 == end3 % 1920):
                            count_three = count_three + 1
                # print(count_two,count_three)
                if count_two >= count_three:
                    item_list.append(i[0])
                    item_list.append(i[1])
                    rps_list.append(item_list)
                    item_list = list()
                    item_list.append(i[2])
                    item_list.append(i[3])
                    item_list.append(i[4])
                    rps_list.append(item_list)
                if count_three > count_two:
                    item_list.append(i[0])
                    item_list.append(i[1])
                    item_list.append(i[2])
                    rps_list.append(item_list)
                    item_list = list()

                    item_list.append(i[3])
                    item_list.append(i[4])
                    rps_list.append(item_list)
            if len(i) == 6:
                start1 = i[0].start
                end1 = i[0].end
                start2 = i[1].start
                end2 = i[1].end
                start3 = i[2].start
                end3 = i[2].end
                item_list = []
                count_two = 0
                count_three = 0
                # print(count_two,count_three)
                for j in rps_list:
                    if (len(j) == 2):
                        if (j[0].start % 1920 == start1 % 1920 and j[0].end % 1920 == end1 % 1920
                                and j[1].start % 1920 == start2 % 1920 and j[1].end % 1920 == end2 % 1920):
                            count_two = count_two + 1
                    if (len(j) == 3):
                        if (j[0].start % 1920 == start1 % 1920 and j[0].end % 1920 == end1 % 1920
                                and j[1].start % 1920 == start2 % 1920 and j[1].end % 1920 == end2 % 1920
                                and j[2].start % 1920 == start3 % 1920 and j[2].end % 1920 == end3 % 1920):
                            count_three = count_three + 1
                # print(count_two,count_three)
                if count_two >= count_three:
                    item_list.append(i[0])
                    item_list.append(i[1])
                    rps_list.append(item_list)
                    item_list = list()
                    item_list.append(i[2])
                    item_list.append(i[3])
                    rps_list.append(item_list)
                    item_list = list()
                    item_list.append(i[4])
                    item_list.append(i[5])
                    rps_list.append(item_list)
                if count_three > count_two:
                    item_list.append(i[0])
                    item_list.append(i[1])
                    item_list.append(i[2])
                    rps_list.append(item_list)
                    item_list = list()
                    item_list.append(i[3])
                    item_list.append(i[4])
                    item_list.append(i[5])
                    rps_list.append(item_list)
            if len(i) == 7:
                start1 = i[0].start
                end1 = i[0].end
                start2 = i[1].start
                end2 = i[1].end
                start3 = i[2].start
                end3 = i[2].end
                item_list = []
                count_two = 0
                count_three = 0
                for j in rps_list:
                    if (len(j) == 2):
                        if (j[0].start % 1920 == start1 % 1920 and j[0].end % 1920 == end1 % 1920
                                and j[1].start % 1920 == start2 % 1920 and j[1].end % 1920 == end2 % 1920):
                            count_two = count_two + 1
                    if (len(j) == 3):
                        if (j[0].start % 1920 == start1 % 1920 and j[0].end % 1920 == end1 % 1920
                                and j[1].start % 1920 == start2 % 1920 and j[1].end % 1920 == end2 % 1920
                                and j[2].start % 1920 == start3 % 1920 and j[2].end % 1920 == end3 % 1920):
                            count_three = count_three + 1
                # print(count_two,count_three)
                if count_two >= count_three:
                    item_list.append(i[0])
                    item_list.append(i[1])
                    rps_list.append(item_list)
                    start4 = i[2].start
                    end4 = i[2].end
                    start5 = i[3].start
                    end5 = i[3].end
                    start6 = i[4].start
                    end6 = i[4].end
                    count_two_deep = 0
                    count_three_deep = 0
                    for j in rps_list:
                        if (len(j) == 2):
                            if (j[0].start % 1920 == start4 % 1920 and j[0].end % 1920 == end4 % 1920
                                    and j[1].start % 1920 == start5 % 1920 and j[1].end % 1920 == end5 % 1920):
                                count_two_deep = count_two_deep + 1
                        if (len(j) == 3):
                            if (j[0].start % 1920 == start4 % 1920 and j[0].end % 1920 == end4 % 1920
                                    and j[1].start % 1920 == start5 % 1920 and j[1].end % 1920 == end5 % 1920
                                    and j[2].start % 1920 == start6 % 1920 and j[2].end % 1920 == end6 % 1920):
                                count_three_deep = count_three_deep + 1
                    if count_two_deep >= count_three_deep:
                        item_list = list()
                        item_list.append(i[2])
                        item_list.append(i[3])
                        rps_list.append(item_list)
                        item_list = list()
                        item_list.append(i[4])
                        item_list.append(i[5])
                        item_list.append(i[6])
                        rps_list.append(item_list)
                    else:
                        item_list = list()
                        item_list.append(i[2])
                        item_list.append(i[3])
                        item_list.append(i[4])
                        rps_list.append(item_list)
                        item_list = list()

                        item_list.append(i[5])
                        item_list.append(i[6])
                        rps_list.append(item_list)
                if count_three > count_two:
                    item_list.append(i[0])
                    item_list.append(i[1])
                    item_list.append(i[2])
                    rps_list.append(item_list)
                    count_count = 0
                    item_list = []
                    for j in i[3:]:
                        item_list.append(j)
                        count_count = count_count + 1
                        if (count_count == 2):
                            rps_list.append(item_list)
                            item_list = list()
                    rps_list.append(item_list)
            if len(i) == 8:
                start1 = i[0].start
                end1 = i[0].end
                start2 = i[1].start
                end2 = i[1].end
                start3 = i[2].start
                end3 = i[2].end
                item_list = []
                count_two = 0
                count_three = 0
                for j in rps_list:
                    if (len(j) == 2):
                        if (j[0].start % 1920 == start1 % 1920 and j[0].end % 1920 == end1 % 1920
                                and j[1].start % 1920 == start2 % 1920 and j[1].end % 1920 == end2 % 1920):
                            count_two = count_two + 1
                    if (len(j) == 3):
                        if (j[0].start % 1920 == start1 % 1920 and j[0].end % 1920 == end1 % 1920
                                and j[1].start % 1920 == start2 % 1920 and j[1].end % 1920 == end2 % 1920
                                and j[2].start % 1920 == start3 % 1920 and j[2].end % 1920 == end3 % 1920):
                            count_three = count_three + 1
                # print(count_two,count_three)
                if count_two >= count_three:
                    item_list.append(i[0])
                    item_list.append(i[1])
                    rps_list.append(item_list)
                    start4 = i[2].start
                    end4 = i[2].end
                    start5 = i[3].start
                    end5 = i[3].end
                    start6 = i[4].start
                    end6 = i[4].end
                    count_two_deep = 0
                    count_three_deep = 0
                    for j in rps_list:
                        if (len(j) == 2):
                            if (j[0].start % 1920 == start4 % 1920 and j[0].end % 1920 == end4 % 1920
                                    and j[1].start % 1920 == start5 % 1920 and j[1].end % 1920 == end5 % 1920):
                                count_two_deep = count_two_deep + 1
                        if (len(j) == 3):
                            if (j[0].start % 1920 == start4 % 1920 and j[0].end % 1920 == end4 % 1920
                                    and j[1].start % 1920 == start5 % 1920 and j[1].end % 1920 == end5 % 1920
                                    and j[2].start % 1920 == start6 % 1920 and j[2].end % 1920 == end6 % 1920):
                                count_three_deep = count_three_deep + 1
                    if count_two_deep >= count_three_deep:
                        item_list = list()
                        item_list.append(i[2])
                        item_list.append(i[3])
                        rps_list.append(item_list)
                        item_list = list()
                        item_list.append(i[4])
                        item_list.append(i[5])
                        rps_list.append(item_list)
                        item_list = list()
                        item_list.append(i[6])
                        item_list.append(i[7])
                        rps_list.append(item_list)
                    else:
                        item_list = list()
                        item_list.append(i[2])
                        item_list.append(i[3])
                        item_list.append(i[4])
                        rps_list.append(item_list)
                        item_list = list()

                        item_list.append(i[5])
                        item_list.append(i[6])
                        item_list.append(i[7])
                        rps_list.append(item_list)
                if count_three > count_two:
                    item_list.append(i[0])
                    item_list.append(i[1])
                    item_list.append(i[2])
                    rps_list.append(item_list)
                    start4 = i[3].start
                    end4 = i[3].end
                    start5 = i[4].start
                    end5 = i[4].end
                    start6 = i[5].start
                    end6 = i[5].end
                    count_two_deep = 0
                    count_three_deep = 0
                    for j in rps_list:
                        if (len(j) == 2):
                            if (j[0].start % 1920 == start4 % 1920 and j[0].end % 1920 == end4 % 1920
                                    and j[1].start % 1920 == start5 % 1920 and j[1].end % 1920 == end5 % 1920):
                                count_two_deep = count_two_deep + 1
                        if (len(j) == 3):
                            if (j[0].start % 1920 == start4 % 1920 and j[0].end % 1920 == end4 % 1920
                                    and j[1].start % 1920 == start5 % 1920 and j[1].end % 1920 == end5 % 1920
                                    and j[2].start % 1920 == start6 % 1920 and j[2].end % 1920 == end6 % 1920):
                                count_three_deep = count_three_deep + 1
                    if count_two_deep >= count_three_deep:
                        item_list = list()
                        item_list.append(i[3])
                        item_list.append(i[4])
                        rps_list.append(item_list)
                        item_list = list()
                        item_list.append(i[5])
                        item_list.append(i[6])
                        item_list.append(i[7])
                        rps_list.append(item_list)
                    else:
                        item_list = list()
                        item_list.append(i[3])
                        item_list.append(i[4])

                        rps_list.append(item_list)
                        item_list = list()

                        item_list.append(i[5])
                        item_list.append(i[6])
                        item_list.append(i[7])
                        rps_list.append(item_list)
        rps_list.sort(key=lambda x: (x[0].start, -x[0].end))
        # print(">>>>>>>>>>", rps_list)
        # for i in rps_list:
        #     pprint.pprint(i)
        #     print('\n')
        # try:
        #     nbar = math.floor(rps_list[0][0].start / 1920) + 1
        # except:
        #     print(file_name)
        #     exit()
        stage4_first = 1
        previous_bar = 1
        rps_type_list = []
        rps_dict = {}
        count_type = 0
        print("===>",len(rps_list))
        for k in range(len(rps_list)):
            start = rps_list[k][0].start

            nbar = math.floor(rps_list[k][0].start / 1920) + 1
            if nbar > previous_bar:
                print('RS_bar%d:' % (nbar), end='\n')
                rps_dict[f'RS_bar{nbar}'] = []

            if stage4_first == 1:
                print('RS_bar%d:' % (nbar), end='\n')
                rps_dict[f'RS_bar{nbar}'] = []
                stage4_first = 0

            if (start >= (nbar - 1) * 1920 and start < nbar * 1920):
                index_rps = 0
                find_index_rps = 0
                for i in rps_type_list:
                    index_rps = index_rps + 1
                    if (len(rps_list[k]) == 2):
                        if (len(i) == 3):
                            if (rps_list[k][0].end - rps_list[k][0].start == i[0] and
                                    rps_list[k][1].start - rps_list[k][0].end == i[1] and
                                    rps_list[k][1].end - rps_list[k][1].start == i[2]):
                                find_index_rps = 1
                                break

                    else:
                        if (len(i) == 5):
                            if (rps_list[k][0].end - rps_list[k][0].start == i[0] and
                                    rps_list[k][1].start - rps_list[k][0].end == i[1] and
                                    rps_list[k][1].end - rps_list[k][1].start == i[2] and
                                    rps_list[k][2].start - rps_list[k][1].end == i[3] and
                                    rps_list[k][2].end - rps_list[k][2].start == i[4]):
                                find_index_rps = 1
                                break

                if (find_index_rps == 1):
                    print('\tRPS%d: ' % (index_rps), end="")

                    if (len(rps_list[k]) == 2):
                        print('2[note1(start=%d,end=%d),note2(start=%d,end=%d)]' % (rps_list[k][0].start,
                                                                                    rps_list[k][0].end,
                                                                                    rps_list[k][1].start,
                                                                                    rps_list[k][1].end))
                        rps_dict[f'RS_bar{nbar}'].append([f"RPS{index_rps}", len(rps_list[k]), [(rps_list[k][0].start,rps_list[k][0].end,),(rps_list[k][1].start,rps_list[k][1].end)]])

                    else:
                        print('3[note1(start=%d,end=%d),note2(start=%d,end=%d),note3(start=%d,end=%d)]' % (
                            rps_list[k][0].start,
                            rps_list[k][0].end,
                            rps_list[k][1].start,
                            rps_list[k][1].end,
                            rps_list[k][2].start,
                            rps_list[k][2].end))
                        rps_dict[f'RS_bar{nbar}'].append([f"RPS{index_rps}", 3,
                                                          [(rps_list[k][0].start, rps_list[k][0].end,),
                                                           (rps_list[k][1].start, rps_list[k][1].end),
                                                           (rps_list[k][2].start, rps_list[k][2].end)]
                                                          ])




                else:

                    if (len(rps_list[k]) == 2):
                        item_list = [rps_list[k][0].end - rps_list[k][0].start,
                                     rps_list[k][1].start - rps_list[k][0].end,
                                     rps_list[k][1].end - rps_list[k][1].start]
                    if (len(rps_list[k]) == 3):
                        item_list = [rps_list[k][0].end - rps_list[k][0].start,
                                     rps_list[k][1].start - rps_list[k][0].end,
                                     rps_list[k][1].end - rps_list[k][1].start,
                                     rps_list[k][2].start - rps_list[k][1].end,
                                     rps_list[k][2].end - rps_list[k][2].start]
                    rps_type_list.append(item_list)
                    count_type = count_type + 1
                    print('\tRPS%d: ' % (count_type), end="")

                    if (len(rps_list[k]) == 2):
                        print('2[note1(start=%d,end=%d),note2(start=%d,end=%d)]' % (rps_list[k][0].start,
                                                                                    rps_list[k][0].end,
                                                                                    rps_list[k][1].start,
                                                                                    rps_list[k][1].end))
                        rps_dict[f'RS_bar{nbar}'].append([f"RPS{index_rps}", len(rps_list[k]),
                                                          [(rps_list[k][0].start, rps_list[k][0].end,),
                                                           (rps_list[k][1].start, rps_list[k][1].end)]])
                    else:
                        print('3[note1(start=%d,end=%d),note2(start=%d,end=%d),note3(start=%d,end=%d)]' % (
                            rps_list[k][0].start,
                            rps_list[k][0].end,
                            rps_list[k][1].start,
                            rps_list[k][1].end,
                            rps_list[k][2].start,
                            rps_list[k][2].end))

                        rps_dict[f'RS_bar{nbar}'].append([f"RPS{index_rps}", 3,
                                                          [(rps_list[k][0].start, rps_list[k][0].end,),
                                                           (rps_list[k][1].start, rps_list[k][1].end),
                                                           (rps_list[k][2].start, rps_list[k][2].end)]
                                                          ])

            previous_bar = nbar



        count_list = [0] * 1000
        print(len(rps_list))
        for i in rps_list:
            cc = 0
            for j in rps_type_list:
                if len(i) == 2 and len(j) == 3:
                    if (i[0].end - i[0].start == j[0] and i[1].start - i[0].end == j[1]
                            and i[1].end - i[1].start == j[2]):
                        count_list[cc] = count_list[cc] + 1
                        break
                if len(i) == 3 and len(j) == 5:
                    if (i[0].end - i[0].start == j[0] and i[1].start - i[0].end == j[1]
                            and i[1].end - i[1].start == j[2] and i[2].start - i[1].end == j[3]
                            and i[2].end - i[2].start == j[4]):
                        count_list[cc] = count_list[cc] + 1
                        break
                cc = cc + 1
        kk = 0
        rps_type_list_dict_temp = {}
        for i in rps_type_list:
            print('RPS%d:' % (kk + 1), end="")
            print(i, end='\tcount=%d\n' % count_list[kk])
            rps_type_list_dict_temp[f"RPS{kk + 1}:{i}"] = f"count={count_list[kk]}"

            kk = kk + 1
        ''' 此处不能删， 为另一种格式的输出'''
        #         for k in range(len(rps_list)):
        #             start=rps_list[k][0].start

        #             nbar=math.floor(rps_list[k][0].start/1920)+1
        #             if nbar>previous_bar:

        #                 print('RS_bar%d:'%(nbar),end='\n')

        #             if stage4_first==1:
        #                 print('RS_bar%d:'%(nbar),end='\n')
        #                 stage4_first=0

        #             if(start>=(nbar-1)*1920 and start<nbar*1920):
        #                 index_rps=0
        #                 find_index_rps=0
        #                 for i in rps_type_list:
        #                     index_rps=index_rps+1
        #                     if(len(rps_list[k])==2):
        #                         if(len(i)==4):
        #                             if(rps_list[k][0].start%1920==i[0] and
        #                               rps_list[k][0].end%1920==i[1] and
        #                               rps_list[k][1].start%1920==i[2] and
        #                               rps_list[k][1].end%1920==i[3]):
        #                                 find_index_rps=1
        #                                 break

        #                     else:
        #                         if(len(i)==6):
        #                             if(rps_list[k][0].start%1920==i[0] and
        #                               rps_list[k][0].end%1920==i[1] and
        #                               rps_list[k][1].start%1920==i[2] and
        #                               rps_list[k][1].end%1920==i[3] and
        #                               rps_list[k][2].start%1920==i[4] and
        #                               rps_list[k][2].end%1920==i[5]):
        #                                 find_index_rps=1
        #                                 break

        #                 if(find_index_rps==1):
        #                     print('\tRPS%d: '%(index_rps),end="")
        #                 else:

        #                     if(len(rps_list[k])==2):
        #                         item_list=[rps_list[k][0].start%1920,rps_list[k][0].end%1920,
        #                                   rps_list[k][1].start%1920,rps_list[k][1].end%1920]
        #                     if(len(rps_list[k])==3):
        #                         item_list=[rps_list[k][0].start%1920,rps_list[k][0].end%1920,
        #                                   rps_list[k][1].start%1920,rps_list[k][1].end%1920,
        #                                   rps_list[k][2].start%1920,rps_list[k][2].end%1920]
        #                     rps_type_list.append(item_list)
        #                     count_type=count_type+1
        #                     print('\tRPS%d: '%(count_type),end="")
        #                 if(len(rps_list[k])==2):
        #                     print('2[note1(start=%d,end=%d),note2(start=%d,end=%d)]'%(rps_list[k][0].start,
        #                                                                              rps_list[k][0].end,
        #                                                                              rps_list[k][1].start,
        #                                                                              rps_list[k][1].end))
        #                 else:
        #                     print('3[note1(start=%d,end=%d),note2(start=%d,end=%d),note3(start=%d,end=%d)]'%(rps_list[k][0].start,
        #                                                                              rps_list[k][0].end,
        #                                                                              rps_list[k][1].start,
        #                                                                              rps_list[k][1].end,
        #                                                                              rps_list[k][2].start,
        #                                                                              rps_list[k][2].end))

        #             previous_bar=nbar
        #         count_list=[0]*100
        #         for i in rps_list:
        #             cc=0
        #             for j in rps_type_list:
        #                 if len(i)==2 and len(j)==4:
        #                     if(i[0].start%1920==j[0] and i[0].end%1920==j[1]
        #                       and i[1].start%1920==j[2] and i[1].end%1920==j[3]):
        #                         count_list[cc]=count_list[cc]+1
        #                         break
        #                 if len(i)==3 and len(j)==6:
        #                     if(i[0].start%1920==j[0] and i[0].end%1920==j[1]
        #                       and i[1].start%1920==j[2] and i[1].end%1920==j[3]
        #                       and i[2].start%1920==j[4] and i[2].end%1920==j[5]):
        #                         count_list[cc]=count_list[cc]+1
        #                         break
        #                 cc=cc+1
        #         kk=0
        #         for i in rps_type_list:
        #             print('RPS%d:'%(kk+1),end="")
        #             print(i,end='\tcount=%d\n'%count_list[kk])

        #             kk=kk+1

        return pattern_final, rps_dict, rps_type_list_dict_temp



if __name__ == '__main__':
    from tqdm import tqdm
    rootdir = '/Users/xinda/Documents/Github/MDP/data/process/paper_skeleton/zhpop/8_melody_filter/zhpop'
    listfile = os.listdir(rootdir)
    for i in tqdm(range(0, len(listfile))):
        midi_path = os.path.join(rootdir, listfile[i])
        file_name = os.path.basename(midi_path)
        dst_path = './output_test'
        m = Melody_Skeleton_Extractor(midi_path)      # midi对象
        skeleton_melody_notes_list, rps_dict, rps_type_list_dict_temp = m.get_skeleton()  # midi的旋律骨架
        print("=============")
        print(skeleton_melody_notes_list)
        print("=============")
        print(rps_dict)
        break


