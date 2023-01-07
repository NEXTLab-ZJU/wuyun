import pprint
import miditoolkit
from itertools import chain

default_resolution = 480
ticks_per_beat = 480  # default resolution = 480 ticks per quarter note, 四分音符 480ticks，十六分音符，120ticks
grid_per_bar = 16
cell = ticks_per_beat * 4 / grid_per_bar
grids_triple = 32
grids_normal = 64


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
    # 1）过滤小于16分音符时值的音符，无切分音意义；
    # 2）根据4分音符，8分音符和16分音符的所有切分音情形进行筛选
    # 3）时间：开始的时间一定要在点上，然后结束的时间在强拍的弱部分，即需要超过强拍时值的一半
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
    # 1) 在4/4拍的MIDI中，当音符的起始位置出现在小节的第一拍或者第三拍上时被认为是节拍重音， 并将该音符加入节拍重音字典。
    # 2) 当切分音出现在节拍重音结束后的3个16分音符的时值内，且该切分音时值大于该节拍重音时值，则节拍重音转变为普通音符，从节拍重音字典中移除。
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
                        '''
                        if (heavy_note.end <= split_note.start < heavy_note.end + 3 * self.step) and \
                                (heavy_note_length < split_note_length):
                            heavy_note_flag = False
                            break
                        '''

                    if heavy_note_flag == False:
                        break

                if heavy_note_flag:
                    heavy_dict_clean[heavy_bar_id].append(heavy_note)

        return heavy_dict_clean


    # ------------------------------------------------------------
    # 类型三：长音
    # 1) 取每个小节中时值最长的音符，加入长音字典；若有多个，均加入；
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
        note_index = 0
        for k, v in self.subsections.items():  # 遍历每个小节的音符
            if k not in skeleton_dict:
                skeleton_dict[k] = []

            for note in v:
                # add note in note_list
                note_object = Note(start=note.start, end=note.end, \
                                   pitch=note.pitch, velocity=note.velocity, index=note_index)
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



def save_rhythm_skeleton(midi_fn, dst_fn):
    m = Melody_Skeleton_Extractor(midi_fn)
    skeleton_melody_notes_list = m.get_skeleton()  # 旋律骨架
    print(f"Rhythm Skeleton:\n {skeleton_melody_notes_list}")

    temp_midi = miditoolkit.MidiFile(midi_fn)
    temp_midi.instruments[0].notes.clear()
    temp_midi.instruments[0].notes.extend(skeleton_melody_notes_list)
    temp_midi.dump(dst_fn)



if __name__ == "__main__":
    import os
    # midi_path = './input_test/freemidi_pop_4.mid'
    # midi_path = '/Users/xinda/Documents/Github/MDP/data/process/paper_skeleton/zhpop/12_6_All_Data/zhpop_melody/zhpop_87.mid'
    midi_path = '/Users/xinda/Documents/Github/MDP/data/process/paper_skeleton/zhpop/12_6_All_Data/zhpop_melody/zhpop_1646.mid'
    dst_path = './output_test'
    m = Melody_Skeleton_Extractor(midi_path)
    print(m._divide_subsections())
    # split_dict = m._get_split()     # 切分音
    # heavy_dict = m._get_stress()    # 节拍重音
    heavy_dict = m._get_stress()    # 节拍重音
    # long_dict = m._get_long()       # 长音
    # skeleton_melody_notes_list = m.get_skeleton()    # 旋律骨架
    # pprint.pprint(split_dict)
    # print(split_dict)
    pprint.pprint(heavy_dict)

    # save midi
    # midi_fn = miditoolkit.MidiFile(midi_path)
    # for i in range(len(midi_fn.instruments)):
    #     midi_fn.instruments[i].notes.clear()
    #
    # if len(skeleton_melody_notes_list) > 0:
    #     midi_fn.instruments[0].notes.extend(skeleton_melody_notes_list)
    #     midi_fn.dump(f"{dst_path}/{os.path.basename(midi_path)}")

