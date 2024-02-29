# Data Preprocessing

### **1 Datasets (with chord progression)**

- Wikifonia
- Hooktheory DB

The chord representation of the two datasets has been unified by us. If the number of chord labels is less than 10% of the bars, the file is discarded.

Please unzip datasets in ```./preprocessing/data/raw/WuYun-datasets_withChord.zip```



### **2. Chord Representation**

```Root__Quality__ChordDegree```. For example, 7\_\_major\_\_{2, 11, 7}.

pitch_name_mapping = {'C':0, 'C#':1,  'D-':1,  'Db':1,  'D':2, 'D#':3,   'Eb':3,  'E-':3, 'E':4, 'F':5, 'F#':6,  'Gb':6,  'G-':6, 'G':7, 'G#':8, 'Ab':8,  'A-':8, 'A':9,   'A#':10,'Bb':10,  'B-':10, 'B':11}

### 3. Preprocessing Pipelines

1. select 4/4 ts ( requirement >= 8 bars )
2. extract melody
3. midi quantization (base and triplets)
4. ~~tonality normalization (abandoned)~~
5. shift pitch range (C3-C5)
6. filter midis by heuristic rules
7. internal dedup by pitch interval 

please read ```mdp_wuyun.py``` for more details. The processing results for each dataset are saved in their respective directories.

```bash
cd preprocessing
python3 mdp_wuyun.py
```



### 4. Split Dataset
Split the dataset in your own way, such as 
- 9:1
- 8:1:1
- Leave out 100 musical pieces (our).

