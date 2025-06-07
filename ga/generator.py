import numpy as np
from typing import List
from globals import SLOTS_PER_DAY
from dataframes.curriculum import Curriculum

def generate_valid_guess(curriculum: Curriculum, time_slot_indices: List[int], room_indices: List[int]):
    T = len(time_slot_indices)
    R = len(room_indices)

    total_durations = (curriculum.df["classes"] * curriculum.df["credits"]).sum()
    total_slots = T * R
    
    if total_durations > total_slots:
        raise ValueError("Not enough slots for all courses")

    arr = np.full((T, R), fill_value=0, dtype=np.int16)

    classes_dict = curriculum.df[['id', 'classes', 'credits']].set_index('id').to_dict(orient='index')
    days_count = T // SLOTS_PER_DAY
    days_indices = list(range(days_count))
    time_slots = list(range(SLOTS_PER_DAY))

    for id_, info in classes_dict.items():
        classes = info['classes']
        credits = info['credits']
        np.random.shuffle(days_indices)
        np.random.shuffle(time_slots)
        np.random.shuffle(room_indices)
        
        session_fill = 1
        used_time = set()
        for d in days_indices:
            placed = False
            class_fill = 1
            for t in time_slots:

                row = d * SLOTS_PER_DAY + t
                if row in used_time:
                    continue

                for r in room_indices:
                    col = r
                    if arr[row, col] != 0:
                        continue

                    arr[row, col] = 100*id_ + 10*class_fill + session_fill
                    if class_fill == classes:
                        placed = True
                        break

                    class_fill += 1

                if placed:
                    used_time.add(row)
                    break
            
            if not placed or class_fill < classes:
                raise ValueError(f"Not enough slots for course {id_}")
            
            if session_fill == credits:
                break

            session_fill += 1

    return arr