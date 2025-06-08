import numpy as np
from math import lcm
from collections import defaultdict
from typing import List, Tuple, Dict

class ParallelClass:
    def __init__(self, chromosome: np.ndarray, parallel_counts: Tuple[int, ...]):
        self.chromosome = chromosome
        self.parallel_counts = parallel_counts
        self.rows, self.cols = chromosome.shape
        self.class_dict = self._extract_parallel_classes()

    def _extract_parallel_classes(self) -> Dict[int, Dict[int, List[Tuple[int, int, int]]]]:
        class_dict = defaultdict(lambda: defaultdict(list))
        for i in range(self.rows):
            for j in range(self.cols):
                val = self.chromosome[i, j]
                if val == 0:
                    continue
                subject = val // 100
                parallel = (val % 100) // 10
                class_dict[subject][parallel].append((i, j, val))
        return class_dict

    def get_all_schedule_matrices(self) -> List[np.ndarray]:
        subjects = sorted(self.class_dict.keys())

        if len(subjects) != len(self.parallel_counts):
            raise ValueError("Mismatch between number of subjects and parallel_counts")

        # Get sorted parallel session lists for each subject
        parallel_lists = []
        for subject in subjects:
            subject_parallels = self.class_dict[subject]
            parallel_lists.append([
                subject_parallels[pid] for pid in sorted(subject_parallels)
            ])

        # Compute LCM of all parallel counts
        total_lcm = self.parallel_counts[0]
        for count in self.parallel_counts[1:]:
            total_lcm = lcm(total_lcm, count)

        configs = []
        for i in range(total_lcm):
            mat = np.zeros((self.rows, self.cols), dtype=int)
            for subj_idx, sessions_list in enumerate(parallel_lists):
                num_parallels = self.parallel_counts[subj_idx]
                selected_sessions = sessions_list[i % num_parallels]
                for r, c, val in selected_sessions:
                    mat[r, c] = val
            configs.append(mat)

        return configs
