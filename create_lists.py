"""Creates file lists used for quantitative evaluation"""

from pathlib import Path
import re

def get_files(dir):
    files = []
    for p in dir.rglob('*'):
        if p.is_file():
            name_to_save = re.sub('/mnt/dsi_vol1/users/yochai_yemini/REVERB/SimData/REVERB_WSJCAM0_et/data', '', str(p))
            name_to_save = re.sub('/mnt/dsi_vol1/users/yochai_yemini/REVERBsim/REVERB_WSJCAM0_et/data', '', name_to_save)
            files.append(name_to_save)
    return sorted(files)


def write_to_file(files, output_file):
    with open(output_file, 'w') as f:
        f.write("\n".join(files))


mics_num = 8
clean_dir = Path('/mnt/dsi_vol1/users/yochai_yemini/REVERB/SimData/REVERB_WSJCAM0_et/data/cln_test2')
files_clean = get_files(clean_dir)
write_to_file(files_clean, 'taskfiles/1ch/SimData_et_for_cln_room1')

base_dir = Path('/mnt/dsi_vol1/users/yochai_yemini/REVERBsim/REVERB_WSJCAM0_et/data')
reverb_dir = base_dir / 'far_test'
files_far = get_files(reverb_dir)
files_far = files_far[::mics_num]
write_to_file(files_far, 'taskfiles/1ch/SimData_et_for_1ch_far_room1_A')

reverb_dir = base_dir / 'near_test'
files_near = get_files(reverb_dir)
files_near = files_near[::mics_num]
write_to_file(files_near, 'taskfiles/1ch/SimData_et_for_1ch_near_room1_A')

reverb_dir = base_dir / 'random_test'
files_near = get_files(reverb_dir)
files_near = files_near[::mics_num]
write_to_file(files_near, 'taskfiles/1ch/SimData_et_for_1ch_random_room1_A')

reverb_dir = base_dir / 'far_one_near_test'
files_near = get_files(reverb_dir)
files_near = files_near[::mics_num]
write_to_file(files_near, 'taskfiles/SimData_et_for_winning_ticket')