import numpy as np
import argparse
import configparser
import os
import toml
import tkinter as tk
import shutil
from datetime import datetime
import pandas as pd

def ConfigReader(config_file):
    

    # returns argparse object with all the config values
    config = configparser.ConfigParser()
    config.read(config_file)
    config = config['DEFAULT']
    
    parser = argparse.ArgumentParser()
    # for config.positions remove the brackets. and split by comma
    parser.add_argument('--dataset_path', default='nothing yet', type=str)
    parser.add_argument('--positions', default=config['positions'], type=str2list)

    parser.add_argument('--vivo_centroids_file', default=config['vivo_centroids_file'], type=str)
    parser.add_argument('--vitro_centroids_file', default=config['vitro_centroids_file'], type=str)

    parser.add_argument('--unmatched_color', default=config['unmatched_color'], type=tuple_float_type)
    parser.add_argument('--matched_color', default=config['matched_color'], type=tuple_float_type)
    parser.add_argument('--struct_color', default=config['struct_color'], type=tuple_float_type)
    parser.add_argument('--accepted_edge_color', default=config['accepted_edge_color'], type=tuple_float_type)
    parser.add_argument('--rejected_edge_color', default=config['rejected_edge_color'], type=tuple_float_type)
    parser.add_argument('--predicted_edge_color', default=config['predicted_edge_color'], type=tuple_float_type)
    parser.add_argument('--struct_edge_color', default=config['struct_edge_color'], type=tuple_float_type)

    parser.add_argument('--vivo_edge_size', default=config['vivo_edge_size'], type=float)
    parser.add_argument('--vitro_edge_size', default=config['vitro_edge_size'], type=float)
    parser.add_argument('--vitro_point_size', default=config['vitro_point_size'], type=int)
    parser.add_argument('--vivo_point_size', default=config['vivo_point_size'], type=int)

    parser.add_argument('--opacity', default=config['opacity'], type=float)
    parser.add_argument('--symbol', default=config['symbol'], type=str)

    parser.add_argument('--save_match_key', default=config['save_match_key'], type=str)
    parser.add_argument('--save_state_key', default=config['save_state_key'], type=str)
    parser.add_argument('--delete_match_key', default=config['delete_match_key'], type=str)
    parser.add_argument('--run_alignment_key', default=config['run_alignment_key'], type=str)
    parser.add_argument('--validate_prediction_key', default=config['validate_prediction_key'], type=str)
    parser.add_argument('--reject_prediction_key', default=config['reject_prediction_key'], type=str)
    parser.add_argument('--maybe_prediction_key', default=config['maybe_prediction_key'], type=str)
    parser.add_argument('--toggle_visibility_key', default=config['toggle_visibility_key'], type=str)
    parser.add_argument('--run_alignment_all_slices_key', default=config['run_alignment_all_slices_key'], type=str)
    parser.add_argument('--import_predictions_key', default=config['import_predictions_key'], type=str)
    parser.add_argument('--display_pair_key', default=config['display_pair_key'], type=str)

    parser.add_argument('--matcher_name', default='Matcher', type=str)
    parser.add_argument('--matcher_path', default='', type=str)

    args = parser.parse_args()
    
    return args
    

def get_system_state(args):
    revvie_path = quick_dir(args.dataset_path, 'revvie')
    if not os.path.exists(revvie_path + 'system_state.toml'):
        with open(revvie_path + 'system_state.toml', 'w') as f:
            state = {}
            for position in args.positions:
                state[position] = 'pristine'
            toml.dump(state, f)

    else:
        with open(revvie_path + 'system_state.toml', 'r') as f:
            state = toml.load(f)
    
    return state



def load_latest_state(args):
    revvie_path = quick_dir(args.dataset_path, 'revvie')
    centroids_path = quick_dir(args.dataset_path, 'centroids')
    stack_path = quick_dir(args.matcher_path, 'stack')
    
    positions = args.positions


    # check to see if there is a slices folder
    if not os.path.exists(args.matcher_path + '/slices'):
        vitro_points_np = np.loadtxt(centroids_path + args.vitro_centroids_file)
        vitro_points_np[:, [1, 2]] = vitro_points_np[:, [2, 1]]

        slices_path = quick_dir(args.matcher_path , 'slices')

        for position in positions:
            position_path = quick_dir(slices_path, position)
            latest_state_path = quick_dir(position_path, 'latest')
            z = get_trailing_number(position)
            z = int(z)
            
            points = vitro_points_np[vitro_points_np[:, 0] == z]
            true_points = points.copy()

            colors = np.ones_like(points[:, :3])
            colors = colors * args.unmatched_color
            edge_colors = colors.copy()
            points = points[:, :4]
            points = np.hstack((points, colors))
            points = np.hstack((points, edge_colors))


            np.save(latest_state_path + 'displayed_points.npy', points)
            np.savetxt(latest_state_path + 'displayed_points.txt', points, fmt='%.1f')
            
            np.save(latest_state_path + 'true_points.npy', true_points)
            np.savetxt(latest_state_path + 'true_points.txt', true_points, fmt='%.1f')
            
            np.save(latest_state_path + 'matches.npy', np.empty((0, 2), dtype=np.int32))
            np.savetxt(latest_state_path + 'matches.txt', np.empty((0, 2), dtype=np.int32), fmt='%i')

            np.save(latest_state_path + 'predicted.npy', np.empty((0, 2), dtype=np.int32))
            np.savetxt(latest_state_path + 'predicted.txt', np.empty((0, 2), dtype=np.int32), fmt='%i')

            np.save(latest_state_path + 'rejected.npy', np.empty((0, 2), dtype=np.int32))
            np.savetxt(latest_state_path + 'rejected.txt', np.empty((0, 2), dtype=np.int32), fmt='%i')

            np.save(latest_state_path + 'accepted.npy', np.empty((0, 2), dtype=np.int32))
            np.savetxt(latest_state_path + 'accepted.txt', np.empty((0, 2), dtype=np.int32), fmt='%i')

            np.save(latest_state_path + 'vivo_matched_indices.npy', np.empty((0, 1)))
            np.savetxt(latest_state_path + 'vivo_matched_indices.txt', np.empty((0, 1)), fmt='%i')

            np.save(latest_state_path + 'vivo_matched_points.npy', np.empty((0, 10)))
            np.savetxt(latest_state_path + 'vivo_matched_points.txt', np.empty((0, 10)), fmt='%.1f')


        vivo_points = np.loadtxt(centroids_path + args.vivo_centroids_file)


        vivo_points[:, [1, 2]] = vivo_points[:, [2, 1]]
        vivo_points_copy = vivo_points.copy()
        vivo_ids = vivo_points_copy[:, 3]
        scan_ids = np.unique(vivo_points_copy[:, 4])
        unique_ids = np.unique(vivo_ids)

        vivo_id_scan_id_df = pd.DataFrame(vivo_ids, columns=['vivo_id'])
        # add columns for each scan_id
        for scan_id in scan_ids:
            vivo_id_scan_id_df[str(scan_id)] = 0
        
        for index, vivo_id in enumerate(unique_ids):
            vivo_id_scan_id_df.loc[index, 'vivo_id'] = vivo_id
            points = vivo_points_copy[vivo_points_copy[:, 3] == vivo_id]
            points_scan_ids = np.unique(points[:, 4])
            for scan_id in points_scan_ids:
                vivo_id_scan_id_df.loc[index, str(scan_id)] = 1
        vivo_id_scan_id_df.to_csv(centroids_path + 'vivo_id_scan_id_df.csv', index=False)

        vivo_points = vivo_points[:, :4]
        #remove duplicate rows in vivo_points
        #don't use np.unique because it sorts the array
        vivo_points_df = pd.DataFrame(vivo_points)
        vivo_points_df = vivo_points_df.drop_duplicates()
        vivo_points = vivo_points_df.to_numpy()

        vivo_ids = vivo_points[:, 3]

        colors = np.ones((unique_ids.shape[0], 3)) * args.unmatched_color
        padded_colors = np.ones_like(vivo_points[:, :3])
        for i, id in enumerate(vivo_ids):
            padded_colors[i] = colors[unique_ids == id][0]
        
        edge_colors = padded_colors.copy()
        
        vivo_points = np.hstack((vivo_points, padded_colors))
        vivo_points = np.hstack((vivo_points, edge_colors))

        np.save(stack_path + 'vivo_points.npy', vivo_points)
        np.savetxt(stack_path + 'vivo_points.txt', vivo_points, fmt='%.1f')

        struct_vivo_points = np.loadtxt(args.dataset_path  + 'centroids/padded_struct_centroids.txt')

        struct_vivo_points[:, [1, 2]] = struct_vivo_points[:, [2, 1]]

        np.save(stack_path + 'struct_vivo_points.npy', struct_vivo_points)
        np.savetxt(stack_path + 'struct_vivo_points.txt', struct_vivo_points, fmt='%.1f')

        
    vitro_points = np.empty((0, 10))
    slices_path = quick_dir(args.matcher_path, 'slices')
    
    vivo_points = np.load(stack_path + 'vivo_points.npy').astype(np.float32)

    for position in positions:
        position_path = quick_dir(slices_path, position)
        latest_state_path = quick_dir(position_path, 'latest')
        
        displayed_points = np.load(latest_state_path + 'displayed_points.npy').astype(np.float32)
        #true_points = np.load(latest_state_path + 'true_points.npy')
        #matches = np.load(latest_state_path + 'matches.npy')
        #predicted = np.load(latest_state_path + 'predicted.npy')    
        vitro_points = np.vstack((vitro_points, displayed_points))

        vivo_indices = np.loadtxt(latest_state_path + 'vivo_matched_indices.txt').astype(np.int32)
        if len(vivo_indices) > 0:
            vivo_matched_points = np.loadtxt(latest_state_path + 'vivo_matched_points.txt').astype(np.float32)
            vivo_points[vivo_indices] = vivo_matched_points
        else:
            continue


#        try:
#            vivo_points[vivo_indices] = vivo_matched_points

 #       except IndexError:
 #           continue
    return vitro_points, vivo_points

def transfer_matches(src_path, dst_path):
    src_matches = np.loadtxt(src_path + 'geneseq_matches.txt')
    slices_centroids = np.loadtxt(src_path + '../centroids/geneseq_slices_centroids.txt')

    matches_dict = {}
    for match in src_matches:
        vitro_id = match[0]
        vivo_id = match[1]
        point = slices_centroids[slices_centroids[:, 3] == vitro_id]
        slice = 'Pos' + str(int(point[0][0]))
        if slice not in matches_dict:
            matches_dict[slice] = {(vitro_id, vivo_id)}
        else:
            matches_dict[slice].add((vitro_id, vivo_id))
    
    print(matches_dict)
    for slice in matches_dict:
        dst_slice_path = quick_dir(dst_path, slice)
        dst_latest_state_path = quick_dir(dst_slice_path, 'latest')
        matches = np.array(list(matches_dict[slice]))
        np.savetxt(dst_latest_state_path + 'matches.txt', matches, fmt='%i')
        np.save(dst_latest_state_path + 'matches.npy', matches)
            


def create_matcher_profile():
    # creates a pop-up box where user can input their name
    root = tk.Tk()
    root.title('Matcher name')
    canvas = tk.Canvas(root, width=300, height=300)
    canvas.pack()

    entry_box = tk.Entry(root)
    # display text 'Input your name' in the pop-up box
    canvas.create_window(150, 150, window=entry_box)
    # display message
    tk.Label(root, text='So, you want to be a matcher? Tell me your name.').pack()
    # create a button to close the pop-up box

    def get_name():
        global name
        name = entry_box.get()
        # display a message in response
        tk.Label(root, text='Welcome to the team, ' + name + '!').pack()
        root.destroy()

    button = tk.Button(text='Submit my name', command=get_name)
    canvas.create_window(150, 180, window=button)
    root.mainloop()
    return name


### Helper functions

def str2list(v):
    v = v.replace('[', '')
    v = v.replace(']', '')
    v = v.replace("'", '')
    v = v.replace('"', '')
    v = v.replace(' ', '')
    v = v.split(',')
    return v

def tuple_int_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)

def tuple_str_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_str = map(str, strings.split(","))
    return tuple(mapped_str)

def tuple_float_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_float = map(float, strings.split(","))
    return tuple(mapped_float)

def tuple_int_str_type(v):
    v = v.replace('(', '')
    v = v.replace(')', '')
    v = v.replace("'", '')
    v = v.replace('"', '')
    v = v.replace(' ', '')
    v = v.split(',')
    v = (int(v[0]), v[1])
    return v

def list_files(path):
    files = os.listdir(path)
    if '.DS_Store' in files: files.remove('.DS_Store')
    return files

def quick_dir(location, folder_name):
    '''
    Check if a directory exists, otherwise creates one
    :param location:
    :param name:
    :return:
    '''

    if location[-1] == '/': folder_path = location + folder_name + '/'
    else: folder_path = location + '/' + folder_name + '/'

    if not os.path.exists(folder_path): os.makedirs(folder_path, exist_ok=True)

    return folder_path

def get_trailing_number(s):
    '''
    Returns the number at the end of a string. Useful when needing to extract the sequencing cycle number from folder name.
    Input:
        s (string): name containing number
    Returns:
        integer at the end of string
    '''
    import re
    m = re.search(r'\d+$', s)
    return int(m.group()) if m else None


def create_backup(path):
    latest_folder = os.path.join(path, 'latest')
    if os.path.exists(latest_folder):
        now = datetime.now()
        backup_name = now.strftime("%d_%b_%Y_%H_%M_%S")
        backup_path = os.path.join(path, backup_name)
        os.makedirs(backup_path, exist_ok=True)
        for file_name in os.listdir(latest_folder):
            source_path = os.path.join(latest_folder, file_name)
            dest_path = os.path.join(backup_path, file_name)
            shutil.copy2(source_path, dest_path)
    else:
        print(f"The latest folder does not exist in path {path}.")
        print(f"The latest folder does not exist in path {path}.")

def time_travel(path, steps_back=1):
    backup_folders = []
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path) and '_' in item:
            try:
                backup_datetime = datetime.strptime(item, '%d_%b_%Y_%H_%M_%S')
                backup_folders.append((backup_datetime, item_path))
            except ValueError:
                pass  # ignore directories that do not match the backup format

    # Sort backup folders chronologically
    backup_folders.sort(key=lambda x: x[0], reverse=True)
    print(f"Backup folders found in {path}:", backup_folders)

    if steps_back < len(backup_folders):
        # Remove backup folders that are more recent than the specified number of steps
        folders_to_delete = backup_folders[:steps_back]
        for folder in folders_to_delete:
            shutil.rmtree(folder[1])
            backup_folders.remove(folder)

    # Set 'latest' folder to the newest backup folder
    if backup_folders:
        latest_path = os.path.join(path, 'latest')
        if os.path.exists(latest_path):
            shutil.rmtree(latest_path)
        shutil.copytree(backup_folders[0][1], latest_path)
        print(f"Latest folder set to {backup_folders[0][1]}")
    else:
        print(f"No backup folders found in {path}. Latest folder not set.")

def matching_rows(A, B):
    matches = [i for i in range(B.shape[0]) if np.any(np.all(A == B[i], axis=1))]
    if len(matches) == 0:
        return B[matches]
    return np.unique(B[matches], axis=0)
