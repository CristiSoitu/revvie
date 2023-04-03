import numpy as np
import argparse
import configparser
import os
import toml


def ConfigReader(config_file):
    

    # returns argparse object with all the config values
    config = configparser.ConfigParser()
    config.read(config_file)
    config = config['DEFAULT']
    
    parser = argparse.ArgumentParser()
    # for config.positions remove the brackets. and split by comma
    parser.add_argument('--dataset_path', default=config['dataset_path'], type=str)
    parser.add_argument('--positions', default=config['positions'], type=str2list)

    parser.add_argument('--unmatched_color', default=config['unmatched_color'], type=tuple_float_type)
    parser.add_argument('--matched_color', default=config['matched_color'], type=tuple_float_type)
    parser.add_argument('--accepted_edge_color', default=config['accepted_edge_color'], type=tuple_float_type)
    parser.add_argument('--rejected_edge_color', default=config['rejected_edge_color'], type=tuple_float_type)

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

    parser.add_argument('--matcher_name', default='Matcher', type=str)


        
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
    centroids_path = quick_dir(revvie_path, 'centroids')
    stack_path = quick_dir(revvie_path, 'stacks')
    
    positions = args.positions


    # check to see if there is a slices folder
    if not os.path.exists(revvie_path + '/slices'):
        vitro_points_np = np.loadtxt(revvie_path + 'centroids/geneseq_slices_centroids.txt')
        vitro_points_np[:, [1, 2]] = vitro_points_np[:, [2, 1]]

        slices_path = quick_dir(revvie_path, 'slices')

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

            np.save(latest_state_path + 'vivo_matched_indices.npy', np.empty((0, 1)))
            np.savetxt(latest_state_path + 'vivo_matched_indices.txt', np.empty((0, 1)), fmt='%i')

            np.save(latest_state_path + 'vivo_matched_points.npy', np.empty((0, 10)))
            np.savetxt(latest_state_path + 'vivo_matched_points.txt', np.empty((0, 10)), fmt='%.1f')


        vivo_points = np.loadtxt(revvie_path + 'centroids/padded_func_centroids.txt')
        vivo_points[:, [1, 2]] = vivo_points[:, [2, 1]]
        vivo_points = vivo_points[:, :4]
        vivo_ids = vivo_points[:, 3]
        unique_ids = np.unique(vivo_ids)
        colors = np.ones((unique_ids.shape[0], 3)) * args.unmatched_color
        padded_colors = np.ones_like(vivo_points[:, :3])
        for i, id in enumerate(vivo_ids):
            padded_colors[i] = colors[unique_ids == id][0]
        
        edge_colors = padded_colors.copy()
        
        vivo_points = np.hstack((vivo_points, padded_colors))
        vivo_points = np.hstack((vivo_points, edge_colors))

        np.save(stack_path + 'vivo_points.npy', vivo_points)
        
    vitro_points = np.empty((0, 10))
    slices_path = quick_dir(revvie_path, 'slices')
    
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
            print(vivo_indices.dtype)
            print(vivo_matched_points.dtype)
            print('vivo_indices', vivo_indices)
            print('vivo_points[vivo_indices]', vivo_points[vivo_indices])
            vivo_points[vivo_indices] = vivo_matched_points
        else:
            continue


#        try:
#            vivo_points[vivo_indices] = vivo_matched_points

 #       except IndexError:
 #           continue
    return vitro_points, vivo_points



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