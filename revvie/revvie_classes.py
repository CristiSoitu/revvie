import numpy as np
import napari
import tifffile as tif
from revvie.revvie_helpers import *
from revvie.delaunay_triangulation_nonrigid import *
import time
import threading
import subprocess


class CloudPoints:
    def __init__(self, xyz, ids, colors, size, name, opacity, ndim, edge_colors, edge_width, symbol, matched_color, unmatched_color, accepted_edge_color, rejected_edge_color):
        self.xyz = xyz.astype(np.int32)
        self.ids = ids.astype(np.int32)
        self.colors = colors
        self.size = int(size)
        self.name = name
        self.opacity = opacity
        self.ndim = ndim
        self.edge_colors = edge_colors
        self.edge_width = edge_width
        self.symbol = symbol
        self.matched_color = matched_color
        self.unmatched_color = unmatched_color
        self.accepted_edge_color = accepted_edge_color
        self.rejected_edge_color = rejected_edge_color

    def add_points(self, xyz, ids, colors, edge_colors):
        self.xyz = np.vstack((self.xyz, xyz.astype(np.int32)))
        self.ids = np.hstack((self.ids, ids.astype(np.int32)))
        self.colors = np.vstack((self.colors, colors))
        self.edge_colors = np.vstack((self.edge_colors, edge_colors))
        
    def add_points_by_vectors(self, vector):
        # if vector is 1D turn it into a 2D array
        if len(vector.shape) == 1:
            vector = vector.reshape(1, -1)
        self.add_points(vector[:, :3], vector[:, 3], vector[:, 4:7], vector[:, 7:10])

    def get_slice_by_index(self, index):
        return 'Pos' + str(int(self.xyz[index, 0]))
    
    def get_slice_by_z(self, z):
        return 'Pos' + str(int(z))
    
    def get_points_by_z(self, z):
        index = np.where(self.xyz[:, 0] == z)
        #return all info in this order xyz, ids, colors, edge_colors 
        points = np.hstack((self.xyz[index], self.ids[index].reshape(-1, 1), self.colors[index], self.edge_colors[index]), dtype=np.float32)
        return points

    def get_points_by_index(self, index):
        #return all info in this order xyz, ids, colors, edge_colors 
        points = np.hstack((self.xyz[index], self.ids[index].reshape(-1, 1), self.colors[index], self.edge_colors[index]), dtype=np.float32)
        return points
        

    def get_id_by_index(self, index):
        return self.ids[index]
    
    def get_index_by_id(self, id):
        index = np.where(self.ids == id)
        return index

    def get_xyz_by_id(self, id):
        index = np.where(self.ids == id)
        return self.xyz[index]

    def get_color_by_id(self, id):
        index = np.where(self.ids == id)
        return self.colors[index]
    
    def get_size_by_id(self, id):
        index = np.where(self.ids == id)
        return self.size[index]
    
    def get_name_by_id(self, id):
        index = np.where(self.ids == id)
        return self.name[index]
    
    def get_opacity_by_id(self, id):
        index = np.where(self.ids == id)
        return self.opacity[index]
    
    def get_ndim_by_id(self, id):
        index = np.where(self.ids == id)
        return self.ndim[index]
    

    def get_slice_by_id(self, id):
        index = np.where(self.ids == id)
        return 'Pos' + str(int(self.xyz[index, 0]))
    
    def set_xyz_by_id(self, id, xyz):
        index = np.where(self.ids == id)
        self.xyz[index] = xyz

    def set_color_by_id(self, id, color):
        index = np.where(self.ids == id)
        self.colors[index] = color

    
    def set_initial_color_by_id(self, id, color):
        index = np.where(self.ids == id)
        self.colors = color
    
    def set_initial_edge_color_by_id(self, id, color):
        index = np.where(self.ids == id)
        self.edge_colors = color

    def set_edge_color_by_id(self, id, color):
        index = np.where(self.ids == id)
        self.edge_colors[index] = color

    def set_size_by_id(self, id, size):
        index = np.where(self.ids == id)
        self.size[index] = size

    def set_name_by_id(self, id, name):
        index = np.where(self.ids == id)
        self.name[index] = name

    def set_opacity_by_id(self, id, opacity):
        index = np.where(self.ids == id)
        self.opacity[index] = opacity

    def set_ndim_by_id(self, id, ndim):
        index = np.where(self.ids == id)
        self.ndim[index] = ndim

    def set_symbol_by_id(self, id, symbol):
        index = np.where(self.ids == id)
        self.symbols[index] = symbol


class RevViewer:

    def __init__(self, vitro_images_list, vivo_images_list, vitro_cloud_points, vivo_cloud_points, args):
        self.args = args
        self.vitro_images_list = vitro_images_list
        self.vivo_images_list = vivo_images_list

        ## Initialize two viewers
        self.vitro_viewer = napari.Viewer(title='Vitro Viewer')
        self.vivo_viewer = napari.Viewer(title='Vivo Viewer')

        ## Initialize the two cloud points
        self.vitro_cloud_points = vitro_cloud_points
        self.vivo_cloud_points = vivo_cloud_points
        self.anchor_vitro_points = None
        self.anchor_vivo_points = None


        ## Bind all relevant keys and functions to viewers
        self.vivo_viewer.bind_key(self.args.toggle_visibility_key, self.toggle_points_visibility)
        self.vitro_viewer.bind_key(self.args.toggle_visibility_key, self.toggle_points_visibility)

        self.vivo_viewer.bind_key(self.args.save_match_key, self.match)
        self.vitro_viewer.bind_key(self.args.save_match_key, self.match)

        self.vivo_viewer.bind_key(self.args.delete_match_key, self.delete_match)
        self.vitro_viewer.bind_key(self.args.delete_match_key, self.delete_match)

        self.vivo_viewer.bind_key(self.args.save_state_key, self.save_state)
        self.vitro_viewer.bind_key(self.args.save_state_key, self.save_state)

        self.vivo_viewer.bind_key(self.args.run_alignment_key, self.run_generate_predictions)
        self.vitro_viewer.bind_key(self.args.run_alignment_key, self.run_generate_predictions)
           
        self.vivo_viewer.bind_key(self.args.generate_grid_key, self.generate_grid)
        self.vitro_viewer.bind_key(self.args.generate_grid_key, self.generate_grid)
        
        self.vivo_viewer.bind_key(self.args.add_anchor_point_key, self.add_anchor_points)
        self.vitro_viewer.bind_key(self.args.add_anchor_point_key, self.add_anchor_points)

        self.vivo_viewer.bind_key(self.args.validate_prediction_key, self.accept_prediction)
        self.vitro_viewer.bind_key(self.args.validate_prediction_key, self.accept_prediction)

        self.vivo_viewer.bind_key(self.args.reject_prediction_key, self.reject_prediction)
        self.vitro_viewer.bind_key(self.args.reject_prediction_key, self.reject_prediction)

        self.vivo_viewer.bind_key(self.args.maybe_prediction_key, self.dontknow_prediction)
        self.vitro_viewer.bind_key(self.args.maybe_prediction_key, self.dontknow_prediction)

        self.vivo_viewer.bind_key(self.args.display_pair_key, self.take_me_there)
        self.vitro_viewer.bind_key(self.args.display_pair_key, self.take_me_there)






## Some useful functionalities       
    def toggle_points_visibility(self, viewer):
        viewer.layers['points'].visible = not viewer.layers['points'].visible

    def save_state(self, viewer):
        for position in args.positions:
            slice_path = self.args.matcher_path + 'slices/' + position + '/latest/'
            z = get_trailing_number(position)
            points = self.vitro_cloud_points.get_points_by_z(z)
            np.save(slice_path + 'displayed_points.npy', points)
            np.savetxt(slice_path + 'displayed_points.txt', points, fmt='%.1f')
        print('State saved.')

    def take_me_there(self, viewer):
        t = threading.Thread(target=self.take_me_there_function, args=(viewer,))
        t.start()

    def take_me_there_function(self, viewer):
        vitro_id, _, _, _, = self.get_selected_ids() 
        if vitro_id is not None:
            slice = self.vitro_cloud_points.get_slice_by_id(vitro_id)
            slice_path = self.args.matcher_path + 'slices/' + slice + '/latest/'
            matches = np.loadtxt(slice_path + 'matches.txt')
            vivo_id = matches[matches[:,0] == vitro_id, 1]
            if len(vivo_id) == 0:
                predicted = np.loadtxt(slice_path + 'predicted.txt')
                vivo_id = predicted[predicted[:,0] == vitro_id, 1]
        
            xyz_vivo_points = self.vivo_cloud_points.get_xyz_by_id(vivo_id)
            middle_point = xyz_vivo_points[2]
            self.vivo_viewer.camera.center = (middle_point[1], middle_point[2])
            self.vivo_viewer.dims.set_current_step(axis=1, value=middle_point[0])
            self.vivo_viewer.camera.zoom = 1.8

            vitro_point_coords = self.vitro_cloud_points.get_xyz_by_id(vitro_id)[0]
            vivo_coords = (middle_point[1], middle_point[2])

            vitro_size = self.args.vitro_point_size + 5
            vivo_size = self.args.vivo_point_size + 5
            vitro_box = np.array([[vitro_point_coords[1] - vitro_size, vitro_point_coords[2] - vitro_size], [vitro_point_coords[1] + vitro_size, vitro_point_coords[2] - vitro_size], [vitro_point_coords[1] + vitro_size, vitro_point_coords[2] + vitro_size], [vitro_point_coords[1] - vitro_size, vitro_point_coords[2] + vitro_size]])
            vivo_box = np.array([[vivo_coords[0] - vivo_size, vivo_coords[1] - vivo_size], [vivo_coords[0] + vivo_size, vivo_coords[1] - vivo_size], [vivo_coords[0] + vivo_size, vivo_coords[1] + vivo_size], [vivo_coords[0] - vivo_size, vivo_coords[1] + vivo_size]])
                
            self.vitro_viewer.layers['shapes'].data = vitro_box
            self.vivo_viewer.layers['shapes'].data = vivo_box
            time.sleep(3)
            self.vitro_viewer.layers['shapes'].data = []
            self.vivo_viewer.layers['shapes'].data = []
        else:
            print('No vitro id selected')


## Match
    def match(self, viewer):
        vitro_id, vivo_id, vitro_index, vivo_index = self.get_match_ids()
        self.update_match_display(vitro_id, vivo_id, face_color=np.random.rand(1, 3), edge_color=self.args.matched_color)
        self.record_match(vitro_id, vivo_id, vitro_index, vivo_index)
        print('Match recorded for vitro id: ' + str(vitro_id) + ' and vivo id: ' + str(vivo_id))

    def record_match(self, vitro_id, vivo_id, vitro_index, vivo_index):
        #records match in the matches.txt file of every slice
        slice = self.vitro_cloud_points.get_slice_by_index(vitro_index)
        slice_path = self.args.matcher_path + 'slices/' + slice + '/latest/'
        matches = np.loadtxt(slice_path + 'matches.txt')
        try:
            matches = np.vstack((matches, np.hstack((vitro_id, vivo_id))))
        except ValueError:
            matches = np.array((vitro_id, vivo_id))
        np.savetxt(slice_path + 'matches.txt', matches, fmt='%i')

        vivo_matched_indices = np.loadtxt(slice_path + 'vivo_matched_indices.txt')
        vivo_index = self.vivo_cloud_points.get_index_by_id(vivo_id)[0]
        try:
            vivo_matched_indices = np.concatenate((vivo_matched_indices, vivo_index), axis=0) 
        except ValueError:
            vivo_matched_indices = np.array(vivo_matched_indices)
        np.savetxt(slice_path + 'vivo_matched_indices.txt', vivo_matched_indices, fmt='%i')
        
        vivo_matched_points = np.loadtxt(slice_path + 'vivo_matched_points.txt')
        vivo_points = self.vivo_cloud_points.get_points_by_index(vivo_index)
        try:
            vivo_matched_points = np.vstack((vivo_matched_points, vivo_points))
        except ValueError:
            vivo_matched_points = np.array(vivo_points)
    
        np.savetxt(slice_path + 'vivo_matched_points.txt', vivo_matched_points, fmt='%.1f')


## Delete a match
    def delete_match(self, viewer):
        vitro_id, vivo_id, vitro_index, vivo_index = self.get_selected_ids()
        print('Deleting match for vitro id: ' + str(vitro_id))
        vitro_id, vivo_id = self.record_deletion(vitro_id, vivo_id, vitro_index, vivo_index)
        self.update_match_display(vitro_id, vivo_id, face_color=self.args.unmatched_color, edge_color=self.args.unmatched_color)

    def record_deletion(self, vitro_id, vivo_id, vitro_index, vivo_index):
        slice = self.vitro_cloud_points.get_slice_by_index(vitro_index)
        slice_path = self.args.matcher_path + 'slices/' + slice + '/latest/'
        
        matches = np.loadtxt(slice_path + 'matches.txt')
        match_to_delete = np.where((matches[:, 0] == vitro_id))
        assert len(match_to_delete[0]) > 0, 'No match found for this vitro id. You may have tried to delete an unmatched point, ' + self.args.matcher_name + '.'
        vivo_id = matches[match_to_delete, 1]
        assert len(vivo_id) == 1, 'No match found or there is more than one match for this vitro id.'

        matches = np.delete(matches, match_to_delete, axis=0)
        np.savetxt(slice_path + 'matches.txt', matches, fmt='%i')
        vivo_matched_indices = np.loadtxt(slice_path + 'vivo_matched_indices.txt')
        vivo_index_to_delete = self.vivo_cloud_points.get_index_by_id(vivo_id[0])[0]
        #vivo_index to delete may be a list of indices
        for index in vivo_index_to_delete:
            vivo_matched_indices = vivo_matched_indices[vivo_matched_indices != index.astype(np.float32)]
        np.savetxt(slice_path + 'vivo_matched_indices.txt', vivo_matched_indices, fmt='%i')

        vivo_matched_points = np.loadtxt(slice_path + 'vivo_matched_points.txt')
        vivo_matched_points = vivo_matched_points[vivo_matched_points[:, 3] != vivo_id[0].astype(np.float32)]
        np.savetxt(slice_path + 'vivo_matched_points.txt', vivo_matched_points, fmt='%.1f')

        return vitro_id, vivo_id[0]


## Generate predictions
    def run_generate_predictions(self, viewer):
        t = threading.Thread(target=self.generate_predictions, args=(viewer,))
        t.start()

    def generate_predictions(self, viewer):
        vitro_id, _, _, _ = self.get_selected_ids()
        slice = self.vitro_cloud_points.get_slice_by_id(vitro_id)
        if slice == None:
            slices = self.args.positions
        else:
            slices = [slice]
        print('Running automatch for slices: ' + str(slices))
        predicted_matches = self.run_automatch(self.args, slices)
        for slice in slices:
            self.display_predicted_matches(slice)
        
        #for slice in slices:
        #    print('Running automatch for slice: ' + slice)
        #    #predicted_matches = run_automatch.py
        #    predicted_matches = self.run_automatch(self.args, slice)
        #    print('generated ' + str(len(predicted_matches)) + ' predictions for slice: ' + slice)
        #    self.display_predicted_matches(slice)

    def run_automatch(self, args, slices):
        if len(slices) == 0:
            slices = args.positions
        
        stack_path = args.matcher_path + 'stack/'
        vivo_func_points = np.load(stack_path + 'vivo_points.npy').astype(np.float32)
        func_ids = vivo_func_points[:, 3]

        vivo_anchor_points = np.empty((0, 4))
        for slice in args.positions:
            slice_path = args.matcher_path + 'slices/' + slice + '/latest/'
            slice_vivo_anchor_points = np.loadtxt(slice_path + 'vivo_anchors.txt')
            slice_vivo_anchor_points = slice_vivo_anchor_points[:, :4]
            #remove points with id 0, i.e. keep slice_vivo_anchor_points[3] != 0
            slice_vivo_anchor_points = slice_vivo_anchor_points[slice_vivo_anchor_points[:, 3] != 0]
            
            vivo_anchor_points = np.vstack((vivo_anchor_points, slice_vivo_anchor_points))

        vivo_anchor_points[:, [1, 2]] = vivo_anchor_points[:, [2, 1]]
        print('Adding to automatch run ', len(vivo_anchor_points), ' anchors')
        for slice in slices:
            vivo_points = np.load(stack_path + 'struct_vivo_points.npy').astype(np.float32)
            #select only the first occurence of ids in column 3
            vivo_points = vivo_points[np.unique(vivo_points[:, 3], return_index=True)[1]]
            vivo_points[:, [0, 2]] = vivo_points[:, [2, 0]]
            
            slice_path = args.matcher_path + 'slices/' + slice + '/latest/'
            create_backup(slice_path + '../')
            #wait 3 seconds to make sure the backup is created
            time.sleep(3)
            matches = np.loadtxt(slice_path + 'matches.txt')
            vitro_anchor_points = np.loadtxt(slice_path + 'vitro_anchors.txt')

            if len(matches) > 3 or len(vivo_anchor_points) > 3:
                vitro_points = np.loadtxt(slice_path + 'true_points.txt')
                vitro_points = vitro_points[:, [0, 2, 1, 3]]
                vitro_anchor_points = vitro_anchor_points[:, :4]
                #vitro_anchor_points = vitro_anchor_points[:, [0, 2, 1, 3]]
                # remove vitro_anchor_points with index 0
                vitro_anchor_points = vitro_anchor_points[vitro_anchor_points[:, 3] != 0]
                vitro_points = np.vstack((vitro_points, vitro_anchor_points))

                vivo_anchor_points = np.loadtxt(slice_path + 'vivo_anchors.txt')
                vivo_anchor_points = vivo_anchor_points[:, :4]
                #remove points with id 0, i.e. keep vivo_anchor_points[3] != 0
                vivo_anchor_points = vivo_anchor_points[vivo_anchor_points[:, 3] != 0]
                #vivo_anchor_points[:, [1, 2]] = vivo_anchor_points[:, [2, 1]]


                vivo_points = np.vstack((vivo_points, vivo_anchor_points))

                if len(matches) == 0:
                    matches = np.empty((0, 2))

                anchors_ids = np.vstack((vitro_anchor_points[:, 3], vivo_anchor_points[:, 3])).T


                matches = np.vstack((matches, anchors_ids))
                output_path = quick_dir(slice_path, 'automatch')


                # Piecewise affine


                pc_return = piecewise_affine_cent(vitro_points, vivo_points, matches)

                for key in pc_return.keys():
                    #check to see if the key is a dataframe
                    if isinstance(pc_return[key], pd.DataFrame):
                        pc_return[key].to_csv(output_path + 'pc_return_' + key + '.csv')


                # Nonrigid
                near_invivo_cents = pc_return['invivo_units'][np.where(np.isin(pc_return['invivo_units'][:, 3], 
                                                                pc_return['matching_probs']['StackUnit']))[0], :]
                aff_invitro_coords = pc_return['invitro_units_trans'].loc[:, ['3D_X', '3D_Y', '3D_Z', 'ID']]

                test_nonrigid = nonrigid_demon(aff_invitro_cent = aff_invitro_coords,
                                        invivo_cent = pc_return['invivo_near_units'],
                                        match_df = pc_return['manual_matches'],
                                        invivo_l = pc_return['invivo_lookup'],
                                        threshes = pc_return['matchprob_filts'],
                                        pot_match_tab = pc_return['matching_probs'])

                #save all fields of test_nonrigid to file in output_path 
                for key in test_nonrigid.keys():
                    if isinstance(test_nonrigid[key], pd.DataFrame):
                        test_nonrigid[key].to_csv(output_path + 'test_nonrigid_' + key + '.csv', index=False)

                # Automatic mathces
                automatic_matches = automatch(test_nonrigid, vivo_points, vitro_points)

                automatic_matches.to_csv(output_path + 'automatic_matches.csv', index=False)

                #transformed_vitro_points = test_nonrigid['invitro_units']
                #extract columns 'SliceUnit' and 'StackUnit' from automatic_matches
                predicted_matches_id = automatic_matches.loc[:, ['SliceUnit', 'StackUnit']]
                #turn this into a numpy array
                predicted_matches = predicted_matches_id.to_numpy().astype(np.int32)
                
                
                #save only those where StackUnit is in func_ids
                print('Predicted ' + str(len(predicted_matches)) + ' struct matches.')
                predicted_matches = predicted_matches[np.where(np.isin(predicted_matches[:, 1], func_ids))[0], :]
                print('Predicted ' + str(len(predicted_matches)) + ' func matches.')
                
                #remove rows where StackUnit or SliceUnit < 0
                predicted_matches = predicted_matches[np.where(predicted_matches[:, 0] >= 0)[0], :]
                predicted_matches = predicted_matches[np.where(predicted_matches[:, 1] >= 0)[0], :]
                print('Predicted ' + str(len(predicted_matches)) + ' matches after removing anchors.')
                rejected = np.loadtxt(slice_path + 'rejected.txt')
                #save only those rows not in rejected
                overlap = matching_rows(predicted_matches, rejected)
                overlap_indices = []
                #find overlap indices in predicted_matches
                for i in range(len(overlap)):
                    for j in range(len(predicted_matches)):
                        if overlap[i, 0] == predicted_matches[j, 0] and overlap[i, 1] == predicted_matches[j, 1]:
                            overlap_indices.append(j)
                predicted_matches = np.delete(predicted_matches, overlap_indices, axis=0)
                rejected = np.empty((0, 2), dtype=np.int32)
                np.savetxt(slice_path + 'rejected.txt', rejected, fmt='%i')
                print('Predicted ' + str(len(predicted_matches)) + ' matches after removing rejected.')

                matches = np.loadtxt(slice_path + 'matches.txt')
                overlap = matching_rows(predicted_matches, matches)
                overlap_indices = []
                #find overlap indices in predicted_matches
                for i in range(len(overlap)):
                    for j in range(len(predicted_matches)):
                        if overlap[i, 0] == predicted_matches[j, 0] and overlap[i, 1] == predicted_matches[j, 1]:
                            overlap_indices.append(j)
                predicted_matches = np.delete(predicted_matches, overlap_indices, axis=0)
                print('Predicted ' + str(len(predicted_matches)) + ' matches after removing matches.')

                accepted = np.empty((0, 2), dtype=np.int32)
                np.savetxt(slice_path + 'accepted.txt', rejected, fmt='%i')

                np.savetxt(slice_path + 'predicted.txt', predicted_matches, fmt='%i')


## Generate grid to add anchor points
    def generate_grid(self, viewer):
        t = threading.Thread(target=self.generate_grid_function, args=(viewer,))
        t.start()

    def generate_grid_function(self, viewer):
        # record corners of the rectangle in the shapes layer
        box = self.vitro_viewer.layers['shapes'].data
        top_left, top_right, bottom_right, bottom_left = box[0][0], box[0][1], box[0][2], box[0][3]
        #generate a x_grid_points by y_grid_points grid of points in the rectangle
        grid_points = np.empty((self.args.x_grid_points * self.args.y_grid_points, 2))
        x_grid = np.linspace(top_left[1], top_right[1], self.args.x_grid_points)
        y_grid = np.linspace(top_left[0], bottom_left[0], self.args.y_grid_points)

        for i in range(self.args.x_grid_points):
            for j in range(self.args.y_grid_points):
                grid_points[i * self.args.y_grid_points + j, :] = [y_grid[j], x_grid[i]]
        half_width = 25
        rectangles = []
        for center in grid_points:
            # Compute the coordinates of the rectangle vertices
            x, y = center
            vertices = np.array([[x-half_width, y-half_width], [x+half_width, y-half_width], [x+half_width, y+half_width], [x-half_width, y+half_width]])
            rectangles.append(vertices)

        # plot rectangles at each grid point in the shapes layer
        self.vitro_viewer.layers['shapes'].data = []
        self.vitro_viewer.layers['shapes'].add_rectangles(rectangles, edge_color='yellow', face_color='transparent', edge_width=4)
        

## Add anchor points
    def add_anchor_points(self, viewer):
        t = threading.Thread(target=self.add_anchor_points_function, args=(viewer,))
        t.start()

    def add_anchor_points_function(self, viewer):
        
        selected_vitro_point = self.vitro_viewer.layers['anchors'].selected_data
        selected_vivo_point = self.vivo_viewer.layers['anchors'].selected_data
        vitro_point = self.vitro_viewer.layers['anchors'].data[-1]
        vivo_point = self.vivo_viewer.layers['anchors'].data[-1]

       #get coordinates of selected points
        [z, y, x] = self.vitro_viewer.layers['anchors'].data[-1]
        slice = self.vitro_cloud_points.get_slice_by_z(z)
        slice_path = self.args.matcher_path + 'slices/' + slice + '/latest/'

        vitro_anchors = np.loadtxt(slice_path + 'vitro_anchors.txt')
        vivo_anchors = np.loadtxt(slice_path + 'vivo_anchors.txt')

    
        anchor_id = len(vitro_anchors) + 1
        face_color = np.random.rand(1, 3)
        edge_color = self.args.anchor_edge_color 
        
        [z, y, x] = self.vitro_viewer.layers['anchors'].data[-1]
        assert vitro_anchors[-1][2] != x, 'You need to select a new vitro point.' 

        vitro_point = np.array([z, y, x, -anchor_id, face_color[0][0], face_color[0][1], face_color[0][2], 
                                edge_color[0], edge_color[1], edge_color[2]])
        try:
            vitro_anchors = np.vstack((vitro_anchors, vitro_point))
        except ValueError:
            vitro_anchors = vitro_point
        # format 2 decimals
        np.savetxt(slice_path + 'vitro_anchors.txt', vitro_anchors, fmt='%.2f')

        [z, y, x] = self.vivo_viewer.layers['anchors'].data[-1]
        assert vivo_anchors[-1][2] != x, 'You need to select a new vivo point.' 
        vivo_point = np.array([z, y, x, -anchor_id, face_color[0][0], face_color[0][1], face_color[0][2], 
                                edge_color[0], edge_color[1], edge_color[2]])  
        try:
            vivo_anchors = np.vstack((vivo_anchors, vivo_point))
        except ValueError:
            vivo_anchors = vivo_point
        
        np.savetxt(slice_path + 'vivo_anchors.txt', vivo_anchors, fmt='%.2f')

        self.anchor_vitro_points.add_points_by_vectors(vitro_point)
        self.anchor_vivo_points.add_points_by_vectors(vivo_point)

        self.vivo_viewer.layers['anchors'].data = self.anchor_vivo_points.xyz
        self.vitro_viewer.layers['anchors'].data = self.anchor_vitro_points.xyz

        self.update_anchor_display(vitro_point, vivo_point, face_color, self.args.anchor_edge_color)
        self.vitro_viewer.layers['anchors'].selected_data = []
        self.vivo_viewer.layers['anchors'].selected_data = []
        print('Added anchor points to slice: ' + slice + ' with id: ' + str(anchor_id), ' and coordinates: ' + str(vitro_point[0:3]) + ' and ' + str(vivo_point[0:3]))

    def delete_anchor_point(self, viewer):
        vitro_index = viewer.layers['anchors'].selected_data
        vivo_index = viewer.layers['anchors'].selected_data
        
        assert vitro_index is not None and vivo_index is not None, 'You must select at least one vitro or vivo point.'

        if vitro_index is not None:
            vitro_index = list(self.get_selected_index(self.vitro_viewer))
            vitro_id = self.anchor_vitro_points.get_id_by_index(vitro_index)



        else:
            return 0




## Accept prediction
    def accept_prediction(self, viewer):
        vitro_id, _, vitro_index, _ = self.get_selected_ids() 
        vivo_id = self.record_accepted_match(vitro_id, vitro_index)
        self.update_match_display(vitro_id, vivo_id, face_color=np.random.rand(1, 3), edge_color=self.args.matched_color)
        print('Prediction accepted for vitro id: ' + str(vitro_id) + ' and vivo id: ' + str(vivo_id))

    def record_accepted_match(self, vitro_id, vitro_index):
        #records match in the matches.txt file of every slice
        slice = self.vitro_cloud_points.get_slice_by_index(vitro_index)
        slice_path = self.args.matcher_path + 'slices/' + slice + '/latest/'
        predicted = np.loadtxt(slice_path + 'predicted.txt')
        vivo_id = predicted[predicted[:, 0] == vitro_id, 1]
        vivo_index = self.vivo_cloud_points.get_index_by_id(vivo_id)[0]
        predicted = np.delete(predicted, np.where(predicted[:, 0] == vitro_id), axis=0)
        np.savetxt(slice_path + 'predicted.txt', predicted, fmt='%i')


        print('vitro id', vitro_id)
        print('vivo id', vivo_id)
        matches = np.loadtxt(slice_path + 'matches.txt')
        try:
            matches = np.vstack((matches, np.hstack((vitro_id, vivo_id))))
        except ValueError:
            matches = np.array((vitro_id, vivo_id))
        np.savetxt(slice_path + 'matches.txt', matches, fmt='%i')

        accepted_match = np.loadtxt(slice_path + 'accepted.txt')
        try:
            accepted_match = np.vstack((accepted_match, np.hstack((vitro_id, vivo_id))))
        except ValueError:
            accepted_match = np.array((vitro_id, vivo_id))
        np.savetxt(slice_path + 'accepted.txt', accepted_match, fmt='%i')

        vivo_matched_indices = np.loadtxt(slice_path + 'vivo_matched_indices.txt')
        vivo_index = self.vivo_cloud_points.get_index_by_id(vivo_id)[0]
        try:
            vivo_matched_indices = np.concatenate((vivo_matched_indices, vivo_index), axis=0) 
        except ValueError:
            vivo_matched_indices = np.array(vivo_matched_indices)
        np.savetxt(slice_path + 'vivo_matched_indices.txt', vivo_matched_indices, fmt='%i')
        
        vivo_matched_points = np.loadtxt(slice_path + 'vivo_matched_points.txt')
        vivo_points = self.vivo_cloud_points.get_points_by_index(vivo_index)
        try:
            vivo_matched_points = np.vstack((vivo_matched_points, vivo_points))
        except ValueError:
            vivo_matched_points = np.array(vivo_points)
    
        np.savetxt(slice_path + 'vivo_matched_points.txt', vivo_matched_points, fmt='%.1f')
        return vivo_id


## Reject prediction
    def reject_prediction(self,viewer):
        vitro_id, _, vitro_index, _ = self.get_selected_ids()
        vivo_id = self.record_rejected_match(vitro_id, vitro_index) 
        self.update_match_display(vitro_id, vivo_id, face_color=self.args.unmatched_color, edge_color=self.args.unmatched_color)
        print('Prediction rejected for vitro id: ' + str(vitro_id) + ' and vivo id: ' + str(vivo_id))

    def record_rejected_match(self, vitro_id, vitro_index):
        #records match in the matches.txt file of every slice
        slice = self.vitro_cloud_points.get_slice_by_index(vitro_index)
        slice_path = self.args.matcher_path + 'slices/' + slice + '/latest/'
        predicted = np.loadtxt(slice_path + 'predicted.txt')
        vivo_id = predicted[predicted[:, 0] == vitro_id, 1]
        predicted = np.delete(predicted, np.where(predicted[:, 0] == vitro_id), axis=0)
        np.savetxt(slice_path + 'predicted.txt', predicted, fmt='%i')


        print('vitro id', vitro_id)
        print('vivo id', vivo_id)
        rejected_match = np.loadtxt(slice_path + 'rejected.txt')
        try:
            rejected_match = np.vstack((rejected_match, np.hstack((vitro_id, vivo_id))))
        except ValueError:
            rejected_match = np.array((vitro_id, vivo_id))
        np.savetxt(slice_path + 'rejected.txt', rejected_match, fmt='%i')
        return vivo_id


## Unsure about prediction
    def dontknow_prediction(self,viewer):
        vitro_id, _, vitro_index, _ = self.get_selected_ids()
        vivo_id = self.record_dontknow_match(vitro_id, vitro_index) 
        self.update_match_display(vitro_id, vivo_id, face_color=self.args.unmatched_color, edge_color=self.args.unmatched_color)
        print('Unsure about prediction for vitro id: ' + str(vitro_id) + ' and vivo id: ' + str(vivo_id))

    def record_dontknow_match(self, vitro_id, vitro_index):
        #records match in the matches.txt file of every slice
        slice = self.vitro_cloud_points.get_slice_by_index(vitro_index)
        slice_path = self.args.matcher_path + 'slices/' + slice + '/latest/'
        predicted = np.loadtxt(slice_path + 'predicted.txt')
        vivo_id = predicted[predicted[:, 0] == vitro_id, 1]
        predicted = np.delete(predicted, np.where(predicted[:, 0] == vitro_id), axis=0)
        np.savetxt(slice_path + 'predicted.txt', predicted, fmt='%i')

        #print('vitro id', vitro_id)
        #print('vivo id', vivo_id)
        dontknow_matches = np.loadtxt(slice_path + 'dontknow.txt')
        try:
            dontknow_matches = np.vstack((dontknow_matches, np.hstack((vitro_id, vivo_id))))
        except ValueError:
            dontknow_matches = np.array((vitro_id, vivo_id))
        np.savetxt(slice_path + 'dontknow.txt', dontknow_matches, fmt='%i')
        return vivo_id


## Prep to run napari
    def add_cloud(self, viewer, cloud):
        viewer.add_points(cloud.xyz, face_color='red', size=cloud.size, name=cloud.name, opacity=cloud.opacity, ndim=cloud.ndim, symbol=cloud.symbol, edge_color='red', edge_width=cloud.edge_width)

    def load_anchor_points(self):
        
        vitro_anchors = np.empty((0, 10))
        vivo_anchors = np.empty((0, 10))
        for slice in self.args.positions:
            slice_path = self.args.matcher_path + 'slices/' + slice + '/latest/'
            slice_vitro_anchors = np.loadtxt(slice_path + 'vitro_anchors.txt')
            slice_vivo_anchors = np.loadtxt(slice_path + 'vivo_anchors.txt')
            if len(slice_vitro_anchors) > 0:
                try:
                    vitro_anchors = np.vstack((vitro_anchors, slice_vitro_anchors))
                    vivo_anchors = np.vstack((vivo_anchors, slice_vivo_anchors))
                except ValueError:
                    vitro_anchors = np.array(slice_vitro_anchors)
                    vivo_anchors = np.array(slice_vivo_anchors)
        
        vitro_anchor_cloud = CloudPoints(vitro_anchors[:, 0:3], ids=vitro_anchors[:, 3], colors=vitro_anchors[:, 4:7], edge_colors=vitro_anchors[:, 7:10], name='anchors',
                                        size=self.args.vitro_anchor_point_size, opacity=self.args.opacity, edge_width=self.args.vitro_edge_size, symbol='disc', ndim=3, 
                                        matched_color=self.args.matched_color, unmatched_color=self.args.unmatched_color, accepted_edge_color=self.args.accepted_edge_color, rejected_edge_color=self.args.rejected_edge_color)
        vivo_anchor_cloud = CloudPoints(vivo_anchors[:, 0:3], ids=vivo_anchors[:, 3], colors=vivo_anchors[:, 4:7], edge_colors=vivo_anchors[:, 7:10], name='anchors',
                                            size=self.args.vivo_anchor_point_size, opacity=self.args.opacity, edge_width=self.args.vivo_edge_size, symbol='disc', ndim=3, 
                                            matched_color=self.args.matched_color, unmatched_color=self.args.unmatched_color, accepted_edge_color=self.args.accepted_edge_color, rejected_edge_color=self.args.rejected_edge_color)

        return vitro_anchor_cloud, vivo_anchor_cloud

    def import_matches(self):
        total_matches = 0
        for slice in self.args.positions:
            slice_path = self.args.matcher_path + 'slices/' + slice + '/latest/'
            matches = np.loadtxt(slice_path + 'matches.txt')
            vivo_matched_indices = np.loadtxt(slice_path + 'vivo_matched_indices.txt')
            vivo_matched_points = np.loadtxt(slice_path + 'vivo_matched_points.txt')
            #sometimes matches is just a (2, ) vector. if so, turn it into a (1,2). Also, sometimes matches is empty. if so, skip it.

            if matches.shape == (2,):
                matches = np.array([matches])


            for match in matches:
                vitro_id = match[0]
                vivo_id = match[1]
                self.update_match_display(vitro_id, vivo_id, face_color=np.random.rand(1, 3), edge_color=self.args.matched_color)

                vivo_index = self.vivo_cloud_points.get_index_by_id(vivo_id)[0]
                try:
                    vivo_matched_indices = np.concatenate((vivo_matched_indices, vivo_index), axis=0) 
                except ValueError:
                    vivo_matched_indices = np.array(vivo_matched_indices)
                
                vivo_points = self.vivo_cloud_points.get_points_by_index(vivo_index)
                try:
                    vivo_matched_points = np.vstack((vivo_matched_points, vivo_points))
                except ValueError:
                    vivo_matched_points = np.array(vivo_points)
            np.savetxt(slice_path + 'vivo_matched_indices.txt', vivo_matched_indices, fmt='%i')
            np.savetxt(slice_path + 'vivo_matched_points.txt', vivo_matched_points, fmt='%.1f')
            if len(matches) > 0:
                print(len(matches), ' matches imported for slice: ' + slice)
            total_matches += len(matches)
        print(total_matches, ' matches imported in total')

    def import_predictions(self):
        for slice in self.args.positions:
            self.display_predicted_matches(slice)

    def run(self):
    
        vitro_images = [tif.imread(image_path) for image_path in self.vitro_images_list]
        vitro_stack = np.stack(vitro_images, axis=0)

        vivo_images = [tif.imread(image_path) for image_path in self.vivo_images_list]
        vivo_stack = np.stack(vivo_images, axis=0)
        self.vitro_viewer.add_image(vitro_stack, name='images')
        self.vivo_viewer.add_image(vivo_stack, name='images')   

        self.vitro_viewer.add_points(self.vitro_cloud_points.xyz, face_color=self.vitro_cloud_points.colors, size=self.vitro_cloud_points.size, name=self.vitro_cloud_points.name, opacity=self.vitro_cloud_points.opacity, ndim=self.vitro_cloud_points.ndim, symbol=self.vitro_cloud_points.symbol, edge_color=self.vitro_cloud_points.edge_colors, edge_width=self.vitro_cloud_points.edge_width)
        self.vivo_viewer.add_points(self.vivo_cloud_points.xyz, face_color=self.vivo_cloud_points.colors, size=self.vivo_cloud_points.size, name=self.vivo_cloud_points.name, opacity=self.vivo_cloud_points.opacity, ndim=self.vivo_cloud_points.ndim, symbol=self.vivo_cloud_points.symbol, edge_color=self.vivo_cloud_points.edge_colors, edge_width=self.vivo_cloud_points.edge_width)

        self.anchor_vitro_points, self.anchor_vivo_points = self.load_anchor_points()
        
        self.vitro_viewer.add_points(self.anchor_vitro_points.xyz, face_color=self.anchor_vitro_points.colors, size=self.anchor_vitro_points.size, name='anchors', ndim=3, symbol='disc', edge_color=self.anchor_vitro_points.edge_colors, edge_width=self.args.vitro_edge_size)
        self.vivo_viewer.add_points(self.anchor_vivo_points.xyz, face_color=self.anchor_vivo_points.colors, size=self.anchor_vivo_points.size, name='anchors', ndim=3, symbol='disc', edge_color=self.anchor_vivo_points.edge_colors, edge_width=self.args.vivo_edge_size)

        #add a shapes layer to the vivo and vitro viewers. circles yellow. 
        self.vitro_viewer.add_shapes(name='shapes', shape_type='ellipse', edge_color='yellow', edge_width=2)
        self.vivo_viewer.add_shapes(name='shapes', shape_type='ellipse', edge_color='yellow', edge_width=2)

        self.vitro_viewer.layers['shapes'].add(data=[], shape_type='ellipse', edge_color='yellow', face_color='transparent', edge_width=2)
        self.vivo_viewer.layers['shapes'].add(data=[], shape_type='ellipse', edge_color='yellow', face_color='transparent', edge_width=2)


        self.import_matches()
        self.import_predictions()
        napari.run()


## A few getters 
    def get_selected_index(self, viewer):
        ## returns the selected points in the viewer
        return viewer.layers['points'].selected_data

    def get_match_ids(self):
        ## returns the indices of the matches, as well as the points ids
        vitro_index = list(self.get_selected_index(self.vitro_viewer))
        vitro_id = self.vitro_cloud_points.get_id_by_index(vitro_index)
        vivo_index = list(self.get_selected_index(self.vivo_viewer))
        vivo_id = self.vivo_cloud_points.get_id_by_index(vivo_index)
        assert len(vitro_index) == 1 and len(vivo_index) == 1, 'Make sure you select one point in each viewer, ' + self.args.matcher_name + '.'
        self.clear_selection()
        return vitro_id, vivo_id, vitro_index, vivo_index

    def get_selected_ids(self):
        vitro_index = list(self.get_selected_index(self.vitro_viewer))
        vivo_index = list(self.get_selected_index(self.vivo_viewer))

        assert len(vitro_index) < 2 and len(vivo_index) < 2, 'You cannot select more than one point to delete in vitro' + self.args.matcher_name + '.'

        vitro_id = self.vitro_cloud_points.get_id_by_index(vitro_index)
        vivo_id = self.vivo_cloud_points.get_id_by_index(vivo_index)
        self.clear_selection()
        if len(vitro_index) > 0:
            return vitro_id, vivo_id, vitro_index, vivo_index
        else:
            print('You must select at least one point in vitro' + self.args.matcher_name + '.')
            return None, None, None, None


## Function to clear selections
    def clear_selection(self):
        self.vitro_viewer.layers['points'].selected_data = []
        self.vivo_viewer.layers['points'].selected_data = []

 
## Functions to update display of matches
    def update_match_display(self, vitro_id, vivo_id, face_color, edge_color):

        self.vivo_cloud_points.set_color_by_id(vivo_id, face_color)
        self.vitro_cloud_points.set_color_by_id(vitro_id, face_color)

        self.vivo_cloud_points.set_edge_color_by_id(vivo_id, edge_color)
        self.vitro_cloud_points.set_edge_color_by_id(vitro_id, edge_color)

        self.vivo_viewer.layers[self.vivo_cloud_points.name].face_color = self.vivo_cloud_points.colors
        self.vitro_viewer.layers[self.vitro_cloud_points.name].face_color = self.vitro_cloud_points.colors

        self.vivo_viewer.layers[self.vivo_cloud_points.name].edge_color = self.vivo_cloud_points.edge_colors
        self.vitro_viewer.layers[self.vitro_cloud_points.name].edge_color = self.vitro_cloud_points.edge_colors

    def display_predicted_matches(self, slice):
        slice_path = self.args.matcher_path + 'slices/' + slice + '/latest/'
        predicted_matches = np.loadtxt(slice_path + 'predicted.txt')
        # check to see if predicted matches is not empty
        if len(predicted_matches) > 0:  
            matches = np.loadtxt(slice_path + 'matches.txt')
            for match in predicted_matches:
                vitro_id = match[0]
                vivo_id = match[1]
                random_color = np.random.rand(1, 3)
                self.update_match_display(vitro_id, vivo_id, face_color=random_color, edge_color=self.args.predicted_edge_color)
            print('Importing ' + str(len(predicted_matches)) + ' predicted matches for slice: ' + slice)
        #save only the predicted matches not in matches
        #    predicted_matches = predicted_matches[~np.isin(predicted_matches, matches).all(axis=1)]
        #    np.savetxt(slice_path + 'predicted.txt', predicted_matches, fmt='%i')

    def update_anchor_display(self, vitro_id, vivo_id, face_color, edge_color):

        self.anchor_vivo_points.set_color_by_id(vivo_id, face_color)
        self.anchor_vitro_points.set_color_by_id(vitro_id, face_color)

        self.anchor_vivo_points.set_edge_color_by_id(vivo_id, edge_color)
        self.anchor_vitro_points.set_edge_color_by_id(vitro_id, edge_color)

        self.vivo_viewer.layers['anchors'].face_color = self.anchor_vivo_points.colors
        self.vitro_viewer.layers['anchors'].face_color = self.anchor_vitro_points.colors

        self.vivo_viewer.layers['anchors'].edge_color = 'white'
        self.vitro_viewer.layers['anchors'].edge_color = 'white'




'''
        @self.vivo_viewer.bind_key(self.args.run_alignment_all_slices_key)
        @self.vitro_viewer.bind_key(self.args.run_alignment_all_slices_key)
        def run_generate_predictions_all_slices(viewer):
            t = threading.Thread(target=generate_predictions_all_slices, args=(viewer,))
            t.start()

        def generate_predictions_all_slices(viewer):
            for slice in args.positions:
                print('Running automatch for slice: ' + slice)
                predicted_matches = self.run_automatch(slice)
                print('generated ' + str(len(predicted_matches)) + ' predictions for slice: ' + slice)
                self.display_predicted_matches(slice)



            @self.vivo_viewer.bind_key('p')
        @self.vitro_viewer.bind_key('p')
        def import_predictions_all_slices(viewer):
            t = threading.Thread(target=import_predictions, args=(viewer, False,))
            t.start()

        def import_predictions(viewer, run_bool=True):
            #vitro_id, _, _, _ = self.get_selected_ids()
            #slice = self.vitro_cloud_points.get_slice_by_id(vitro_id)
            for slice in args.positions:
                print('Importing predictions for slice: ' + slice)
                self.display_predicted_matches(slice)


        @self.vivo_viewer.bind_key(self.args.import_predictions_key)
        @self.vitro_viewer.bind_key(self.args.import_predictions_key)
        def import_predictions_one_slice(viewer):
            t = threading.Thread(target=import_predictions_slice, args=(viewer, False,))
            t.start()

        def import_predictions_slice(viewer, run_bool=True):
            vitro_id, _, _, _ = self.get_selected_ids()
            slice = self.vitro_cloud_points.get_slice_by_id(vitro_id)
            print('Importing predictions for slice: ' + slice)
            self.display_predicted_matches(slice)



    def run_automatch1(slice, args):
        
        

        stack_path = args.matcher_path + 'stack/'
        vivo_points = np.load(stack_path + 'vivo_points.npy').astype(np.float32)
        #select only the first occurence of ids in column 3
        vivo_points = vivo_points[np.unique(vivo_points[:, 3], return_index=True)[1]]
        vivo_points[:, [0, 2]] = vivo_points[:, [2, 0]]


        slice_path = args.matcher_path + 'slices/' + slice + '/latest/'
        create_backup(slice_path + '../')
        #wait 3 seconds to make sure the backup is created
        time.sleep(3)
        matches = np.loadtxt(slice_path + 'matches.txt')
        vitro_points = np.loadtxt(slice_path + 'true_points.txt')

        vitro_points = vitro_points[:, [0, 2, 1, 3]]

        output_path = args.dataset_path + 'automatch/'


        # Piecewise affine


        pc_return = piecewise_affine_cent(vitro_points, vivo_points, matches)

        for key in pc_return.keys():
            #check to see if the key is a dataframe
            if isinstance(pc_return[key], pd.DataFrame):
                pc_return[key].to_csv(output_path + 'pc_return_' + key + '.csv')


        # Nonrigid
        near_invivo_cents = pc_return['invivo_units'].iloc[np.where(np.isin(pc_return['invivo_units']['ID'], 
                                                            pc_return['matching_probs']['StackUnit']))[0], :][['X', 'Y', 'Z', 'ID']]
        aff_invitro_coords = pc_return['invitro_units_trans'].loc[:, ['3D_X', '3D_Y', '3D_Z', 'ID']]
        test_nonrigid = nonrigid_demon(aff_invitro_cent = aff_invitro_coords,
                                        invivo_cent = near_invivo_cents,
                                        match_df = pc_return['manual_matches'],
                                        threshes = pc_return['matchprob_filts'])

        #save all fields of test_nonrigid to file in output_path 
        for key in test_nonrigid.keys():
            if isinstance(test_nonrigid[key], pd.DataFrame):
                test_nonrigid[key].to_csv(output_path + 'test_nonrigid_' + key + '.csv', index=False)

        # Automatic mathces
        automatic_matches = automatch(test_nonrigid, vivo_points, vitro_points)

        automatic_matches.to_csv(output_path + 'automatic_matches.csv', index=False)

        #transformed_vitro_points = test_nonrigid['invitro_units']
        #extract columns 'SliceUnit' and 'StackUnit' from automatic_matches
        predicted_matches_id = automatic_matches.loc[:, ['SliceUnit', 'StackUnit']]
        #turn this into a numpy array
        predicted_matches = predicted_matches_id.to_numpy().astype(np.int32)
        np.savetxt(slice_path + 'predicted.txt', predicted_matches, fmt='%i')
        return predicted_matches_id.astype(np.int32)

        

        
    def run_automatch(self, slice):


        stack_path = self.args.matcher_path + 'stack/'

        vivo_func_points = np.load(stack_path + 'vivo_points.npy').astype(np.float32)
        func_ids = vivo_func_points[:, 3]

        vivo_points = np.load(stack_path + 'struct_vivo_points.npy').astype(np.float32)
        #select only the first occurence of ids in column 3
        vivo_points = vivo_points[np.unique(vivo_points[:, 3], return_index=True)[1]]
        vivo_points[:, [0, 2]] = vivo_points[:, [2, 0]]


        slice_path = self.args.matcher_path + 'slices/' + slice + '/latest/'
        create_backup(slice_path + '../')
        #wait 3 seconds to make sure the backup is created
        time.sleep(3)
        matches = np.loadtxt(slice_path + 'matches.txt')
        vitro_points = np.loadtxt(slice_path + 'true_points.txt')

        vitro_points = vitro_points[:, [0, 2, 1, 3]]

        output_path = self.args.dataset_path + 'automatch/'


        # Piecewise affine


        pc_return = piecewise_affine_cent(vitro_points, vivo_points, matches)

        for key in pc_return.keys():
            #check to see if the key is a dataframe
            if isinstance(pc_return[key], pd.DataFrame):
                pc_return[key].to_csv(output_path + 'pc_return_' + key + '.csv')


        # Nonrigid
        near_invivo_cents = pc_return['invivo_units'][np.where(np.isin(pc_return['invivo_units'][:, 3], 
                                                      pc_return['matching_probs']['StackUnit']))[0], :]
        aff_invitro_coords = pc_return['invitro_units_trans'].loc[:, ['3D_X', '3D_Y', '3D_Z', 'ID']]

        test_nonrigid = nonrigid_demon(aff_invitro_cent = aff_invitro_coords,
                                invivo_cent = pc_return['invivo_near_units'],
                                match_df = pc_return['manual_matches'],
                                invivo_l = pc_return['invivo_lookup'],
                                threshes = pc_return['matchprob_filts'],
                                pot_match_tab = pc_return['matching_probs'])

        #save all fields of test_nonrigid to file in output_path 
        for key in test_nonrigid.keys():
            if isinstance(test_nonrigid[key], pd.DataFrame):
                test_nonrigid[key].to_csv(output_path + 'test_nonrigid_' + key + '.csv', index=False)

        # Automatic mathces
        automatic_matches = automatch(test_nonrigid, vivo_points, vitro_points)

        automatic_matches.to_csv(output_path + 'automatic_matches.csv', index=False)

        #transformed_vitro_points = test_nonrigid['invitro_units']
        #extract columns 'SliceUnit' and 'StackUnit' from automatic_matches
        predicted_matches_id = automatic_matches.loc[:, ['SliceUnit', 'StackUnit']]
        #turn this into a numpy array
        predicted_matches = predicted_matches_id.to_numpy().astype(np.int32)
        print('Predicted ' + str(len(predicted_matches)) + ' struct matches.')
        predicted_matches = predicted_matches[np.where(np.isin(predicted_matches[:, 1], func_ids))[0], :]
        print('Predicted ' + str(len(predicted_matches)) + ' func matches.')

        np.savetxt(slice_path + 'predicted.txt', predicted_matches, fmt='%i')
        return predicted_matches_id.astype(np.int32)




'''        