import numpy as np
import napari
import tifffile as tif
from revvie.revvie_helpers import *
from revvie.delaunay_triangulation_nonrigid import *
import time
import threading



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


    def get_slice_by_index(self, index):
        return 'Pos' + str(int(self.xyz[index, 0]))
    
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
        self.vitro_images_list = vitro_images_list
        self.vivo_images_list = vivo_images_list
        self.args = args

        vitro_images = [tif.imread(image_path) for image_path in vitro_images_list]
        vitro_stack = np.stack(vitro_images, axis=0)

        vivo_images = [tif.imread(image_path) for image_path in vivo_images_list]
        vivo_stack = np.stack(vivo_images, axis=0)


        self.vitro_viewer = napari.Viewer(title='Vitro Viewer')
        self.vivo_viewer = napari.Viewer(title='Vivo Viewer')


        self.vitro_viewer.add_image(vitro_stack, name='images')
        self.vivo_viewer.add_image(vivo_stack, name='images')   


        self.vitro_cloud_points = vitro_cloud_points
        self.vivo_cloud_points = vivo_cloud_points

        self.vitro_viewer.add_points(vitro_cloud_points.xyz, face_color=vitro_cloud_points.colors, size=vitro_cloud_points.size, name=vitro_cloud_points.name, opacity=vitro_cloud_points.opacity, ndim=vitro_cloud_points.ndim, symbol=vitro_cloud_points.symbol, edge_color=vitro_cloud_points.edge_colors, edge_width=vitro_cloud_points.edge_width)
        self.vivo_viewer.add_points(vivo_cloud_points.xyz, face_color=vivo_cloud_points.colors, size=vivo_cloud_points.size, name=vivo_cloud_points.name, opacity=vivo_cloud_points.opacity, ndim=vivo_cloud_points.ndim, symbol=vivo_cloud_points.symbol, edge_color=vivo_cloud_points.edge_colors, edge_width=vivo_cloud_points.edge_width)


        #add a shapes layer to the vivo and vitro viewers. circles yellow. 
        self.vitro_viewer.add_shapes(name='shapes', shape_type='ellipse', edge_color='yellow', edge_width=2)
        self.vivo_viewer.add_shapes(name='shapes', shape_type='ellipse', edge_color='yellow', edge_width=2)

        self.vitro_viewer.layers['shapes'].add(data=[], shape_type='ellipse', edge_color='yellow', face_color='transparent', edge_width=2)
        self.vivo_viewer.layers['shapes'].add(data=[], shape_type='ellipse', edge_color='yellow', face_color='transparent', edge_width=2)





        @self.vivo_viewer.bind_key(self.args.toggle_visibility_key)
        @self.vitro_viewer.bind_key(self.args.toggle_visibility_key)
        def toggle_points_visibility(viewer):
            viewer.layers['points'].visible = not viewer.layers['points'].visible

        @self.vivo_viewer.bind_key(self.args.save_match_key)
        @self.vitro_viewer.bind_key(self.args.save_match_key)
        def match(viewer):
            vitro_id, vivo_id, vitro_index, vivo_index = self.get_match_ids()
            self.update_match_display(vitro_id, vivo_id, face_color=np.random.rand(1, 3), edge_color=self.args.matched_color)
            self.record_match(vitro_id, vivo_id, vitro_index, vivo_index)
            print('Match recorded for vitro id: ' + str(vitro_id) + ' and vivo id: ' + str(vivo_id))
                
        @self.vivo_viewer.bind_key(self.args.delete_match_key)
        @self.vitro_viewer.bind_key(self.args.delete_match_key)
        def delete_match(viewer):
            vitro_id, vivo_id, vitro_index, vivo_index = self.get_selected_ids()
            print('Deleting match for vitro id: ' + str(vitro_id))
            vitro_id, vivo_id = self.record_deletion(vitro_id, vivo_id, vitro_index, vivo_index)
            self.update_match_display(vitro_id, vivo_id, face_color=self.args.unmatched_color, edge_color=self.args.unmatched_color)

        @self.vivo_viewer.bind_key(self.args.save_state_key)
        @self.vitro_viewer.bind_key(self.args.save_state_key)
        def save_state(viewer):
            for position in args.positions:
                slice_path = self.args.matcher_path + 'slices/' + position + '/latest/'
                z = get_trailing_number(position)
                points = self.vitro_cloud_points.get_points_by_z(z)
                np.save(slice_path + 'displayed_points.npy', points)
                np.savetxt(slice_path + 'displayed_points.txt', points, fmt='%.1f')
            print('State saved.')


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


        @self.vivo_viewer.bind_key(self.args.run_alignment_key)
        @self.vitro_viewer.bind_key(self.args.run_alignment_key)
        def run_generate_predictions(viewer):
            t = threading.Thread(target=generate_predictions, args=(viewer, True,))
            t.start()

        def generate_predictions(viewer, run_bool=True):
            vitro_id, _, _, _ = self.get_selected_ids()
            slice = self.vitro_cloud_points.get_slice_by_id(vitro_id)
            print('Running automatch for slice: ' + slice)
            if run_bool:
                predicted_matches = self.run_automatch(slice)
            self.display_predicted_matches(slice)

        
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

           
        @self.vivo_viewer.bind_key(self.args.validate_prediction_key)
        @self.vitro_viewer.bind_key(self.args.validate_prediction_key)
        def validate_prediction(viewer):
            vitro_id, _, vitro_index, _ = self.get_selected_ids() 
            vivo_id = self.record_accepted_match(vitro_id, vitro_index)
            self.update_match_display(vitro_id, vivo_id, face_color=np.random.rand(1, 3), edge_color=self.args.matched_color)
            print('Prediction accepted for vitro id: ' + str(vitro_id) + ' and vivo id: ' + str(vivo_id))

        @self.vivo_viewer.bind_key(self.args.reject_prediction_key)
        @self.vitro_viewer.bind_key(self.args.reject_prediction_key)
        def reject_prediction(viewer):
            vitro_id, _, vitro_index, _ = self.get_selected_ids()
            vivo_id = self.record_rejected_match(vitro_id, vitro_index) 
            self.update_match_display(vitro_id, vivo_id, face_color=self.args.unmatched_color, edge_color=self.args.unmatched_color)
            print('Prediction rejected for vitro id: ' + str(vitro_id) + ' and vivo id: ' + str(vivo_id))


        @self.vivo_viewer.bind_key('x')
        @self.vitro_viewer.bind_key('x')
        def transfer_all_matches(viewer):
            for slice in args.positions:
                slice_path = args.matcher_path + 'slices/' + slice + '/latest/'
                matches = np.loadtxt(slice_path + 'matches.txt')
                vivo_matched_indices = np.loadtxt(slice_path + 'vivo_matched_indices.txt')
                vivo_matched_points = np.loadtxt(slice_path + 'vivo_matched_points.txt')
                #sometimes matches is just a (2, ) vector. if so, turn it into a (1,2). Also, sometimes matches is empty. if so, skip it.

                if matches.shape == (2,):
                    matches = np.array([matches])
                
                print(len(matches))
                print(matches.shape)

                
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
                print(len(matches), ' matches transferred for slice: ' + slice)
                np.savetxt(slice_path + 'vivo_matched_indices.txt', vivo_matched_indices, fmt='%i')
                np.savetxt(slice_path + 'vivo_matched_points.txt', vivo_matched_points, fmt='%.1f')

        @self.vivo_viewer.bind_key(self.args.display_pair_key)
        @self.vitro_viewer.bind_key(self.args.display_pair_key)
        def take_me_there(viewer):
            t = threading.Thread(target=take_me_there_function, args=(viewer,))
            t.start()

        def take_me_there_function(viewer):
            vitro_id, _, _, _, = self.get_selected_ids() 
            if vitro_id is not None:
                slice = self.vitro_cloud_points.get_slice_by_id(vitro_id)
                slice_path = args.matcher_path + 'slices/' + slice + '/latest/'
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

                vitro_size = args.vitro_point_size + 5
                vivo_size = args.vivo_point_size + 5
                vitro_box = np.array([[vitro_point_coords[1] - vitro_size, vitro_point_coords[2] - vitro_size], [vitro_point_coords[1] + vitro_size, vitro_point_coords[2] - vitro_size], [vitro_point_coords[1] + vitro_size, vitro_point_coords[2] + vitro_size], [vitro_point_coords[1] - vitro_size, vitro_point_coords[2] + vitro_size]])
                vivo_box = np.array([[vivo_coords[0] - vivo_size, vivo_coords[1] - vivo_size], [vivo_coords[0] + vivo_size, vivo_coords[1] - vivo_size], [vivo_coords[0] + vivo_size, vivo_coords[1] + vivo_size], [vivo_coords[0] - vivo_size, vivo_coords[1] + vivo_size]])
                    
                self.vitro_viewer.layers['shapes'].data = vitro_box
                self.vivo_viewer.layers['shapes'].data = vivo_box
                time.sleep(3)
                self.vitro_viewer.layers['shapes'].data = []
                self.vivo_viewer.layers['shapes'].data = []








            else:
                print('No vitro id selected')

    def display_predicted_matches(self, slice):
        slice_path = self.args.matcher_path + 'slices/' + slice + '/latest/'
        predicted_matches = np.loadtxt(slice_path + 'predicted.txt')
        # check to see if predicted matches is not empty
        if len(predicted_matches) > 0:  
            matches = np.loadtxt(slice_path + 'matches.txt')
            for match in predicted_matches:
                if match not in matches:
                    vitro_id = match[0]
                    vivo_id = match[1]
                    random_color = np.random.rand(1, 3)
                    self.update_match_display(vitro_id, vivo_id, face_color=random_color, edge_color=self.args.predicted_edge_color)
            #save only the predicted matches not in matches
            predicted_matches = predicted_matches[~np.isin(predicted_matches, matches).all(axis=1)]
            np.savetxt(slice_path + 'predicted.txt', predicted_matches, fmt='%i')

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



    def add_cloud(self, viewer, cloud):
        viewer.add_points(cloud.xyz, face_color='red', size=cloud.size, name=cloud.name, opacity=cloud.opacity, ndim=cloud.ndim, symbol=cloud.symbol, edge_color='red', edge_width=cloud.edge_width)




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

    def clear_selection(self):
        self.vitro_viewer.layers['points'].selected_data = []
        self.vivo_viewer.layers['points'].selected_data = []

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



    def update_match_display(self, vitro_id, vivo_id, face_color, edge_color):

        self.vivo_cloud_points.set_color_by_id(vivo_id, face_color)
        self.vitro_cloud_points.set_color_by_id(vitro_id, face_color)

        self.vivo_cloud_points.set_edge_color_by_id(vivo_id, edge_color)
        self.vitro_cloud_points.set_edge_color_by_id(vitro_id, edge_color)

        self.vivo_viewer.layers[self.vivo_cloud_points.name].face_color = self.vivo_cloud_points.colors
        self.vitro_viewer.layers[self.vitro_cloud_points.name].face_color = self.vitro_cloud_points.colors

        self.vivo_viewer.layers[self.vivo_cloud_points.name].edge_color = self.vivo_cloud_points.edge_colors
        self.vitro_viewer.layers[self.vitro_cloud_points.name].edge_color = self.vitro_cloud_points.edge_colors




    def run(self):
        napari.run()


def run_automatch_outside(args, slices=[]):
    if len(slices) == 0:
        slices = args.positions
    
    stack_path = args.matcher_path + 'stack/'
    vivo_func_points = np.load(stack_path + 'vivo_points.npy').astype(np.float32)
    func_ids = vivo_func_points[:, 3]

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
        if len(matches) > 3:
            vitro_points = np.loadtxt(slice_path + 'true_points.txt')

            vitro_points = vitro_points[:, [0, 2, 1, 3]]

            output_path = quick_dir(args.dataset_path, 'automatch')


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