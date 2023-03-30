import numpy as np
import napari
import tifffile as tif



class CloudPoints:
    def __init__(self, xyz, ids, colors, size, name, opacity, ndim, edge_colors, edge_width, symbol, matched_color, unmatched_color, accepted_edge_color, rejected_edge_color):
        self.xyz = xyz
        self.ids = ids
        self.colors = colors
        self.size = size
        self.slices = 'Pos' + str(self.xyz[:, 0])
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

    def get_id_by_index(self, index):
        return self.ids[index]

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
        return self.slices[index]
    
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

    def __init__(self, vitro_images_list, vivo_images_list, vitro_cloud_points, vivo_cloud_points):
        self.vitro_images_list = vitro_images_list
        self.vivo_images_list = vivo_images_list

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

        @self.vivo_viewer.bind_key('e')
        @self.vitro_viewer.bind_key('e')
        def select_matches(viewer):
            vitro_id, vivo_id = self.get_match_ids()
            self.record_match(vitro_id, vivo_id)
            self.update_match_display(vitro_id, vivo_id)


    def get_selected_index(self, viewer):
        return viewer.layers['points'].selected_data

    def get_match_ids(self):
        
        vitro_index = list(self.get_selected_index(self.vitro_viewer))
        vitro_id = self.vitro_cloud_points.get_id_by_index(vitro_index)
        vivo_index = list(self.get_selected_index(self.vivo_viewer))
        vivo_id = self.vivo_cloud_points.get_id_by_index(vivo_index)
        
        return vitro_id, vivo_id

    def record_match(self, vitro_id, vivo_id):


    def update_match_display(self, vitro_id, vivo_id):  
        self.vivo_cloud_points.set_color_by_id(vivo_id, self.vivo_cloud_points.matched_color)
        self.vitro_cloud_points.set_color_by_id(vitro_id, self.vitro_cloud_points.matched_color)

        self.vivo_cloud_points.set_edge_color_by_id(vivo_id, self.vivo_cloud_points.matched_color)
        self.vitro_cloud_points.set_edge_color_by_id(vitro_id, self.vitro_cloud_points.matched_color)

        self.vivo_viewer.layers[self.vivo_cloud_points.name].face_color = self.vivo_cloud_points.colors
        self.vitro_viewer.layers[self.vitro_cloud_points.name].face_color = self.vitro_cloud_points.colors

        self.vivo_viewer.layers[self.vivo_cloud_points.name].edge_color = self.vivo_cloud_points.edge_colors
        self.vitro_viewer.layers[self.vitro_cloud_points.name].edge_color = self.vitro_cloud_points.edge_colors
        

    def run(self):
        napari.run()


