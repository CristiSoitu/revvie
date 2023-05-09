
from revvie.revvie_classes import *
from revvie.revvie_helpers import *
import warnings


## Program that implements revvie, a pipleine that matches in vivo and in vitro neurons.
'''
Assumes the following structure of the dataset:
    - dataset/
             /centroids/in_vitro_centroids.txt, in_vivo_centroids.txt
             /images/in_vitro_images.tif, in_vivo_images.tif
             revvie_config.toml     
'''

#dataset = 'BCM28382'
dataset = 'BCM27679_1_validation'
#dataset = 'BCM27679_1'
dataset_path = '/Users/soitu/Desktop/datasets/' + dataset + '/revvie/'

config_file = dataset_path + 'revvie_config.toml'


def main():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        args = ConfigReader(config_file)
        #args.matcher_name = create_matcher_profile()
        args.matcher_name = 'Anchor'
        args.dataset_path = dataset_path
        args.matcher_path = quick_dir(args.dataset_path, 'matcher_' + args.matcher_name)

    #    src_path = dataset_path + '../matching/matcher_' + args.matcher_name + '/'
    #    dst_path = args.matcher_path + 'slices/'
    #positions to run stats on: [19, 20, 22, Pos23, 28]
    #    slices = ['Pos19', 'Pos20', 'Pos22', 'Pos23', 'Pos28']
    #    slices = ['Pos19']
    #    slices = ['Pos19', 'Pos28']
    #    run_automatch_outside(args, slices)
    #    breakpoint()

    #    transfer_matches(src_path, dst_path)
    #    breakpoint()
    
    
    # load images and points
        vitro_points, vivo_points = load_latest_state(args)
#        breakpoint()
        
    #    vitro_images_list = ['blood_vessels.tif', 'somas_blood_vessels.tif', 'genes.tif', 'barcoded.tif']
    #    vitro_images_list = ['blood_vessels.tif', 'somas_blood_vessels.tif', 'genes.tif']
        vitro_images_list = ['blood_vessels.tif', 'somas_blood_vessels.tif']
    
        vitro_images_list = [args.dataset_path + 'images/' + image for image in vitro_images_list]
        #vivo_images_list = ['rotated_blood_vessels_stack.tif', 'rotated_gfp_stack.tif']
        vivo_images_list = ['rotated_blood_vessels_stack.tif']

        vivo_images_list = [args.dataset_path + 'images/' + image for image in vivo_images_list]
        
        # add extra points if needed
        vivo_struct_points = np.loadtxt(args.dataset_path + 'centroids/' + 'padded_struct_centroids.txt')
        vivo_struct_points[:, [1, 2]] = vivo_struct_points[:, [2, 1]]
        


        vitro_cloud_points = CloudPoints(xyz=vitro_points[:, 0:3], ids=vitro_points[:, 3], colors=vitro_points[:, 4:7], edge_colors=vitro_points[:, 7:10],
                                        size=args.vitro_point_size, opacity=args.opacity, edge_width=args.vitro_edge_size, symbol=args.symbol, ndim=3, name='points',
                                        matched_color=args.matched_color, unmatched_color=args.unmatched_color, accepted_edge_color=args.accepted_edge_color, rejected_edge_color=args.rejected_edge_color)
        vivo_cloud_points = CloudPoints(xyz=vivo_points[:, 0:3], ids=vivo_points[:, 3], colors=vivo_points[:, 4:7], edge_colors=vivo_points[:, 7:10],
                                        size=args.vivo_point_size, opacity=args.opacity, edge_width=args.vivo_edge_size, symbol=args.symbol, ndim=3, name='points',
                                        matched_color=args.matched_color, unmatched_color=args.unmatched_color, accepted_edge_color=args.accepted_edge_color, rejected_edge_color=args.rejected_edge_color)            
        
        vivo_struct_cloud_points = CloudPoints(xyz=vivo_struct_points[:, 0:3], ids=vivo_struct_points[:, 3], colors=args.struct_color, edge_colors=args.struct_edge_color,
                                        size=args.vivo_point_size, opacity=args.opacity, edge_width=args.vivo_edge_size, symbol=args.symbol, ndim=3, name='struct_points',
                                        matched_color=args.matched_color, unmatched_color=args.unmatched_color, accepted_edge_color=args.accepted_edge_color, rejected_edge_color=args.rejected_edge_color)
        
        # create revvie viewer
        revViewer = RevViewer(vitro_images_list, vivo_images_list, vitro_cloud_points, vivo_cloud_points, args)
        revViewer.add_cloud(revViewer.vivo_viewer, vivo_struct_cloud_points)

        
        revViewer.run()

main()
