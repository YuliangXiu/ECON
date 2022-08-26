import argparse
from tqdm import tqdm
from lib.common.config import get_cfg_defaults
from lib.dataset.PIFuDataset import PIFuDataset

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--show',
                        action='store_true',
                        help='vis sampler 3D')
    parser.add_argument('-s',
                        '--speed',
                        action='store_true',
                        help='vis sampler 3D')
    parser.add_argument('-l',
                        '--list',
                        action='store_true',
                        help='vis sampler 3D')
    parser.add_argument('-c',
                        '--config',
                        default='./configs/train/icon-filter.yaml',
                        help='vis sampler 3D')
    parser.add_argument('-d',
                        '--dataset',
                        default='thuman2')
    args_c = parser.parse_args()

    args = get_cfg_defaults()
    args.merge_from_file(args_c.config)
    
    if args_c.dataset == 'cape':
        
        # for cape test set
        cfg_test_mode = [
                "test_mode",
                True,
                "dataset.types",
                ["cape"],
                "dataset.scales",
                [100.0],
                "dataset.rotation_num",
                3
            ]
        args.merge_from_list(cfg_test_mode)

    # dataset sampler
    dataset = PIFuDataset(args, split='test', vis=args_c.show)
    print(f"Number of subjects :{len(dataset.subject_list)}")
    data_dict = dataset[1]

    if args_c.list:
        for k in data_dict.keys():
            if not hasattr(data_dict[k], "shape"):
                print(f"{k}: {data_dict[k]}")
            else:
                print(f"{k}: {data_dict[k].shape}")

    if args_c.show:
        # for item in dataset:
        item = dataset[0]
        # dataset.visualize_sampling3D(item, mode='cmap')
        dataset.visualize_sampling3D(item, mode='occ')
        # dataset.visualize_sampling3D(item, mode='normal')
        # dataset.visualize_sampling3D(item, mode='sdf')
        # dataset.visualize_sampling3D(item, mode='vis')

    if args_c.speed:
        # original: 2 it/s
        # smpl online compute: 2 it/s
        # normal online compute: 1.5 it/s
        import cProfile, pstats
        prof = cProfile.Profile()
        prof.enable()
        step = 0
        for item in tqdm(dataset):
            step += 1
            if step > 3:
                prof.disable()
                break
            # for k in item.keys():
            #     if 'voxel' in k:
            #         if not hasattr(item[k], "shape"):
            #             print(f"{k}: {item[k]}")
            #         else:
            #             print(f"{k}: {item[k].shape}")
            # print("--------------------")
        prof.dump_stats("/tmp/prof_log")
        
        p = pstats.Stats("/tmp/prof_log")
        p.sort_stats(pstats.SortKey.CUMULATIVE, pstats.SortKey.TIME)
        p.print_stats(30)