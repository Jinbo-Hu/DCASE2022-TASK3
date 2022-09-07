from pathlib import Path

class dacase2022_dask3:
    ''' DCASE 2022 Task 3 dataset

    '''
    def __init__(self, root_dir, cfg, args):
        self.label_dic = {'Female speech, woman speaking': 0,
                            'Male speech, man speaking': 1,
                            'Clapping': 2,
                            'Telephone': 3,
                            'Laughter': 4,
                            'Domestic sounds': 5,
                            'Walk, footsteps': 6,
                            'Door, open or close': 7,
                            'Music': 8,
                            'Musical instrument': 9,
                            'Water tap, faucet': 10,
                            'Bell': 11,
                            'Knock': 12 }

        self.label_resolution = 0.1 # 0.1s is the label resolution
        self.max_ov = 3 # max overlap
        self.num_classes = len(self.label_dic)
        self.root_dir = Path(root_dir)
        self.starss22_dir = self.root_dir.joinpath('STARSS22')
        self.synth_dir = self.root_dir.joinpath('synth_dataset')
        self.dataset_dir = dict()
        self.dataset_dir['dev'] = {'foa': dict(), 'mic': dict(), 'meta': dict()}
        self.dataset_dir['dev']['foa']['STARSS22'] = \
            [self.starss22_dir.joinpath('foa_dev').joinpath('dev-train-sony'),
             self.starss22_dir.joinpath('foa_dev').joinpath('dev-train-tau'),
             self.starss22_dir.joinpath('foa_dev').joinpath('dev-test-sony'),
             self.starss22_dir.joinpath('foa_dev').joinpath('dev-test-tau')]
        self.dataset_dir['dev']['mic']['STARSS22'] = \
            [self.starss22_dir.joinpath('mic_dev').joinpath('dev-train-sony'),
             self.starss22_dir.joinpath('mic_dev').joinpath('dev-train-tau'),
             self.starss22_dir.joinpath('mic_dev').joinpath('dev-test-sony'),
             self.starss22_dir.joinpath('mic_dev').joinpath('dev-test-tau')]
        self.dataset_dir['dev']['meta']['STARSS22'] = \
            [self.starss22_dir.joinpath('metadata_dev').joinpath('dev-train-sony'),
             self.starss22_dir.joinpath('metadata_dev').joinpath('dev-train-tau'),
             self.starss22_dir.joinpath('metadata_dev').joinpath('dev-test-sony'),
             self.starss22_dir.joinpath('metadata_dev').joinpath('dev-test-tau')]
        self.dataset_dir['eval'] = {
            'foa':  { 'STARSS22': [self.starss22_dir.joinpath('foa_eval')] }, 
            'mic':  { 'STARSS22': [self.starss22_dir.joinpath('mic_eval')] },
            'meta': { 'STARSS22': [] },
            }

        if not args.dataset == 'STARSS22':
            synth_dataset_list = args.dataset.split(',')
            for _synth_dataset in synth_dataset_list:
                self.dataset_dir['dev']['foa'][_synth_dataset] = [self.synth_dir.joinpath(_synth_dataset).joinpath('foa')]
                self.dataset_dir['dev']['mic'][_synth_dataset] = [self.synth_dir.joinpath(_synth_dataset).joinpath('mic')]
                self.dataset_dir['dev']['meta'][_synth_dataset] = [self.synth_dir.joinpath(_synth_dataset).joinpath('metadata')]
            
        



