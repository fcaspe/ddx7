import librosa
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
import hydra
from pathlib import Path
from functools import partial
from shutil import copyfile
import torch
import numpy as np
import h5py
from ddx7 import spectral_ops
import operator
import functools
import json
import os
from ddx7.core import _DB_RANGE

'''
URMP Data processor class adapted from HTP paper.
https://github.com/mosheman5/timbre_painting

'''

class ProcessData():
    def __init__(self, silence_thresh_dB, sr, device, seq_len,
                crepe_params, loudness_params,
                rms_params, hop_size, max_len, center,
                overlap = 0.0,
                debug = False,
                contiguous = False,
                contiguous_clip_noise = False):
        super().__init__()
        self.silence_thresh_dB = silence_thresh_dB
        self.crepe_params = crepe_params
        self.sr = sr
        self.device = torch.device(device)
        self.seq_len = seq_len
        self.loudness_params = loudness_params
        self.rms = rms_params
        self.max_len = max_len
        self.hop_size = hop_size
        self.feat_size = self.max_len*self.sr //self.hop_size
        self.audio_size = self.max_len*self.sr
        self.center = center
        self.overlap = overlap
        self.debug = debug
        self.contiguous = contiguous
        self.contiguous_clip_noise = contiguous_clip_noise

    def set_confidence(self,confidence):
        self.crepe_params.confidence_threshold = confidence

    def process_indices(self, indices: list) -> list:
        # Length in samples.
        max_len = self.max_len * self.sr

        def expand_long(indices_tuple: tuple) -> list:
            if indices_tuple[1] - indices_tuple[0] > max_len:
                ret = [(start, start+max_len) for start in np.arange(indices_tuple[0], indices_tuple[1] - max_len, max_len)]
                ret.append((ret[-1][-1], min(ret[-1][-1] + max_len, indices_tuple[1])))
                return ret
            else:
                return [indices_tuple]

        new_indices = [*map(expand_long, indices)]
        new_indices = functools.reduce(operator.concat, new_indices, [])
        new_indices = [x for x in new_indices if (x[1] - x[0] > self.seq_len * self.sr)]
        return new_indices

    def pad_to_expected_size(self,features,expected_size,pad_value):

        #Pad to next integer division if we are processing a whole file in one go.
        if(self.contiguous == True):
            # Pad up to next integer division
            pad_len = (features.shape[-1] // expected_size + 1)*expected_size - features.shape[-1]
            #print(f'feat len {features.shape[-1]} expected {expected_size} pad {pad_len}')
            features = np.pad(features,(0,pad_len),'constant',constant_values=pad_value)
            return features
        else:
            if(self.debug):
                print("Feat shape {} - expected size: {}".format(features.shape[-1],expected_size))
            if(features.shape[-1] < expected_size):
                pad_len = expected_size - features.shape[-1]
                features = np.pad(features,(0,pad_len),'constant',constant_values=pad_value)
            if(features.shape[-1] > expected_size):
                raise Exception('Expected size is smaller than current value')
        return features


    def extract_f0(self, audio):
        (f0,confidence) = spectral_ops.calc_f0(audio,
                                rate=self.sr,
                                hop_size=self.hop_size,
                                fmin=self.crepe_params.fmin,
                                fmax=self.crepe_params.fmax,
                                model=self.crepe_params.model,
                                batch_size=self.crepe_params.batch_size,
                                device=self.device,
                                center=self.center)

        if confidence.mean() < self.crepe_params.confidence_threshold:
            #print("Low confidence: {}".format(confidence.mean()))
            raise ValueError('Low f0 confidence')

        f0 = self.pad_to_expected_size(f0,
                expected_size = self.feat_size,
                pad_value=0)

        return f0

    def calc_loudness(self,audio):
        loudness = spectral_ops.calc_loudness(audio, rate=self.sr,
                                            n_fft=self.loudness_params.nfft,
                                            hop_size=self.hop_size,
                                            center=self.center,)

        loudness = self.pad_to_expected_size(loudness,
                expected_size = self.feat_size,
                pad_value=-_DB_RANGE)
        return loudness

    # TODO: Add center padding capability here.
    def calc_rms(self,audio):
        rms = spectral_ops.calc_power(audio, frame_size=self.rms.frame_size,
                                        hop_size=self.hop_size,pad_end=True)
        rms = self.pad_to_expected_size(rms,
                expected_size = self.feat_size,
                pad_value=-_DB_RANGE)
        return rms

    def save_data(self, audio, f0, loudness, rms, h5f, counter):
        h5f.create_dataset(f'{counter}_audio', data=audio)
        h5f.create_dataset(f'{counter}_f0', data=f0)
        h5f.create_dataset(f'{counter}_loudness', data=loudness)
        h5f.create_dataset(f'{counter}_rms', data=rms)
        return counter + 1

    def init_h5(self, data_dir):
        return h5py.File(data_dir / f'{self.sr}.h5', 'w')

    def close_h5(self, h5f):
        h5f.close()

    '''
    Main audio processing function
    '''
    def run_on_files(self, data_dir, input_dir, output_dir):
        audio_files = list((input_dir/data_dir).glob('*.wav'))
        output_dir = output_dir / data_dir
        output_dir.mkdir(exist_ok=True)

        # Open container
        h5f = self.init_h5(output_dir)
        counter = 0

        for audio_file in tqdm(audio_files):
            if(self.debug): print("Processing: {}".format(audio_file))

            # load and split files
            data, sr = librosa.load(audio_file.as_posix(), sr=self.sr)
            data = librosa.util.normalize(data) # Peak-normalize audio
            sounds_indices = []
            if(self.contiguous):
                sounds_indices.append([0,len(data)])
            else:
                sounds_indices = librosa.effects.split(data, top_db=self.silence_thresh_dB)
                #print("[DEBUG] Sound indices {}".format(sounds_indices))
                sounds_indices = self.process_indices(sounds_indices)
            if len(sounds_indices) == 0:
                continue


            for indices in sounds_indices:
                audio = data[indices[0]:indices[1]]
                if(self.debug): print("\tIndexes: {} {} - len: {}".format(indices[0],indices[1],indices[1]-indices[0]))

                # Feature retrieval segment

                try: # Only process audio with enough CREPE confidence
                    f0 = self.extract_f0(audio)
                except ValueError:
                    continue

                # Further downsamples the audio back to the other specified sample rates and returns a dictionary.
                loudness = self.calc_loudness(audio)
                rms = self.calc_rms(audio)
                if(self.contiguous):
                    if(self.contiguous_clip_noise):
                        if(self.debug): print("[DEBUG] clipping noise")
                        clip_pos = (f0 > 1900.0)
                        loudness[clip_pos] = -_DB_RANGE
                    audio = self.pad_to_expected_size(audio,f0.shape[0]*self.hop_size,0)

                else:
                    audio = self.pad_to_expected_size(audio,self.audio_size,0)
                if(self.debug): print(f'\t Store block {counter}: f0 : {f0.shape} - loudness : {loudness.shape} - rms {rms.shape} - audio : {audio.shape}')
                counter = self.save_data(audio, f0, loudness, rms, h5f, counter)

        # Finished storing f0 and loudness
        self.close_h5(h5f)


    def run_on_dirs(self, input_dir: Path, output_dir: Path):
        #print("Starting with crepe confidence: {}".format(self.crepe_params.confidence_threshold))
        folders = [x for x in input_dir.glob('./*') if x.is_dir()]
        for folder in tqdm(folders):
            self.run_on_files(folder.name, input_dir, output_dir)


def create_mono_urmp(instrument_key, audio_files, target_dir, instruments_dict):
    target_dir = target_dir / instruments_dict[instrument_key]
    if not target_dir.exists():
        target_dir.mkdir()
    cur_audio_files = [audio_file for audio_file in audio_files if f'_{instrument_key}_' in audio_file.name]
    [copyfile(audio_file, target_dir / audio_file.name) for audio_file in cur_audio_files]

def create_mono_testset(audio_files, target_dir, instrument):

    target_dir = target_dir / instrument
    if not target_dir.exists():
        target_dir.mkdir()
    cur_audio_files = [audio_file for audio_file in audio_files]
    [copyfile(audio_file, target_dir / audio_file.name) for audio_file in cur_audio_files]


def make_testset(args):
    CWD = Path(hydra.utils.get_original_cwd()) # Get current directory
    os.chdir(CWD)

    if args.testset is not None:

        if(args.skip_copy is False):

            # Create directories if needed
            dirs = args.testset.input_dir.split("/")
            target_dir = CWD
            for d in dirs:
                target_dir = target_dir / d
                #print(target_dir)
                target_dir.mkdir(exist_ok=True)

            testset_path = CWD / args.testset.source_folder
            print("[INFO] Testset source path: {}".format(testset_path))
            print("[INFO] will source files from directories: {}".format(args.testset.instruments))

            for instrument in args.testset.instruments:
                #print(instrument)
                # Find relevant audio files.
                test_audio_path = testset_path / instrument
                test_audio_files = list(test_audio_path.glob(f'./*.wav'))
                #print(test_audio_files)

                create_mono_testset(audio_files=test_audio_files,
                                    target_dir=target_dir,
                                    instrument=instrument)

    if(args.skip_process is False):

        # Create output dirs if needed
        dirs = args.testset.output_dir.split("/")
        target_dir = CWD
        for d in dirs:
            target_dir = target_dir / d
            #print(target_dir)
            target_dir.mkdir(exist_ok=True)

        # Process Test Set

        data_processor = hydra.utils.instantiate(args.data_processor)

        #Override original crepe confidence to process all the testset file.
        data_processor.set_confidence(0.0)
        data_processor.contiguous = args.testset.contiguous
        data_processor.contiguous_clip_noise = args.testset.clip_noise #Clip frequencies tracked due to noise.

        data_processor.run_on_dirs(CWD / args.testset.input_dir,
                                CWD / args.testset.output_dir)
    return

def make_urmp(args):
 # Phase 0 - copy all urmp wavs to corresponding folders
    CWD = Path(hydra.utils.get_original_cwd()) # Get current directory
    os.chdir(CWD)

    if args.urmp is not None:
        urmp_path = CWD / args.urmp.source_folder

        if(args.skip_copy is False):

            # Create directories if needed
            dirs = args.urmp.input_dir.split("/")
            target_dir = CWD
            for d in dirs:
                target_dir = target_dir / d
                #print(target_dir)
                target_dir.mkdir(exist_ok=True)

            # Find relevant audio files.
            urmp_audio_files = list(urmp_path.glob(f'./*/{args.urmp.mono_regex}*.wav'))

            print("[INFO] URMP Path: {}".format(urmp_path))

            print("[INFO] Number of files: {}".format(len(urmp_audio_files)))

            print(args.urmp.instruments.keys())
            # Partial function with instruments pre-configured for processing.
            create_mono_urmp_partial = partial(create_mono_urmp,
                                            audio_files=urmp_audio_files,
                                            target_dir=target_dir,
                                            instruments_dict=args.urmp.instruments)

            # Spawn threads to copy files.
            thread_map(create_mono_urmp_partial, list(args.urmp.instruments.keys()))

    # Process Train Set
    if(args.skip_process is False):

        # Create output dirs if needed
        dirs = args.urmp.output_dir.split("/")
        target_dir = CWD
        for d in dirs:
            target_dir = target_dir / d
            #print(target_dir)
            target_dir.mkdir(exist_ok=True)

        data_processor = hydra.utils.instantiate(args.data_processor)

        data_processor.run_on_dirs(CWD / args.urmp.input_dir,
                                CWD / args.urmp.output_dir)
    return

@hydra.main(config_path="./",config_name="data_config.yaml")
def main(args):

    if(args.process_testset is True): make_testset(args)

    if(args.process_urmp is True): make_urmp(args)

if __name__ == "__main__":
    main()