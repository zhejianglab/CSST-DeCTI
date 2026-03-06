import os
from torch.utils.data import DataLoader

from decti import ColumnImagePairDataset, init_manager


def main():

    root = os.path.dirname(os.path.abspath(__file__))
    input_paths = [os.path.join(root, "data", "csst_1_cti.fits.gz")]
    target_patsh = [os.path.join(root, "data", "csst_1_nocti.fits.gz")]

    data_manager = init_manager('csst_msc_sim')
    dataset = ColumnImagePairDataset(input_paths, target_patsh, data_manager)

    # single sample
    col1, col2, meta = dataset[0]
    print(f"single sample: col1={col1.shape}, col2={col2.shape}, meta={meta}")

    # make DataLoader
    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
    )

    # circle for training
    for batch_idx, (batch_col1, batch_col2, meta) in enumerate(dataloader):
        # batch_col1, batch_col2: (batch_size, length)
        print(batch_idx, batch_col1.shape, batch_col2.shape, meta)
        if batch_idx >= 3:
            break


if __name__ == '__main__':
    main()
